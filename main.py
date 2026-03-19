"""
AI Citation Audit API v3
========================
Input: just a URL.
The tool auto-detects the brand, competitors, and content context from the page itself.
Works for any website, any industry, any brand.

Install:
    pip install fastapi uvicorn requests beautifulsoup4 playwright google-genai
    playwright install chromium

Run:
    export GEMINI_API_KEY=your_key
    python ai_audit.py

Audit a page:
    POST /audit
    {"url": "https://example.com/some-page"}
"""

import os
import json
import re
import time
import random
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Set, Tuple
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from urllib.parse import urlparse, urljoin
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

try:
    from playwright.async_api import async_playwright
except Exception:
    async_playwright = None

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ==================== CONFIG ====================

IS_PROD         = os.getenv("NODE_ENV") == "production"
DEBUG_LOGS      = os.getenv("DEBUG_LOGS", "false").lower() == "true"
REQUEST_TIMEOUT = 30
OUTPUT_DIR      = "./html_outputs"
METADATA_DIR    = "./metadata_outputs"
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.5-flash"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_playwright = None
_browser = None
_http_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _playwright, _browser, _http_client
    timeout = httpx.Timeout(REQUEST_TIMEOUT)
    _http_client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    if async_playwright is not None:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(headless=True)
    try:
        yield
    finally:
        if _http_client is not None:
            await _http_client.aclose()
        if _browser is not None:
            try:
                await _browser.close()
            except Exception as e:
                if DEBUG_LOGS: log(f"[LIFESPAN] Browser close error: {e}")
        if _playwright is not None:
            try:
                await _playwright.stop()
            except Exception as e:
                if DEBUG_LOGS: log(f"[LIFESPAN] Playwright stop error: {e}")

app = FastAPI(
    title="AI Citation Audit API",
    description="Audits any URL for AI citation readiness. Auto-detects brand and competitors.",
    version="3.0.0",
    lifespan=lifespan,
)

# ==================== CORS CONFIG ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== INPUT MODEL ====================

class AuditRequestModel(BaseModel):
    """URL is required. product_name is optional. user_id and product_id are for database tracking."""
    url: str = Field(..., description="The page URL to audit")
    product_name: Optional[str] = Field(None, description="Optional: override auto-detected brand/product name")
    user_id: Optional[str] = Field(None, description="Optional: userId for database tracking")
    product_id: Optional[str] = Field(None, description="Optional: productId for database tracking")

# ==================== MATRIX MODELS ====================

class StaticDynamicMatrixModel(BaseModel):
    static_word_count: int
    dynamic_word_count: int
    static_percentage: float
    dynamic_percentage: float
    aeo_score: float
    flags: List[str] = Field(default_factory=list)

class PageRenderabilityMatrixModel(BaseModel):
    """How visible is this page to AI crawlers that do not execute JavaScript."""
    builder_platform_detected: Optional[str] = None
    is_js_dependent: bool = False
    raw_html_word_count: int = 0
    rendered_word_count: int = 0
    content_loss_without_js_pct: float = 0.0
    crawlable_content_ratio: float = 0.0
    flags: List[str] = Field(default_factory=list)

class ReadinessMatrixModel(BaseModel):
    has_summary_block: bool
    summary_signals: List[str] = Field(default_factory=list)
    external_link_count: int
    authority_external_link_count: int
    authority_domains: List[str] = Field(default_factory=list)
    modularity_score: float
    flags: List[str] = Field(default_factory=list)

class StructuredDataMatrixModel(BaseModel):
    jsonld_blocks_found: int
    jsonld_types: List[str] = Field(default_factory=list)
    schema_score: float
    has_main_tag: bool
    has_article_tag: bool
    has_section_tags: bool
    metadata_present: Dict[str, bool] = Field(default_factory=dict)
    flags: List[str] = Field(default_factory=list)

class BrandProminenceMatrixModel(BaseModel):
    """How prominently the page surfaces its own brand — the most common AI citation failure."""
    detected_brand: Optional[str] = None
    detection_source: Optional[str] = None
    brand_mention_count: int = 0
    brand_in_title: bool = False
    brand_in_h1: bool = False
    brand_in_first_paragraph: bool = False
    brand_in_meta_description: bool = False
    brand_density_per_1000_words: float = 0.0
    brand_first_appearance_position_pct: float = 0.0
    flags: List[str] = Field(default_factory=list)

class CompetitiveContentMatrixModel(BaseModel):
    """Detects when a page promotes competitors more than its own brand."""
    detected_competitors: List[str] = Field(default_factory=list)
    brand_mention_count: int = 0
    competitor_total_mentions: int = 0
    brand_vs_competitor_ratio: float = 0.0
    brand_in_headings_count: int = 0
    competitor_heading_count: int = 0
    competitors_in_headings: List[str] = Field(default_factory=list)
    competitor_mention_breakdown: Dict[str, int] = Field(default_factory=dict)
    is_comparison_or_alternatives_article: bool = False
    content_works_against_brand: bool = False
    flags: List[str] = Field(default_factory=list)

class CitationWorthinessMatrixModel(BaseModel):
    """AI systems prioritise pages with unique, verifiable, quotable data."""
    numeric_stat_count: int = 0
    percentage_stat_count: int = 0
    has_original_research_signals: bool = False
    has_proprietary_data_signals: bool = False
    stat_density_per_1000_words: float = 0.0
    quotable_statements_count: int = 0
    has_comparison_table: bool = False
    table_count: int = 0
    has_numbered_insights: bool = False
    citation_score: float = 0.0
    flags: List[str] = Field(default_factory=list)

class ContentIntentMatrixModel(BaseModel):
    """Page intent determines how AI systems use this content."""
    detected_intent: str = "unknown"
    is_alternatives_article: bool = False
    is_competitor_comparison: bool = False
    is_brand_showcase: bool = False
    is_educational: bool = False
    is_transactional: bool = False
    title_frames_competitors: bool = False
    title_frames_brand: bool = False
    brand_sentence_ratio: float = 0.0
    intent_alignment_score: float = 0.0
    flags: List[str] = Field(default_factory=list)

class EEATMatrixModel(BaseModel):
    """Expertise, Experience, Authoritativeness, Trustworthiness signals."""
    has_author_attribution: bool = False
    has_publish_date: bool = False
    has_last_modified_date: bool = False
    has_about_or_company_link: bool = False
    has_external_citations: bool = False
    has_expert_quotes_or_attribution: bool = False
    has_author_credentials: bool = False
    trust_signals_found: List[str] = Field(default_factory=list)
    eeat_score: float = 0.0
    flags: List[str] = Field(default_factory=list)

class HeadingOwnershipMatrixModel(BaseModel):
    """Which entities dominate H1/H2/H3 — AI weights these heavily for topic association."""
    h1_text: Optional[str] = None
    all_h2_topics: List[str] = Field(default_factory=list)
    brand_owned_headings: List[str] = Field(default_factory=list)
    competitor_owned_headings: List[str] = Field(default_factory=list)
    neutral_headings: List[str] = Field(default_factory=list)
    brand_heading_ratio: float = 0.0
    competitor_heading_ratio: float = 0.0
    heading_structure_score: float = 0.0
    flags: List[str] = Field(default_factory=list)

# ==================== AI ANALYSIS MODELS ====================

class IssueModel(BaseModel):
    priority: str
    category: str
    issue: str
    impact: str
    fix: str

class QuickWinModel(BaseModel):
    effort: str
    action: str
    expected_impact: str

class AIAnalysisModel(BaseModel):
    overall_ai_citation_score: Optional[float] = None
    ai_citation_grade: Optional[str] = None
    executive_summary: Optional[str] = None
    critical_finding: Optional[str] = None
    issues: List[IssueModel] = Field(default_factory=list)
    quick_wins: List[QuickWinModel] = Field(default_factory=list)
    content_strategy_recommendations: List[str] = Field(default_factory=list)
    schema_recommendations: List[str] = Field(default_factory=list)
    competitive_positioning_recommendations: List[str] = Field(default_factory=list)

    critical_improvements: List[str] = Field(default_factory=list)

class FullAuditOutputModel(BaseModel):
    url: str
    audit_timestamp: str
    detected_brand: Optional[str] = None

    static_dynamic: StaticDynamicMatrixModel
    page_renderability: PageRenderabilityMatrixModel
    readiness: ReadinessMatrixModel
    structured_data: StructuredDataMatrixModel
    brand_prominence: BrandProminenceMatrixModel
    competitive_content: CompetitiveContentMatrixModel
    citation_worthiness: CitationWorthinessMatrixModel
    content_intent: ContentIntentMatrixModel
    eeat_signals: EEATMatrixModel
    heading_ownership: HeadingOwnershipMatrixModel

    ai_analysis: Optional[AIAnalysisModel] = None

# ==================== UTILITIES ====================

def log(msg: Any):
    if DEBUG_LOGS:
        print(str(msg))

def is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return bool(p.hostname) and p.scheme in ("http", "https")
    except Exception:
        return False

def normalize(text: str) -> str:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()

def wc(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def plain_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()
    return normalize(soup.get_text(" ", strip=True))

def filtered_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()
    return str(soup)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    reraise=True
)
async def fetch_raw_html(url: str) -> str:
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    ]
    hdrs = {
        "User-Agent": random.choice(agents),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized")
    r = await _http_client.get(url, headers=hdrs)
    r.raise_for_status()
    return r.text

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RuntimeError, TimeoutError)),
    reraise=True
)
async def fetch_rendered_html(url: str) -> str:
    if async_playwright is None:
        raise RuntimeError("Playwright not installed. Run: pip install playwright && playwright install chromium")
    if _browser is None:
        raise RuntimeError("Playwright browser not initialized")
    ctx = await _browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
    try:
        page = await ctx.new_page()
        await page.goto(url, wait_until="networkidle", timeout=REQUEST_TIMEOUT * 1000)
        return await page.content()
    finally:
        await ctx.close()

def save_json(data: Any, url: str, label: str):
    Path(METADATA_DIR).mkdir(parents=True, exist_ok=True)
    hostname = re.sub(r"[^a-z0-9]", "_", urlparse(url).hostname or "unknown", flags=re.IGNORECASE)
    ts = int(time.time() * 1000)
    fp = os.path.join(METADATA_DIR, f"{hostname}_{ts}_{label}.json")
    payload = data.model_dump() if hasattr(data, "model_dump") else (data.dict() if hasattr(data, "dict") else data)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log(f"[SAVED] {fp}")

def _write_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def _supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }

async def _supabase_insert(table: str, payload: Any, return_representation: bool = True) -> Any:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase not configured")
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = _supabase_headers()
    if return_representation:
        headers["Prefer"] = "return=representation"
    r = await _http_client.post(url, headers=headers, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase insert failed ({table}): {r.status_code} {r.text[:300]}")
    if not r.text:
        return None
    try:
        return r.json()
    except Exception:
        return r.text

async def upload_audit_to_supabase(result: Dict[str, Any], url: str, user_id: Optional[str] = None, product_id: Optional[str] = None) -> Optional[str]:
    def _get_flex(d: Dict[str, Any], key: str) -> Any:
        """Helper to get key from dict using underscores or spaces, case-insensitive."""
        if not d or not isinstance(d, dict): return None
        search_keys = [
            key, 
            key.replace("_", " "), 
            key.replace(" ", "_"),
            key.lower(),
            key.lower().replace("_", " "),
            key.capitalize().replace("_", " ")
        ]
        for sk in search_keys:
            if sk in d: return d[sk]
        return None

    hdr = _get_flex(result, "AUDIT_HEADER") or {}
    risk_data = _get_flex(result, "CCC_RISK_SCORE") or {}
    metrics = _get_flex(risk_data, "Supporting_metrics") or _get_flex(result, "Supporting_metrics") or {}
    static_dynamic = _get_flex(result, "static_dynamic") or {}

    site_summary = _get_flex(result, "site_summary") or {}
    dimension_scores = _get_flex(result, "dimension_scores") or {}
    llm_selection_story = _get_flex(result, "llm_selection_story")

    # Handle nested vs flat risk score/band/summary
    risk_score_val = ""
    if isinstance(risk_data, dict):
        score = _get_flex(risk_data, "score") or ""
        band = _get_flex(risk_data, "risk_band") or ""
        if score and band:
            risk_score_val = f"{score} / 100 — {band}"
        elif score:
            risk_score_val = str(score)
        else:
            risk_score_val = str(risk_data.get("score") or "")
    else:
        risk_score_val = str(risk_data)

    publishing_brand = _get_flex(hdr, "Publishing_brand") or _get_flex(site_summary, "title") or (result.get("detected_brand") if isinstance(result.get("detected_brand"), str) else "")
    summary_text = (
        _get_flex(risk_data, "summary")
        or _get_flex(result, "CCC_SUMMARY")
        or _get_flex(result, "SUMMARY")
        or _get_flex(result, "summary")
        or _get_flex(result, "overall_summary")
    )

    seo_audit_row: Dict[str, Any] = {
        "page_audited": _get_flex(hdr, "Page_audited") or url,
        "publishing_brand": publishing_brand,
        "primary_topic": _get_flex(hdr, "Primary_topic"),
        "competitors_identified": _get_flex(hdr, "Competitors_identified"),
        "audit_layers_processed": _safe_int(_get_flex(hdr, "Audit_layers_processed")),
        "ccc_risk_score": risk_score_val or str(result.get("CCC_RISK_SCORE") or ""),
        "summary": summary_text,
        "leak_vectors_found": _safe_int(_get_flex(metrics, "Leak_vectors_found")),
        "competitors_actively_amplified": _safe_int(_get_flex(metrics, "Competitors_actively_amplified")),
        "third_party_validations": _safe_int(_get_flex(metrics, "Third_party_validations_for_publishing_brand")),
        "static_word_count": _safe_int(_get_flex(static_dynamic, "static_word_count")),
        "dynamic_word_count": _safe_int(_get_flex(static_dynamic, "dynamic_word_count")),
        "aeo_score": _safe_float(_get_flex(static_dynamic, "aeo_score")),
        "user_id": user_id,
        "product_id": product_id,
        "url": url,
        "page_type": result.get("page_type"),
        "is_landing_page": result.get("is_landing_page", False),
        "overall_score": str(_get_flex(result, "overall_score") or ""),
        "site_summary": site_summary if site_summary else None,
        "dimension_scores": dimension_scores if dimension_scores else None,
        "llm_selection_story": llm_selection_story if llm_selection_story else None,
    }

    if not seo_audit_row.get("publishing_brand"):
        seo_audit_row["publishing_brand"] = "unknown"

    inserted = await _supabase_insert("seo_audits", seo_audit_row, return_representation=True)
    audit_id: Optional[str] = None
    if isinstance(inserted, list) and inserted:
        audit_id = inserted[0].get("id")
    elif isinstance(inserted, dict):
        audit_id = inserted.get("id")
    if not audit_id:
        return None

    hypotheses = _get_flex(result, "HYPOTHESES") or _get_flex(result, "hypotheses")
    if isinstance(hypotheses, list) and hypotheses:
        rows: List[Dict[str, Any]] = []
        for idx, h in enumerate(hypotheses):
            if not isinstance(h, dict):
                continue

            hypothesis_code: Optional[str] = None
            title: Optional[str] = None
            for k, v in h.items():
                if isinstance(k, str) and re.fullmatch(r"H\d+", k.strip()):
                    hypothesis_code = k.strip()
                    if isinstance(v, str):
                        title = v
                    break

            if not title and isinstance(h.get("title"), str):
                title = h.get("title")

            if not hypothesis_code:
                hypothesis_code = f"H{h.get('id', idx + 1)}"

            # Handle both old 'evidence_from_input_data' and new 'signals' array
            evidence: Optional[Dict[str, Any]] = None
            if "signals" in h:
                evidence = {"signals": h.get("signals")}
            elif "Evidence_from_input_data" in h:
                evidence = {"Evidence_from_input_data": h.get("Evidence_from_input_data")}
            elif "evidence_from_input_data" in h:
                evidence = {"evidence_from_input_data": h.get("evidence_from_input_data")}

            rows.append(
                {
                    "audit_id": audit_id,
                    "hypothesis_code": hypothesis_code,
                    "title": title,
                    "confidence": str(h.get("Confidence") or h.get("confidence") or ""),
                    "impact": h.get("Impact") or h.get("impact") or h.get("tier"),
                    "category": h.get("Category") or h.get("category"),
                    "mechanism": h.get("Mechanism") or h.get("mechanism") or h.get("mechanism_explanation"),
                    "recommended_fix": h.get("Recommended_fix") or h.get("recommended_fix"),
                    "evidence": evidence,
                }
            )

        rows = [
            r for r in rows
            if any(
                r.get(k)
                for k in [
                    "title",
                    "confidence",
                    "impact",
                    "category",
                    "mechanism",
                    "recommended_fix",
                    "evidence",
                ]
            )
        ]

        if rows:
            await _supabase_insert("audit_hypotheses", rows, return_representation=False)

    roadmap = _get_flex(result, "PRIORITY_FIX_ROADMAP")
    if isinstance(roadmap, list) and roadmap:
        rows2: List[Dict[str, Any]] = []
        for idx, step in enumerate(roadmap):
            if not isinstance(step, dict) or not step:
                continue
            step_title: Optional[str] = None
            step_desc: Optional[str] = None
            step_order: Optional[int] = None

            if isinstance(step.get("title"), str) and isinstance(step.get("description"), str):
                step_title = step.get("title")
                step_desc = step.get("description")
                step_order = idx + 1
            else:
                k = next(iter(step.keys()))
                v = step.get(k)
                if not isinstance(k, str):
                    continue
                step_title = k
                step_desc = v if isinstance(v, str) else (json.dumps(v) if v is not None else None)
                m = re.match(r"^(\d+)", k)
                if m:
                    step_order = _safe_int(m.group(1))
            rows2.append(
                {
                    "audit_id": audit_id,
                    "step_order": step_order,
                    "title": step_title,
                    "description": step_desc,
                }
            )

        rows2 = [r for r in rows2 if r.get("title") or r.get("description")]

        if rows2:
            await _supabase_insert("audit_roadmap_steps", rows2, return_representation=False)

    return audit_id

# ==================== BRAND AUTO-DETECTION ====================

def detect_brand(soup: BeautifulSoup, page_url: str) -> Tuple[Optional[str], str]:
    """
    Infers the brand name from the page using multiple signals in priority order:
    1. JSON-LD Organization / WebSite / Corporation name
    2. og:site_name meta tag
    3. application-name meta tag
    4. Title tag suffix (part after the last separator like | or -)
    5. Domain root as last resort
    """
    # 1. JSON-LD
    for script in soup.find_all("script"):
        raw_type = (script.get("type") or "").strip().lower() if hasattr(script, "get") else ""
        if raw_type not in ("application/ld+json", "application/json+ld"):
            continue
        try:
            data = json.loads(script.string or "")

            def _org_name(obj: Any) -> Optional[str]:
                if isinstance(obj, dict):
                    if obj.get("@type") in ("Organization", "WebSite", "Corporation", "LocalBusiness", "SoftwareApplication"):
                        n = obj.get("name", "")
                        if isinstance(n, str) and 2 < len(n.strip()) < 60:
                            return n.strip()
                    for v in obj.values():
                        r = _org_name(v)
                        if r:
                            return r
                elif isinstance(obj, list):
                    for i in obj:
                        r = _org_name(i)
                        if r:
                            return r
                return None

            name = _org_name(data)
            if name:
                return name, "json_ld_organization"
        except Exception:
            continue

    # 2. og:site_name
    tag = soup.find("meta", property="og:site_name")
    if tag:
        val = (tag.get("content") or "").strip()
        if 2 < len(val) < 60:
            return val, "og_site_name"

    # 3. application-name
    tag = soup.find("meta", attrs={"name": "application-name"})
    if tag:
        val = (tag.get("content") or "").strip()
        if 2 < len(val) < 60:
            return val, "meta_application_name"

    # 4. Title suffix
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text().strip()
        for sep in ["|", "–", "—", "·", " - "]:
            if sep in title:
                parts = [p.strip() for p in title.split(sep)]
                candidate = parts[-1]
                noise = ("top", "best", "how", "guide", "review", "2024", "2025", "2026", "free")
                if (2 < len(candidate) < 50 and not any(kw in candidate.lower() for kw in noise)):
                    return candidate, "title_suffix"

    # 5. Domain root
    hostname = re.sub(r"^www\.", "", urlparse(page_url).hostname or "")
    root = hostname.split(".")[0].capitalize()
    if len(root) > 2:
        return root, "domain_name"

    return None, "undetected"

# ==================== COMPETITOR AUTO-DETECTION ====================

def detect_competitors(
    soup: BeautifulSoup,
    text: str,
    brand: Optional[str],
    page_url: str,
) -> List[str]:
    """
    Discovers third-party brand/product names mentioned on the page.
    Uses three strategies:
    1. Numbered or named headings (most reliable for comparison/list articles)
    2. Comparison-keyword patterns ("vs X", "alternative to X")
    3. Frequently recurring proper nouns
    """
    found: List[str] = []
    brand_lower = brand.lower() if brand else ""
    seen: Set[str] = set()

    STOP_WORDS: Set[str] = {
        "The", "A", "An", "In", "On", "At", "For", "With", "And", "Or", "But",
        "Not", "It", "Is", "Are", "Was", "Be", "This", "That", "These", "Those",
        "Their", "They", "We", "Our", "You", "Your", "Which", "When", "Where",
        "How", "Why", "What", "Who", "Best", "Top", "New", "Free", "More",
        "Most", "All", "Some", "Many", "Each", "Every", "Any",
    }

    def _add(candidate: str):
        c = candidate.strip()
        c_lower = c.lower()
        if (c and 2 < len(c) < 40
                and c_lower not in seen
                and (not brand_lower or brand_lower not in c_lower)
                and c not in STOP_WORDS):
            seen.add(c_lower)
            found.append(c)

    # Strategy 1: numbered/titled headings
    heading_re = re.compile(
        r"^(?:#?\s*\d+[.\):]?\s+)?([A-Z][a-zA-Z0-9][\w\s\-\.]{1,28}?)(?:\s*[:–—|/]|\s*$)"
    )
    for tag in soup.find_all(["h2", "h3", "h4"]):
        m = heading_re.match(tag.get_text().strip())
        if m:
            _add(m.group(1).strip())

    # Strategy 2: comparison keywords
    cmp_re = re.compile(
        r"(?:vs\.?\s+|versus\s+|compared to\s+|alternative(?:s)? to\s+|instead of\s+)"
        r"([A-Z][a-zA-Z0-9\-\.]{2,25})",
        re.IGNORECASE
    )
    for m in cmp_re.finditer(text):
        _add(m.group(1))

    reverse_cmp_re = re.compile(
        r"([A-Z][a-zA-Z0-9\-\.]{2,25})\s+(?:alternative|vs\.?|competitor)",
        re.IGNORECASE
    )
    for m in reverse_cmp_re.finditer(text):
        _add(m.group(1))

    # Strategy 3: frequently recurring capitalised words/phrases
    freq: Dict[str, int] = {}
    for m in re.finditer(r"\b([A-Z][a-z]{1,15}(?:\s[A-Z][a-z]{1,15})?)\b", text):
        word = m.group(1)
        if word in STOP_WORDS or (brand_lower and brand_lower in word.lower()):
            continue
        freq[word] = freq.get(word, 0) + 1

    for word, cnt in freq.items():
        if cnt >= 3:
            _add(word)

    # Deduplicate keeping order, cap at 15
    deduped: List[str] = []
    seen2: Set[str] = set()
    for c in found:
        if c.lower() not in seen2:
            seen2.add(c.lower())
            deduped.append(c)
    return deduped[:15]

# ==================== MATRIX ANALYZERS ====================

def compute_static_dynamic(raw_wc: int, rendered_wc: int) -> StaticDynamicMatrixModel:
    rendered_wc = max(rendered_wc, 0)
    raw_wc = max(raw_wc, 0)
    if rendered_wc == 0:
        return StaticDynamicMatrixModel(
            static_word_count=raw_wc, dynamic_word_count=0,
            static_percentage=0.0, dynamic_percentage=0.0,
            aeo_score=0.0, flags=["rendered_text_empty"],
        )
    static_pct = (raw_wc / rendered_wc) * 100.0 if rendered_wc else 0.0
    static_pct = max(0.0, min(static_pct, 100.0))
    dynamic_pct = max(0.0, 100.0 - static_pct)

    # Simple crawlability proxy: how much of the rendered text is already present in raw HTML.
    # 100 = fully crawlable without JS, 0 = fully JS-dependent.
    aeo_score = static_pct

    flags: List[str] = []
    if static_pct < 25.0:
        flags.append("highly_js_dependent")
    elif static_pct < 60.0:
        flags.append("partially_js_dependent")

    return StaticDynamicMatrixModel(
        static_word_count=raw_wc,
        dynamic_word_count=max(rendered_wc - raw_wc, 0),
        static_percentage=round(static_pct, 2),
        dynamic_percentage=round(dynamic_pct, 2),
        aeo_score=round(aeo_score, 2),
        flags=flags,
    )

def clean_html_for_aeo(html_content: str) -> str:
    """
    Cleans HTML content for AEO/GEO (Answer Engine Optimization / Generative Engine Optimization) analysis
    by removing visual noise and preserving semantic structure and critical metadata.
    
    Preserves:
    - Structured data (JSON-LD, Microdata)
    - Semantic HTML (headings, paragraphs, lists, tables)
    - Important meta tags (description, article metadata, Open Graph)
    - Content hierarchy and structure
    - Alt text from images (for context)
    - Blockquotes and citations
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Capture JSON-LD scripts up-front to guarantee preservation even if later cleaning
    # steps unwrap or decompose surrounding structure.
    preserved_jsonld_scripts: List[str] = []
    for script in soup.find_all('script'):
        try:
            raw_type = script.get('type') if hasattr(script, 'get') else None
            script_type = raw_type.strip().lower() if isinstance(raw_type, str) else ''
            if script_type in ['application/ld+json', 'application/json+ld']:
                preserved_jsonld_scripts.append(str(script))
        except (AttributeError, TypeError):
            continue
    
    # 1. Remove visual/interactive elements that don't contribute to content understanding
    for tag in soup.find_all(['style', 'noscript', 'iframe', 'form', 'button', 'input', 'select', 'textarea']):
        tag.decompose()
    
    # 2. Remove SVG but keep semantic icons if they have aria-label or title
    for svg in soup.find_all('svg'):
        try:
            aria_label = svg.get('aria-label', '').strip() if svg.get('aria-label') else ''
            title_tag = svg.find('title')
            title_text = title_tag.get_text().strip() if title_tag else ''
            
            if aria_label or title_text:
                svg.replace_with(f' [Icon: {aria_label or title_text}] ')
            else:
                svg.decompose()
        except (AttributeError, TypeError):
            svg.decompose()
    
    # 3. Keep scripts ONLY if they are JSON-LD structured data (critical for SEO/AEO)
    # Strictly preserve: <script type="application/ld+json"> ... </script>
    for script in soup.find_all('script'):
        try:
            script_type = ''
            if hasattr(script, 'get'):
                raw_type = script.get('type')
                if isinstance(raw_type, str):
                    script_type = raw_type.strip().lower()
            if script_type not in ['application/ld+json', 'application/json+ld']:
                script.decompose()
        except (AttributeError, TypeError):
            script.decompose()
    
    # 4. Handle Images - Keep alt text for context, remove the actual image
    for img in soup.find_all('img'):
        try:
            alt_text = img.get('alt', '').strip() if img.get('alt') else ''
            title_text = img.get('title', '').strip() if img.get('title') else ''
            
            if alt_text or title_text:
                replacement = f' [Image: {alt_text or title_text}] '
                img.replace_with(replacement)
            else:
                img.decompose()
        except (AttributeError, TypeError):
            img.decompose()
    
    # 5. Remove navigational elements (they don't contribute to main content)
    for tag in soup.find_all(['nav', 'footer']):
        tag.decompose()
    
    # 6. Keep header but only if it contains important content (h1, h2, etc)
    for header in soup.find_all('header'):
        try:
            # If header contains important headings, keep it; otherwise remove
            if not header.find(['h1', 'h2', 'h3']):
                header.decompose()
        except (AttributeError, TypeError):
            header.decompose()
    
    # 7. Filter Meta tags - Keep only SEO/AEO relevant ones
    for meta in soup.find_all('meta'):
        try:
            name = meta.get('name', '').lower() if meta.get('name') else ''
            property_attr = meta.get('property', '').lower() if meta.get('property') else ''
            
            # Keep important meta tags
            important_names = ['description', 'keywords', 'author', 'robots', 'article:published_time', 
                              'article:modified_time', 'article:author', 'article:section', 'article:tag']
            important_properties = ['og:title', 'og:description', 'og:type', 'og:url', 'og:image',
                                   'og:site_name', 'article:published_time', 'article:modified_time',
                                   'article:author', 'article:section', 'article:tag']
            
            is_important = (
                name in important_names or
                property_attr in important_properties or
                property_attr.startswith('article:')
            )
            
            if not is_important:
                meta.decompose()
        except (AttributeError, TypeError):
            meta.decompose()
    
    # 8. Filter Link tags - Keep canonical, alternate, and structured data links
    for link in soup.find_all('link'):
        try:
            rel = link.get('rel', []) if hasattr(link, 'get') else []
            rel_str = ' '.join(rel).lower() if isinstance(rel, list) else str(rel).lower()
            
            keep_links = ['canonical', 'alternate', 'amphtml', 'prev', 'next']
            
            if not any(keep_rel in rel_str for keep_rel in keep_links):
                link.decompose()
        except (AttributeError, TypeError):
            link.decompose()
    
    # 9. Define valid semantic tags that are important for content structure and AEO
    valid_tags: Set[str] = {
        # Document structure
        'head', 'body',
        # Headings (critical for content hierarchy)
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        # Text content
        'p', 'span', 'div', 'section', 'article', 'main', 'aside',
        # Lists (important for structured content)
        'ul', 'ol', 'li', 'dl', 'dt', 'dd',
        # Tables (structured data presentation)
        'table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th', 'caption',
        # Semantic text elements
        'strong', 'em', 'b', 'i', 'mark', 'small', 'del', 'ins', 'sub', 'sup',
        'blockquote', 'cite', 'q', 'abbr', 'code', 'pre', 'kbd', 'samp', 'var',
        # Structural elements
        'address', 'time', 'figure', 'figcaption',
        # Meta elements
        'title', 'meta', 'script', 'link', 'base',
        # Definition and details
        'details', 'summary',
        # Header (if it contains h1-h3)
        'header'
    }
    
    # 10. Clean attributes but preserve semantic ones
    # Create a list of tags to process to avoid modification during iteration
    all_tags = list(soup.find_all())
    
    for tag in all_tags:
        try:
            # Skip if tag has been removed from tree or is None
            if tag is None or not hasattr(tag, 'name') or tag.name is None:
                continue
                
            tag_name = tag.name.lower() if tag.name else ''
            
            if not tag_name or tag_name not in valid_tags:
                # Unwrap non-semantic tags but keep their content
                if hasattr(tag, 'unwrap'):
                    tag.unwrap()
            else:
                # Define allowed attributes per tag type
                allowed_attrs: Set[str] = set()
                
                if tag_name == 'script':
                    allowed_attrs = {'type', 'id'}
                elif tag_name == 'meta':
                    allowed_attrs = {'name', 'property', 'content', 'charset', 'http-equiv'}
                elif tag_name == 'link':
                    allowed_attrs = {'rel', 'href', 'hreflang', 'type'}
                elif tag_name == 'time':
                    allowed_attrs = {'datetime'}
                elif tag_name in ['article', 'section', 'div']:
                    allowed_attrs = {'id', 'itemscope', 'itemtype', 'itemprop'}  # Keep microdata
                elif tag_name == 'a':
                    allowed_attrs = {'href', 'rel'}
                elif tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    allowed_attrs = {'id'}  # Keep IDs for anchor links
                
                # Remove non-allowed attributes - check if attrs exists
                if hasattr(tag, 'attrs') and tag.attrs is not None:
                    attrs_to_remove = [attr for attr in list(tag.attrs.keys()) if attr not in allowed_attrs]
                    for attr in attrs_to_remove:
                        try:
                            del tag.attrs[attr]
                        except (KeyError, AttributeError):
                            pass
        except (AttributeError, TypeError, ValueError) as e:
            # Skip problematic tags
            log(f'[HTML_CLEAN] Skipping tag due to error: {str(e)}')
            continue
    
    # Re-insert preserved JSON-LD scripts into <head> if they are missing after cleaning.
    # (Some documents get heavily unwrapped, and we still want structured data retained.)
    try:
        if preserved_jsonld_scripts:
            if soup.head is None:
                head_tag = soup.new_tag('head')
                soup.insert(0, head_tag)

            existing_jsonld_count = 0
            for s in soup.find_all('script'):
                raw_type = s.get('type') if hasattr(s, 'get') else None
                st = raw_type.strip().lower() if isinstance(raw_type, str) else ''
                if st in ['application/ld+json', 'application/json+ld']:
                    existing_jsonld_count += 1

            if existing_jsonld_count == 0:
                for script_html in preserved_jsonld_scripts:
                    frag = BeautifulSoup(script_html, 'html.parser')
                    script_tag = frag.find('script')
                    if script_tag is not None:
                        soup.head.append(script_tag)
    except Exception:
        pass
    
    # 11. Get the cleaned HTML
    text = str(soup)
    
    # 12. Clean up excessive whitespace while preserving structure
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    
    return '\n'.join(lines)

def analyze_page_renderability(raw_html: str, rendered_html: str, page_url: str) -> PageRenderabilityMatrixModel:
    platform_signals: Dict[str, List[str]] = {
        "Framer":      ["framer.com", "framerusercontent", "Made in Framer"],
        "Webflow":     ["webflow.com", "webflow.io", "wf-form"],
        "Next.js":     ["__NEXT_DATA__", "_next/static"],
        "Nuxt.js":     ["__NUXT__", "_nuxt/"],
        "Gatsby":      ["gatsby-"],
        "WordPress":   ["wp-content", "wp-includes"],
        "Squarespace": ["squarespace.com", "static.squarespace"],
        "Wix":         ["wix.com", "static.wixstatic"],
        "Shopify":     ["cdn.shopify.com"],
        "Ghost":       ["ghost.io", "content.ghost.org"],
        "HubSpot":     ["hubspot.com", "hs-scripts"],
    }
    detected_platform: Optional[str] = None
    for platform, sigs in platform_signals.items():
        if any(s in raw_html for s in sigs):
            detected_platform = platform
            break

    raw_wc = wc(plain_text(raw_html))
    rendered_wc = wc(plain_text(rendered_html))

    return PageRenderabilityMatrixModel(
        builder_platform_detected=detected_platform,
        is_js_dependent=False,
        raw_html_word_count=raw_wc,
        rendered_word_count=rendered_wc,
        content_loss_without_js_pct=0.0,
        crawlable_content_ratio=0.0,
        flags=[],
    )

def analyze_readiness(html: str, page_url: str) -> ReadinessMatrixModel:
    soup = BeautifulSoup(html or "", "html.parser")
    summary_signals: List[str] = []
    has_summary = False
    body = soup.body or soup

    candidates = body.find_all(["p", "ul", "ol"], limit=80)
    first_window = candidates[: max(1, int(len(candidates) * 0.2))]
    kws = ["summary", "tl;dr", "tldr", "key takeaways", "highlights", "in brief", "overview"]
    for el in first_window:
        txt = (el.get_text(" ", strip=True) or "").lower()
        for kw in kws:
            if kw in txt:
                summary_signals.append(kw)
                has_summary = True
                break
        if has_summary:
            break

    if not has_summary:
        fh = body.find("h1") or body.find("h2")
        if fh:
            nxt = fh.find_next_sibling()
            for _ in range(6):
                if not nxt:
                    break
                if getattr(nxt, "name", None) in ["ul", "ol"]:
                    has_summary = True
                    summary_signals.append("list_after_heading")
                    break
                nxt = nxt.find_next_sibling()

    base_domain = (urlparse(page_url).hostname or "").lower()
    authority_domains: Set[str] = set()
    ext_links = 0
    for a in body.find_all("a", href=True):
        abs_url = urljoin(page_url, a["href"])
        p = urlparse(abs_url)
        if p.scheme not in ("http", "https"):
            continue
        ld = (p.hostname or "").lower()
        if not ld or ld == base_domain or ld.endswith("." + base_domain):
            continue
        ext_links += 1
        if (any(ld.endswith(s) for s in [".gov", ".edu", ".ac.uk"]) or
                "wikipedia.org" in ld or
                any(ld.endswith(s) for s in [".nih.gov", ".who.int", ".nature.com", ".ieee.org"])):
            authority_domains.add(ld)

    p_count = len(body.find_all("p"))
    struct_count = sum(len(body.find_all(t)) for t in ["ul", "ol", "table", "blockquote"])

    return ReadinessMatrixModel(
        has_summary_block=has_summary,
        summary_signals=sorted(set(summary_signals)),
        external_link_count=ext_links,
        authority_external_link_count=len(authority_domains),
        authority_domains=sorted(authority_domains),
        modularity_score=0.0,
        flags=[],
    )

def analyze_structured_data(html: str, page_url: str) -> StructuredDataMatrixModel:
    soup = BeautifulSoup(html or "", "html.parser")
    jsonld_types: Set[str] = set()
    jsonld_blocks = 0

    for script in soup.find_all("script"):
        raw_type = (script.get("type") or "").strip().lower()
        if raw_type not in ("application/ld+json", "application/json+ld"):
            continue
        payload = script.string or script.get_text() or ""
        if not payload.strip():
            continue
        jsonld_blocks += 1
        try:
            data = json.loads(payload)
            def _collect(obj: Any):
                if isinstance(obj, dict):
                    t = obj.get("@type")
                    if isinstance(t, str) and t.strip():
                        jsonld_types.add(t.strip())
                    elif isinstance(t, list):
                        for i in t:
                            if isinstance(i, str):
                                jsonld_types.add(i.strip())
                    for v in obj.values():
                        _collect(v)
                elif isinstance(obj, list):
                    for i in obj:
                        _collect(i)
            _collect(data)
        except Exception:
            pass

    body = soup.body or soup

    return StructuredDataMatrixModel(
        jsonld_blocks_found=jsonld_blocks,
        jsonld_types=sorted(jsonld_types),
        schema_score=0.0,
        has_main_tag=body.find("main") is not None,
        has_article_tag=body.find("article") is not None,
        has_section_tags=body.find("section") is not None,
        metadata_present={
            "title":         bool(soup.find("title") and soup.find("title").get_text().strip()),
            "description":   bool(soup.find("meta", attrs={"name": "description"})),
            "og_title":      bool(soup.find("meta", property="og:title")),
            "og_description":bool(soup.find("meta", property="og:description")),
            "canonical_url": bool(soup.find("link", rel="canonical")),
        },
        flags=[],
    )

def analyze_brand_prominence(
    soup: BeautifulSoup,
    text: str,
    page_url: str,
    brand: Optional[str],
    detection_source: str,
) -> BrandProminenceMatrixModel:
    if not brand:
        return BrandProminenceMatrixModel(flags=["brand_could_not_be_detected"])

    bl = brand.lower()
    tl = text.lower()
    mentions = len(re.findall(re.escape(bl), tl))

    title_tag = soup.find("title")
    in_title = bl in (title_tag.get_text() or "").lower() if title_tag else False

    h1 = soup.find("h1")
    in_h1 = bl in (h1.get_text() or "").lower() if h1 else False

    body = soup.body or soup
    fp = body.find("p")
    in_fp = bl in (fp.get_text() or "").lower() if fp else False

    md = soup.find("meta", attrs={"name": "description"})
    in_md = bl in (md.get("content", "") or "").lower() if md else False

    return BrandProminenceMatrixModel(
        detected_brand=brand,
        detection_source=detection_source,
        brand_mention_count=mentions,
        brand_in_title=in_title,
        brand_in_h1=in_h1,
        brand_in_first_paragraph=in_fp,
        brand_in_meta_description=in_md,
        brand_density_per_1000_words=0.0,
        brand_first_appearance_position_pct=0.0,
        flags=[],
    )

def analyze_competitive_content(
    soup: BeautifulSoup,
    text: str,
    page_url: str,
    brand: Optional[str],
    competitors: List[str],
) -> CompetitiveContentMatrixModel:
    flags: List[str] = []
    bl = brand.lower() if brand else ""
    tl = text.lower()

    brand_count = len(re.findall(re.escape(bl), tl)) if bl else 0
    breakdown: Dict[str, int] = {}
    total_comp = 0
    for comp in competitors:
        cnt = len(re.findall(re.escape(comp.lower()), tl))
        if cnt:
            breakdown[comp] = cnt
            total_comp += cnt

    all_headings = [h.get_text().strip() for h in soup.find_all(["h1", "h2", "h3", "h4"])]
    brand_h = 0
    comp_h = 0
    comps_in_h: List[str] = []
    for heading in all_headings:
        hl = heading.lower()
        if bl and bl in hl:
            brand_h += 1
        for comp in competitors:
            if comp.lower() in hl:
                comp_h += 1
                if comp not in comps_in_h:
                    comps_in_h.append(comp)
                break

    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    th = ((title_tag.get_text() if title_tag else "") + " " +
          (h1_tag.get_text() if h1_tag else "")).lower()
    is_comparison = any(kw in th for kw in [
        "alternative", "vs ", "versus", "comparison", "compare",
        "top ", "best ", "competitor", "replace", "instead of", "similar to",
    ])

    return CompetitiveContentMatrixModel(
        detected_competitors=competitors,
        brand_mention_count=brand_count,
        competitor_total_mentions=total_comp,
        brand_vs_competitor_ratio=0.0,
        brand_in_headings_count=brand_h,
        competitor_heading_count=comp_h,
        competitors_in_headings=comps_in_h,
        competitor_mention_breakdown=breakdown,
        is_comparison_or_alternatives_article=is_comparison,
        content_works_against_brand=False,
        flags=[],
    )

def analyze_citation_worthiness(text: str, soup: BeautifulSoup) -> CitationWorthinessMatrixModel:
    flags: List[str] = []
    numeric_stats = re.findall(
        r"\b\d[\d,]*\.?\d*\s*(?:%|million|billion|trillion|\bk\b|\bM\b|\bB\b|\$|€|£|₹|\bx\b)",
        text, re.IGNORECASE,
    )
    pct_stats = re.findall(r"\b\d+\.?\d*\s*%", text)

    research_patterns = [
        r"\bour (?:data|research|study|survey|analysis|findings|report)\b",
        r"\bwe (?:found|discovered|analyzed|tested|measured|observed)\b",
        r"\baccording to our\b",
        r"\bproprietary\b", r"\bexclusive data\b", r"\binternal (?:data|study)\b",
    ]
    has_research = any(re.search(p, text, re.IGNORECASE) for p in research_patterns)

    proprietary_patterns = [
        r"\b(?:handled|processed|tracked|served)\b.{0,30}\d",
        r"\b\d[\d,]+\s+(?:installs|users|apps|requests|events|sessions|downloads|clients)\b",
    ]
    has_proprietary = any(re.search(p, text, re.IGNORECASE) for p in proprietary_patterns)

    table_count = len(soup.find_all("table"))
    ol_tags = soup.find_all("ol")
    has_numbered = any(len(ol.find_all("li")) >= 3 for ol in ol_tags)
    quotable = len(re.findall(r'["\u201c\u201d][^"\u201c\u201d]{15,200}["\u201c\u201d]', text))

    return CitationWorthinessMatrixModel(
        numeric_stat_count=len(numeric_stats),
        percentage_stat_count=len(pct_stats),
        has_original_research_signals=has_research,
        has_proprietary_data_signals=has_proprietary,
        stat_density_per_1000_words=0.0,
        quotable_statements_count=quotable,
        has_comparison_table=table_count > 0,
        table_count=table_count,
        has_numbered_insights=has_numbered,
        citation_score=0.0,
        flags=[],
    )

def analyze_content_intent(
    soup: BeautifulSoup,
    text: str,
    brand: Optional[str],
) -> ContentIntentMatrixModel:
    flags: List[str] = []
    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    ct = ((title_tag.get_text() if title_tag else "") + " " +
          (h1_tag.get_text() if h1_tag else "")).lower()

    is_alt   = any(kw in ct for kw in ["alternative", "alternatives", "instead of", "replace"])
    is_cmp   = any(kw in ct for kw in ["vs ", "versus", "comparison", "compare", "vs."])
    is_list  = any(kw in ct for kw in ["top ", "best ", "list of", "leading ", "ranking"])
    is_edu   = any(kw in ct for kw in ["how to", "guide", "tutorial", "learn", "what is"])
    is_txn   = any(kw in ct for kw in ["pricing", "buy", "sign up", "free trial", "demo", "get started"])
    bl       = brand.lower() if brand else ""
    is_brand = bool(bl and bl in ct and not (is_alt or is_cmp))

    intent = (
        "alternatives_article"  if is_alt   else
        "comparison_article"    if is_cmp   else
        "curated_list"          if is_list  else
        "educational"           if is_edu   else
        "transactional"         if is_txn   else
        "brand_showcase"        if is_brand else
        "informational"
    )

    return ContentIntentMatrixModel(
        detected_intent=intent,
        is_alternatives_article=is_alt,
        is_competitor_comparison=is_cmp,
        is_brand_showcase=is_brand,
        is_educational=is_edu,
        is_transactional=is_txn,
        title_frames_competitors=(is_alt or (is_cmp and not is_brand)),
        title_frames_brand=is_brand,
        brand_sentence_ratio=0.0,
        intent_alignment_score=0.0,
        flags=[],
    )

def analyze_eeat(soup: BeautifulSoup, text: str, page_url: str) -> EEATMatrixModel:
    flags: List[str] = []
    trust: List[str] = []

    has_author = bool(soup.select_one(
        '[rel="author"], .author, #author, [itemprop="author"], '
        'meta[name="author"], .byline, [class*="byline"], '
        'meta[property="article:author"]'
    ))
    if has_author:
        trust.append("author_attribution")

    has_pub, has_mod = False, False
    for sel in ["time[datetime]", "[itemprop='datePublished']", "[itemprop='dateModified']",
                "meta[property='article:published_time']", "meta[property='article:modified_time']"]:
        tag = soup.select_one(sel)
        if tag:
            if "modified" in sel or "updated" in sel:
                has_mod = True
                trust.append("last_modified_date")
            else:
                has_pub = True
                trust.append("publish_date")

    has_about = bool(soup.find("a", href=re.compile(r"/about|/company|/team|/who-we-are", re.I)))
    if has_about:
        trust.append("about_company_link")

    base = (urlparse(page_url).hostname or "").lower()
    ext = [a for a in soup.find_all("a", href=True)
           if (urlparse(urljoin(page_url, a["href"])).hostname or "").lower() not in ("", base)]
    has_ext = len(ext) >= 3
    if has_ext:
        trust.append("external_references")

    has_quotes = bool(soup.find("blockquote")) or \
        len(re.findall(r"\b(?:according to|stated|explained|noted|confirmed)\b", text, re.IGNORECASE)) >= 2
    if has_quotes:
        trust.append("expert_quotes_or_attribution")

    cred = any(re.search(p, text, re.IGNORECASE) for p in
               [r"\bCEO\b", r"\bfounder\b", r"\bexpert\b", r"\banalyst\b",
                r"\byears of experience\b", r"\bspecialist\b", r"\bhead of\b", r"\bVP\b"])
    if cred:
        trust.append("author_credentials")

    return EEATMatrixModel(
        has_author_attribution=has_author,
        has_publish_date=has_pub,
        has_last_modified_date=has_mod,
        has_about_or_company_link=has_about,
        has_external_citations=has_ext,
        has_expert_quotes_or_attribution=has_quotes,
        has_author_credentials=cred,
        trust_signals_found=trust,
        eeat_score=0.0,
        flags=[],
    )

def analyze_heading_ownership(
    soup: BeautifulSoup,
    brand: Optional[str],
    competitors: List[str],
) -> HeadingOwnershipMatrixModel:
    flags: List[str] = []
    bl = brand.lower() if brand else ""
    cl = [c.lower() for c in competitors]

    h1 = soup.find("h1")
    h1_text = h1.get_text().strip() if h1 else None
    all_h2 = [h.get_text().strip() for h in soup.find_all("h2")]
    all_headings = [h.get_text().strip() for h in soup.find_all(["h1", "h2", "h3", "h4"])
                    if h.get_text().strip()]

    brand_owned: List[str] = []
    comp_owned:  List[str] = []
    neutral:     List[str] = []

    for heading in all_headings:
        hl = heading.lower()
        is_b = bool(bl and bl in hl)
        is_c = any(c in hl for c in cl) if cl else False
        if is_b and not is_c:
            brand_owned.append(heading)
        elif is_c and not is_b:
            comp_owned.append(heading)
        else:
            neutral.append(heading)

    return HeadingOwnershipMatrixModel(
        h1_text=h1_text,
        all_h2_topics=all_h2[:20],
        brand_owned_headings=brand_owned,
        competitor_owned_headings=comp_owned,
        neutral_headings=neutral[:15],
        brand_heading_ratio=0.0,
        competitor_heading_ratio=0.0,
        heading_structure_score=0.0,
        flags=[],
    )

# ==================== GEMINI AI LAYER ====================

def load_system_prompt(page_type: str = "blog") -> str:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = "audit_final_output_prompt.md" if page_type == "blog" else "audit_landing_page_prompt.md"
        fp = os.path.join(base_dir, filename)
        with open(fp, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

# We'll load these dynamically now instead of a global constant
def get_system_prompt(page_type: str) -> str:
    return load_system_prompt(page_type)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
async def call_gemini(audit_payload: Dict[str, Any], html_for_llm: str, page_type: str = "blog") -> Tuple[Dict[str, Any], Dict[str, Optional[int]]]:
    if not GENAI_AVAILABLE:
        return {
            "error": "google-genai package not installed",
            "action": "Run: pip install google-genai",
        }, {
            "input_tokens_prompt": None,
            "input_tokens_html": None,
            "input_tokens_total": None,
            "output_tokens": None,
        }
    if not GEMINI_API_KEY:
        return {
            "error": "GEMINI_API_KEY environment variable not set",
            "action": "Set GEMINI_API_KEY before running the server.",
        }, {
            "input_tokens_prompt": None,
            "input_tokens_html": None,
            "input_tokens_total": None,
            "output_tokens": None,
        }
    raw = ""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt_text = (
            "Analyse this website audit and produce a comprehensive AI citation readiness report.\n\n"
            "=== STRUCTURED AUDIT DATA ===\n"
            f"{json.dumps(audit_payload, indent=2, default=str)}\n\n"
        )
        user_prompt = (
            prompt_text +
            "=== FILTERED HTML SOURCE ===\n"
            f"{html_for_llm}\n\n"
            "Return your analysis as a single JSON object."
        )

        input_tokens_prompt: Optional[int] = None
        input_tokens_html: Optional[int] = None
        input_tokens_total: Optional[int] = None
        try:
            tok_prompt = await asyncio.to_thread(client.models.count_tokens, model=GEMINI_MODEL, contents=prompt_text)
            input_tokens_prompt = getattr(tok_prompt, "total_tokens", None)
        except Exception:
            input_tokens_prompt = None
        try:
            tok_html = await asyncio.to_thread(client.models.count_tokens, model=GEMINI_MODEL, contents=html_for_llm)
            input_tokens_html = getattr(tok_html, "total_tokens", None)
        except Exception:
            input_tokens_html = None
        try:
            tok_total = await asyncio.to_thread(client.models.count_tokens, model=GEMINI_MODEL, contents=user_prompt)
            input_tokens_total = getattr(tok_total, "total_tokens", None)
        except Exception:
            input_tokens_total = None

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=get_system_prompt(page_type),
                temperature=0.2,
                # max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        )
        raw = (response.text or "").strip()

        # Best-effort output token count (not returned)
        output_tokens: Optional[int] = None
        try:
            um = getattr(response, "usage_metadata", None)
            if um is not None:
                output_tokens = getattr(um, "candidates_token_count", None)
        except Exception:
            output_tokens = None

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {"error": "Gemini returned non-object JSON"}, {
                "input_tokens_prompt": input_tokens_prompt,
                "input_tokens_html": input_tokens_html,
                "input_tokens_total": input_tokens_total,
                "output_tokens": output_tokens,
            }
        return parsed, {
            "input_tokens_prompt": input_tokens_prompt,
            "input_tokens_html": input_tokens_html,
            "input_tokens_total": input_tokens_total,
            "output_tokens": output_tokens,
        }
    except json.JSONDecodeError as e:
        log(f"[GEMINI] JSON parse error: {e}")
        return {
            "error": "Gemini returned malformed JSON",
            "details": str(e),
        }, {
            "input_tokens_prompt": None,
            "input_tokens_html": None,
            "input_tokens_total": None,
            "output_tokens": None,
        }
    except Exception as e:
        log(f"[GEMINI] Error: {e}")
        return {
            "error": "Gemini error",
            "details": str(e)[:300],
        }, {
            "input_tokens_prompt": None,
            "input_tokens_html": None,
            "input_tokens_total": None,
            "output_tokens": None,
        }

# ==================== AUDIT ORCHESTRATOR ====================

async def run_audit(url: str, product_name: Optional[str] = None, user_id: Optional[str] = None, product_id: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Optional[int]]]:
    """
    Full pipeline — input is a URL. Optional product_name overrides auto-detection.
    1. Fetch raw HTML (static snapshot)
    2. Fetch JS-rendered HTML via Playwright
    3. Use provided product_name OR auto-detect brand + competitors from page content
    4. Run all 10 matrices
    5. Send structured results to Gemini 2.5 Flash
    6. Return comprehensive report
    """
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")

    log(f"\n[AUDIT] Starting: {url}")

    # Step 1: HTML
    log("[AUDIT] Fetching raw HTML...")
    raw_html = await fetch_raw_html(url)

    rendered_html = raw_html
    try:
        log("[AUDIT] Fetching rendered HTML via Playwright...")
        rendered_html = await fetch_rendered_html(url)
        log(f"[AUDIT] Raw: {len(raw_html)} chars | Rendered: {len(rendered_html)} chars")
    except RuntimeError as e:
        log(f"[AUDIT] Playwright unavailable — using raw HTML. ({e})")
    except Exception as e:
        log(f"[AUDIT] Render failed — using raw HTML. ({e})")

    # Step 2: Parse
    r_soup = BeautifulSoup(rendered_html, "html.parser")
    full_text = plain_text(rendered_html)
    llm_html = await asyncio.to_thread(clean_html_for_aeo, rendered_html)
    raw_wc = wc(plain_text(raw_html))
    ren_wc = wc(full_text)

    if DEBUG_LOGS:
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", urlparse(url).hostname or "unknown")
            timestamp = int(time.time() * 1000)
            html_filename = f"{safe_name}_{timestamp}_filtered.html"
            html_path = os.path.join(OUTPUT_DIR, html_filename)
            await asyncio.to_thread(_write_text_file, html_path, llm_html)
            log(f"[AUDIT] Filtered HTML saved to: {html_path}")
        except Exception as e:
            log(f"[AUDIT] Failed to save filtered HTML: {e}")

    # Step 3: Use provided product_name OR auto-detect brand & competitors
    if product_name:
        brand = product_name
        brand_source = "user_provided"
        log(f"[AUDIT] Brand: '{brand}' (from user input)")
    else:
        brand, brand_source = detect_brand(r_soup, url)
        log(f"[AUDIT] Brand: '{brand}' (from {brand_source})")

    competitors = detect_competitors(r_soup, full_text, brand, url)
    log(f"[AUDIT] Competitors ({len(competitors)}): {competitors[:5]}")

    # Step 4: Matrices
    log("[AUDIT] Running 10 matrices...")
    static_dynamic      = compute_static_dynamic(raw_wc, ren_wc)
    page_renderability  = analyze_page_renderability(raw_html, rendered_html, url)
    readiness           = analyze_readiness(rendered_html, url)
    structured_data     = analyze_structured_data(rendered_html, url)
    brand_prominence    = analyze_brand_prominence(r_soup, full_text, url, brand, brand_source)
    competitive_content = analyze_competitive_content(r_soup, full_text, url, brand, competitors)
    citation_worthiness = analyze_citation_worthiness(full_text, r_soup)
    content_intent      = analyze_content_intent(r_soup, full_text, brand)
    eeat_signals        = analyze_eeat(r_soup, full_text, url)
    heading_ownership   = analyze_heading_ownership(r_soup, brand, competitors)

    # Step 5: Classify Page Type (Blog vs Landing Page)
    # Heuristic: Blogs have article tags, more text, or educational intent.
    # Landing pages have more CTAs, feature sections, or root domains.
    has_article_tag = bool(r_soup.find("article"))
    long_content = ren_wc > 600
    is_educational = content_intent.detected_intent in ["educational", "informational", "curated_list", "comparison_article"]
    
    if (has_article_tag or long_content or is_educational) and content_intent.detected_intent not in ["transactional", "brand_showcase"]:
        page_type = "blog"
    else:
        page_type = "landing"
        
    log(f"[AUDIT] Classified as: {page_type} (Article: {has_article_tag}, Words: {ren_wc}, Intent: {content_intent.detected_intent})")

    # Step 6: Gemini
    log("[AUDIT] Sending to Gemini 2.5 Flash...")
    audit_payload = {
        "url": url,
        "page_type": page_type,
        "detected_brand": brand,
        "brand_detection_source": brand_source,
        "detected_competitors": competitors,
        "static_dynamic": static_dynamic.model_dump(),
        "page_renderability": page_renderability.model_dump(),
        "readiness": readiness.model_dump(),
        "structured_data": structured_data.model_dump(),
        "brand_prominence": brand_prominence.model_dump(),
        "competitive_content": competitive_content.model_dump(),
        "citation_worthiness": citation_worthiness.model_dump(),
        "content_intent": content_intent.model_dump(),
        "eeat_signals": eeat_signals.model_dump(),
        "heading_ownership": heading_ownership.model_dump(),
    }
    ai_analysis, token_usage = await call_gemini(audit_payload, llm_html, page_type=page_type)
    ai_analysis["page_type"] = page_type # For database storage and UI

    if isinstance(ai_analysis, dict):
        log(f"[AUDIT] Gemini keys: {list(ai_analysis.keys())[:10]}")
        log(
            "[AUDIT] Tokens — prompt: %s | html: %s | total: %s | output: %s"
            % (
                token_usage.get("input_tokens_prompt"),
                token_usage.get("input_tokens_html"),
                token_usage.get("input_tokens_total"),
                token_usage.get("output_tokens"),
            )
        )

    # Include static vs dynamic renderability metrics in the final output JSON.
    # We keep Gemini's JSON intact and augment it with a deterministic metrics block.
    result: Dict[str, Any]
    if isinstance(ai_analysis, dict):
        result = dict(ai_analysis)
    else:
        result = {"ai_analysis": ai_analysis}

    result["static_dynamic"] = static_dynamic.model_dump()
    result["is_landing_page"] = (page_type == "landing")

    if DEBUG_LOGS:
        await asyncio.to_thread(save_json, result, url, "ai_analysis")

    async def _bg_upload() -> None:
        try:
            await upload_audit_to_supabase(result, url, user_id, product_id)
        except Exception as e:
            log(f"[SUPABASE] Upload failed: {e}")

    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        asyncio.create_task(_bg_upload())

    return result, token_usage

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "name": "AI Citation Audit API",
        "version": "3.0.0",
        "how_to_use": 'POST /audit with {"url": "https://example.com/page"}',
        "what_it_detects": [
            "Auto-detects brand from JSON-LD / og:site_name / title / domain",
            "Auto-detects competitors from headings, comparison patterns, recurring proper nouns",
            "Identifies if your own content is training AI to recommend your competitors",
            "Detects if Framer/Webflow/Wix makes your content invisible to AI crawlers",
            "Scores E-E-A-T signals, citation worthiness, structured data",
            "Uses Gemini 2.5 Flash to produce prioritised issues and actionable fixes",
        ],
        "endpoints": {
            "POST /audit":          "AI analysis only (Gemini)",
            "POST /audit/matrices": "AI analysis only (Gemini)",
            "GET  /health":         "Service health check",
        },
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "environment": "production" if IS_PROD else "development",
        "gemini_available": GENAI_AVAILABLE,
        "gemini_configured": bool(GEMINI_API_KEY),
        "playwright_available": async_playwright is not None and _browser is not None,
    }

@app.post("/audit")
async def audit_endpoint(body: AuditRequestModel, response: Response):
    """
    Full AI-powered audit. Input: URL (required) + product_name (optional).

    Example:
        POST /audit
        {"url": "https://stripe.com/blog/payments-infrastructure-future"}
        POST /audit
        {"url": "https://example.com", "product_name": "MyBrand"}

    Returns:
    - Auto-detected or user-provided brand name and competitors
    - 10 structured audit matrices
    - Gemini 2.5 Flash AI analysis with prioritised issues and fixes
    """
    try:
        result, token_usage = await run_audit(body.url.strip(), body.product_name, body.user_id, body.product_id)
        response.headers["X-Gemini-Input-Tokens-Prompt"] = str(token_usage.get("input_tokens_prompt") or "")
        response.headers["X-Gemini-Input-Tokens-HTML"] = str(token_usage.get("input_tokens_html") or "")
        response.headers["X-Gemini-Input-Tokens-Total"] = str(token_usage.get("input_tokens_total") or "")
        response.headers["X-Gemini-Output-Tokens"] = str(token_usage.get("output_tokens") or "")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audit/matrices")
async def audit_matrices_endpoint(body: AuditRequestModel, response: Response):
    """
    Runs all 10 matrices and returns AI analysis only (no raw matrix data).
    Provides prioritised issues and actionable fixes from Gemini.

    Example:
        POST /audit/matrices
        {"url": "https://stripe.com/blog/payments-infrastructure-future"}
        POST /audit/matrices
        {"url": "https://example.com", "product_name": "MyBrand"}
    """
    try:
        result, token_usage = await run_audit(body.url.strip(), body.product_name, body.user_id, body.product_id)
        response.headers["X-Gemini-Input-Tokens-Prompt"] = str(token_usage.get("input_tokens_prompt") or "")
        response.headers["X-Gemini-Input-Tokens-HTML"] = str(token_usage.get("input_tokens_html") or "")
        response.headers["X-Gemini-Input-Tokens-Total"] = str(token_usage.get("input_tokens_total") or "")
        response.headers["X-Gemini-Output-Tokens"] = str(token_usage.get("output_tokens") or "")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)