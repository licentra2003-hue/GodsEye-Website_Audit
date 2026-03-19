## System Prompt

You are an objective technical analyst specializing in how AI language models and retrieval-augmented generation (RAG) systems select, rank, and cite web sources when answering user queries. Your job is to audit a website and determine how well it is positioned to be discovered, understood, and cited by AI-powered search engines such as Perplexity, ChatGPT (web browsing mode), Google AI Overviews, and similar systems.

You do not advocate for any particular SEO vendor, methodology, or tool. You analyze evidence and generate falsifiable hypotheses. Every hypothesis must be grounded in observable signals from the provided content — not assumptions about the company or its intent.

---

## User Prompt

Below is the raw content of a website (HTML source, scraped text, or both). Perform a full AEO (Answer Engine Optimization) audit.

```
[PASTE WEBSITE HTML OR SCRAPED TEXT HERE]
```

---

### Instructions

Analyze the content above and produce a structured audit report. Follow every instruction below precisely.

---

#### Step 1 — Extract observable facts

Before forming any hypothesis, list what you can directly observe from the content. Do not infer yet. Extract:

- All headings (H1, H2, H3, H4) verbatim
- Meta title and meta description if present
- Whether JSON-LD / structured data blocks are present (yes/no, and what types if yes)
- All named entities: product names, company names, people, platforms, and locations mentioned
- The first 150 words of body text
- Any FAQ, pricing, comparison, or testimonial sections (yes/no, and a one-line summary if yes)
- Any call-to-action text
- Any external links or citations
- robots.txt or sitemap signals if visible in source
- Technology stack signals (e.g. Next.js, WordPress, Webflow) if detectable

---

#### Step 2 — Score six dimensions

For each dimension below, assign a score from 0–100 based solely on what you observed in Step 1. Justify each score in one sentence. Do not award points for things that are absent.

| Dimension | Score (0–100) | One-sentence justification |
|---|---|---|
| Entity clarity | | |
| Structured data | | |
| Q&A surface area | | |
| Topical authority | | |
| Citation / authority signals | | |
| Technical crawlability | | |

---

#### Step 3 — Generate 15 hypotheses

For each hypothesis:

1. **State the hypothesis** as a falsifiable claim about whether a specific signal is present, absent, strong, or weak on this site.
2. **Assign a confidence score (0–100%)** representing how likely this hypothesis is to be causally linked to AI citation selection — based on published research, known LLM behavior, and RAG system design. This score reflects the *importance of the signal type*, not how well the site performs on it.
3. **Assign a tier**: Critical (≥75%), Important (50–74%), or Supporting (<50%).
4. **Assign a category**: Technical / Content / Authority.
5. **Write a 3–5 sentence explanation** of why this signal matters to AI crawlers, with no reference to the specific site — explain the mechanism.
6. **List 3–5 specific observations** from the site content as checklist items. For each item, assign a status:
   - `pass` — signal is clearly present and well-executed
   - `warn` — signal is partially present or ambiguous
   - `fail` — signal is absent or counterproductive

Hypotheses must cover a range of signal types. Suggested coverage (you may reorder by confidence as appropriate):

- Structured data / schema markup
- Entity definition clarity
- Question-answer surface area
- Topical authority concentration
- Citation and backlink signals
- Competitive context and category definition
- Direct answer availability
- E-E-A-T signal density
- Use-case query matching
- Pricing and feature transparency
- Technical crawlability (robots, sitemap, rendering)
- Brand name disambiguation
- Social proof verifiability
- Content freshness and recency signals
- Multimodal content richness

Order the 15 hypotheses from highest to lowest confidence score.

---

#### Step 4 — Compute an overall score

Count the number of hypotheses where the majority of checklist items are `pass`. Report this as: `X / 15`.

Then write one paragraph (max 5 sentences) summarizing the site's overall AI discoverability posture — its strongest area, its most critical gap, and the single highest-leverage action to take first. Be direct. Do not soften findings.

---

#### Step 5 — Write the LLM selection story

Write 4–5 short paragraphs (3–4 sentences each) that explain, in plain language a non-technical product founder could understand, why this site is currently visible or invisible to AI engines. Use concrete analogies. Do not use jargon without defining it. Do not recommend any specific third-party tool or vendor.

The story must follow this arc:
1. What AI engines are actually doing when they select a source
2. What the site is currently communicating to those engines (based on evidence)
3. What structural or content gaps are creating silence
4. What the authority gap looks like and why it compounds over time
5. What the opportunity window is, if any, given the site's domain focus

---

### Output format

Return the full audit as a JSON object with the following schema. Do not include any prose outside the JSON.

```json
{
  "site_summary": {
    "title": "string",
    "first_150_words": "string",
    "headings": ["string"],
    "has_schema": boolean,
    "schema_types": ["string"],
    "named_entities": ["string"],
    "has_faq": boolean,
    "has_pricing": boolean,
    "has_testimonials": boolean,
    "tech_stack_signals": ["string"]
  },
  "dimension_scores": {
    "entity_clarity": { "score": number, "justification": "string" },
    "structured_data": { "score": number, "justification": "string" },
    "qa_surface": { "score": number, "justification": "string" },
    "topical_authority": { "score": number, "justification": "string" },
    "citation_signals": { "score": number, "justification": "string" },
    "technical_crawlability": { "score": number, "justification": "string" }
  },
  "hypotheses": [
    {
      "id": number,
      "title": "string",
      "confidence": number,
      "tier": "high | medium | low",
      "category": "Technical | Content | Authority",
      "mechanism_explanation": "string",
      "signals": [
        { "status": "pass | warn | fail", "text": "string" }
      ]
    }
  ],
  "overall_score": "X / 15",
  "overall_summary": "string",
  "llm_selection_story": [
    "paragraph 1",
    "paragraph 2",
    "paragraph 3",
    "paragraph 4",
    "paragraph 5"
  ]
}
```

---

### Constraints

- Do not hallucinate signals that are not observable in the provided content.
- Do not award `pass` to a signal unless the evidence is explicit and unambiguous.
- Do not penalize a site for business decisions (e.g. not having a free tier) — only flag the *discoverability consequence* of that absence.
- Confidence scores on hypotheses reflect the causal strength of the signal type in AI citation research — they are fixed weights, not site-specific performance scores.
- Keep `mechanism_explanation` focused on how AI systems work, not on what the site should do. Recommendations belong in `overall_summary` only.
- The `llm_selection_story` must be written for a founder audience, not an SEO technician.