import json
from typing import Any, Dict, Optional
import os
from pathlib import Path

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from google.genai.local_tokenizer import LocalTokenizer
    LOCAL_GEMINI_AVAILABLE = True
except ImportError:
    LOCAL_GEMINI_AVAILABLE = False

try:
    from vertexai.preview import tokenization
    VERTEXAI_TOKENIZER_AVAILABLE = True
except ImportError:
    VERTEXAI_TOKENIZER_AVAILABLE = False


def count_tokens_from_text(text: str, model: str = "gemini-1.5-flash") -> int:
    """Count tokens in a text string using available tokenizers.
    
    Priority order:
    1. Google GenAI LocalTokenizer (for Gemini models)
    2. Vertex AI tokenizer (for Gemini models)
    3. tiktoken (for OpenAI models)
    4. Fallback: character count approximation
    
    Args:
        text: The text to count tokens for
        model: The model name to use for tokenization
              Default: gemini-1.5-flash
              Common options: gemini-1.5-flash, gemini-pro, gpt-4, gpt-3.5-turbo
    
    Returns:
        The number of tokens
    """
    # Try Google GenAI LocalTokenizer first (for Gemini models)
    if model.startswith("gemini"):
        result = count_tokens_with_local_gemini(text, model)
        if result is not None:
            return result
        
        # Fallback to Vertex AI
        result = count_tokens_with_vertexai(text, model)
        if result is not None:
            return result
    
    # Try tiktoken for OpenAI models or as fallback
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            # If model not found, use cl100k_base (GPT-4 default)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    
    # Final fallback: approximate using character count (rough estimate)
    # This is a very rough approximation: ~4 chars per token
    return len(text) // 4


def count_tokens_from_dict(data: Dict[str, Any], model: str = "gemini-1.5-flash") -> int:
    """Count tokens from a dictionary by converting it to JSON string.
    
    Args:
        data: The dictionary to count tokens for
        model: The model name to use for tokenization (default: gemini-1.5-flash)
    
    Returns:
        The number of tokens
    """
    json_str = json.dumps(data, ensure_ascii=False)
    return count_tokens_from_text(json_str, model)


def count_tokens_from_html_file(file_path: str, model: str = "gemini-1.5-flash") -> int:
    """Count tokens from an HTML file.
    
    Args:
        file_path: Path to the HTML file
        model: The model name to use for tokenization (default: gemini-1.5-flash)
    
    Returns:
        The number of tokens in the file
    
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return count_tokens_from_text(html_content, model)


def count_tokens_from_file(file_path: str, model: str = "gemini-1.5-flash") -> int:
    """Count tokens from a JSON or HTML file.
    
    Args:
        file_path: Path to the file (JSON or HTML)
        model: The model name to use for tokenization (default: gemini-1.5-flash)
    
    Returns:
        The number of tokens in the file
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON (for JSON files)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.html' or file_ext == '.htm':
        return count_tokens_from_html_file(file_path, model)
    elif file_ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return count_tokens_from_dict(data, model)
    else:
        # Default: treat as text file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return count_tokens_from_text(content, model)


def count_tokens_with_local_gemini(text: str, model_name: str = "gemini-1.5-flash") -> Optional[int]:
    """Count tokens using Google GenAI LocalTokenizer.
    
    Calculates the token count for a given text without an API call.
    Note: The first run will download the model vocabulary (~1-2MB).
    
    Args:
        text: The text to count tokens for
        model_name: The Gemini model name (default: gemini-1.5-flash)
    
    Returns:
        The number of tokens, or None if not available
    """
    if not LOCAL_GEMINI_AVAILABLE:
        return None
    
    try:
        tokenizer = LocalTokenizer(model_name=model_name)
        result = tokenizer.count_tokens(text)
        return result.total_tokens
    except Exception as e:
        print(f"Local Gemini tokenizer error: {e}")
        return None


def count_tokens_with_vertexai(text: str, model_name: str = "gemini-1.5-flash") -> Optional[int]:
    """Count tokens using Vertex AI tokenizer.
    
    Calculates the token count for a given text using Vertex AI's local tokenizer.
    
    Args:
        text: The text to count tokens for
        model_name: The Gemini model name (default: gemini-1.5-flash)
    
    Returns:
        The number of tokens, or None if not available
    """
    if not VERTEXAI_TOKENIZER_AVAILABLE:
        return None
    
    try:
        tokenizer = tokenization.get_tokenizer_for_model(model_name)
        result = tokenizer.count_tokens(text)
        return result.total_tokens
    except Exception as e:
        print(f"Vertex AI tokenizer error: {e}")
        return None


def get_token_stats(file_path: str, model: str = "gemini-1.5-flash") -> Dict[str, Any]:
    """Get detailed token statistics for a JSON or HTML file.
    
    Args:
        file_path: Path to the file (JSON or HTML)
        model: The model name to use for tokenization (default: gemini-1.5-flash)
    
    Returns:
        Dictionary with token statistics including:
        - total_tokens: Total token count
        - file_size_bytes: File size in bytes
        - file_size_kb: File size in KB
        - estimated_cost: Estimated cost (for OpenAI models)
        - file_type: Type of file (json, html, or text)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size_bytes = os.path.getsize(file_path)
    file_size_kb = file_size_bytes / 1024
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.html' or file_ext == '.htm':
        file_type = 'html'
    elif file_ext == '.json':
        file_type = 'json'
    else:
        file_type = 'text'
    
    total_tokens = count_tokens_from_file(file_path, model)
    
    # Rough cost estimation (adjust rates as needed)
    # Gemini 1.5 Flash: ~$0.000075 per 1K tokens
    # Gemini 1.5 Pro: ~$0.00035 per 1K tokens
    # GPT-4: ~$0.03 per 1K input tokens, $0.06 per 1K output tokens
    # GPT-3.5-turbo: ~$0.0015 per 1K input tokens, $0.002 per 1K output tokens
    if "gemini-1.5-pro" in model:
        cost_per_1k = 0.00035
    elif "gemini" in model:
        cost_per_1k = 0.000075
    elif "gpt-4" in model:
        cost_per_1k = 0.03
    else:
        cost_per_1k = 0.0015
    
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    
    return {
        "total_tokens": total_tokens,
        "file_size_bytes": file_size_bytes,
        "file_size_kb": round(file_size_kb, 2),
        "estimated_cost_usd": round(estimated_cost, 6),
        "model_used": model,
        "file_type": file_type
    }


def print_token_stats(file_path: str, model: str = "gemini-1.5-flash"):
    """Print token statistics for a JSON or HTML file in a readable format.
    
    Args:
        file_path: Path to the file (JSON or HTML)
        model: The model name to use for tokenization (default: gemini-1.5-flash)
    """
    try:
        stats = get_token_stats(file_path, model)
        
        print(f"\n{'='*60}")
        print(f"Token Statistics for: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        print(f"File Type:           {stats['file_type'].upper()}")
        print(f"Total Tokens:       {stats['total_tokens']:,}")
        print(f"File Size:          {stats['file_size_kb']:.2f} KB")
        print(f"Model Used:         {stats['model_used']}")
        print(f"Estimated Cost:      ${stats['estimated_cost_usd']:.6f} (input)")
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python token_utils.py <file_path> [model]")
        print("\nExample:")
        print("  python token_utils.py metadata_outputs/adjoe_io_1771791326109_ast.json")
        print("  python token_utils.py html_outputs/adjoe_io_1771791326140.html")
        print("  python token_utils.py metadata_outputs/adjoe_io_1771791326109_ast.json gemini-1.5-pro")
        print("  python token_utils.py html_outputs/adjoe_io_1771791326140.html gpt-4")
        sys.exit(1)
    
    file_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gemini-1.5-flash"
    
    print_token_stats(file_path, model)
