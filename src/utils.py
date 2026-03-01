import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError:
        PdfReader = None


def clean_text(s: str) -> str:
    if not s:
        return ""
    # Remove null bytes and other control characters (keeping newline and tab)
    # \x00-\x08: control chars (NUL to BS)
    # \x0b: vertical tab
    # \x0c: form feed
    # \x0e-\x1f: shift out/in etc.
    # \x7f: delete
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    
    # Aggressively strip non-ASCII characters to avoid proxy/SSL issues
    s = s.encode("ascii", "ignore").decode("ascii")

    # Normalize excessive whitespace
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    if PdfReader is None:
        raise ImportError("pypdf or PyPDF2 is required to read PDF files.")
        
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF {pdf_path}: {e}")

    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append({"page": i, "text": clean_text(txt)})
    return pages


def add_line_numbers(text: str, start: int = 1) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    width = max(4, len(str(start + len(lines))))
    out = []
    for idx, line in enumerate(lines, start=start):
        out.append(f"{idx:>{width}} | {line}")
    return "\n".join(out)


def build_paper_text_gemini(pages: List[Dict[str, Any]]) -> str:
    """Builds text for Gemini reviewer with line numbers."""
    blocks = []
    for p in pages:
        page_num = p["page"]
        text = p["text"]
        numbered = add_line_numbers(text, start=1)
        blocks.append(f"=== Page {page_num} ===\n{numbered}")
    return "\n\n".join(blocks)


def join_pages_with_markers(pages: List[Dict[str, Any]]) -> str:
    """Joins pages for GPT reviewer."""
    parts = []
    for p in pages:
        parts.append(f"[Page {p['page']}]\n{p['text']}\n")
    return "\n".join(parts).strip()


def chunk_pages(
    pages: List[Dict[str, Any]],
    max_chars_per_chunk: int,
    min_pages_per_chunk: int = 2,
) -> List[Tuple[int, int, str]]:
    """Return list of (start_page, end_page, chunk_text)."""
    chunks: List[Tuple[int, int, str]] = []
    if not pages:
        return chunks

    start_idx = 0
    n = len(pages)
    
    while start_idx < n:
        current_chunk_pages = []
        current_length = 0
        end_idx = start_idx
        
        # Always take at least one page
        first_page = pages[start_idx]
        formatted_first = f"[Page {first_page['page']}]\n{first_page['text']}\n\n"
        current_chunk_pages.append(formatted_first)
        current_length += len(formatted_first)
        end_idx += 1
        
        while end_idx < n:
            next_page = pages[end_idx]
            formatted_next = f"[Page {next_page['page']}]\n{next_page['text']}\n\n"
            
            # Check constraints
            if current_length + len(formatted_next) > max_chars_per_chunk:
                if (end_idx - start_idx) >= min_pages_per_chunk:
                    break
                break
            
            current_chunk_pages.append(formatted_next)
            current_length += len(formatted_next)
            end_idx += 1
            
        # Construct chunk
        chunk_text = "".join(current_chunk_pages).strip()
        start_page_num = pages[start_idx]["page"]
        end_page_num = pages[end_idx - 1]["page"]
        
        chunks.append((start_page_num, end_page_num, chunk_text))
        start_idx = end_idx
        
    return chunks


def build_sampled_context(pages: List[Dict[str, Any]], max_chars: int) -> str:
    """
    Build a sampled context when the full paper is too long:
    - Always include first 3 pages and last 2 pages (if available)
    - Evenly sample remaining pages
    """
    n = len(pages)
    if n == 0:
        return ""

    # Determine indices (0-based) for must-have pages
    # First 3 pages: indices 0, 1, 2
    must_indices = set()
    for i in range(min(3, n)):
        must_indices.add(i)
    # Last 2 pages: indices n-1, n-2
    for i in range(max(0, n - 2), n):
        must_indices.add(i)

    # Convert to sorted list
    selected_indices = sorted(list(must_indices))

    # Pre-calculate formatted strings for all pages to know their lengths
    # (This might be memory intensive for huge papers, but typically fine for text)
    page_blocks = [f"[Page {p['page']}]\n{p['text']}\n\n" for p in pages]
    
    # Calculate current size
    current_size = sum(len(page_blocks[i]) for i in selected_indices)
    
    if current_size >= max_chars:
        # Truncate if must-haves are already too big
        full_text = "".join(page_blocks[i] for i in selected_indices)
        return full_text[:max_chars]

    # Sample from remaining
    remaining_indices = [i for i in range(n) if i not in must_indices]
    
    if remaining_indices:
        # Simple heuristic: try to add pages until full
        budget = max_chars - current_size
        avg_page_len = sum(len(page_blocks[i]) for i in remaining_indices) / len(remaining_indices) if remaining_indices else 0
        
        if avg_page_len > 0:
            target_count = int(budget / avg_page_len)
            if target_count < 1:
                target_count = 1
            
            step = max(1, len(remaining_indices) // target_count)
            
            for i in remaining_indices[::step]:
                if current_size + len(page_blocks[i]) > max_chars:
                    break
                selected_indices.append(i)
                current_size += len(page_blocks[i])
    
    selected_indices.sort()
    return "".join(page_blocks[i] for i in selected_indices)


def build_sampled_context_gemini(pages: List[Dict[str, Any]], max_chars: int) -> str:
    """
    Build a sampled context for Gemini when the full paper is too long.
    Similar to build_sampled_context but preserves line numbers and format.
    """
    n = len(pages)
    if n == 0:
        return ""

    # Determine indices (0-based) for must-have pages
    must_indices = set()
    for i in range(min(3, n)):
        must_indices.add(i)
    for i in range(max(0, n - 2), n):
        must_indices.add(i)

    selected_indices = sorted(list(must_indices))

    # Pre-calculate formatted strings
    def format_page(p):
        page_num = p["page"]
        text = p["text"]
        numbered = add_line_numbers(text, start=1)
        return f"=== Page {page_num} ===\n{numbered}"

    page_blocks = [format_page(p) for p in pages]
    
    current_size = sum(len(page_blocks[i]) for i in selected_indices) + (len(selected_indices) - 1) * 2 # +2 for newlines
    
    if current_size >= max_chars:
        # Truncate if must-haves are already too big
        full_text = "\n\n".join(page_blocks[i] for i in selected_indices)
        return full_text[:max_chars]

    # Sample from remaining
    remaining_indices = [i for i in range(n) if i not in must_indices]
    
    if remaining_indices:
        budget = max_chars - current_size
        avg_page_len = sum(len(page_blocks[i]) for i in remaining_indices) / len(remaining_indices) if remaining_indices else 0
        
        if avg_page_len > 0:
            target_count = int(budget / avg_page_len)
            if target_count < 1:
                target_count = 1
            
            step = max(1, len(remaining_indices) // target_count)
            
            for i in remaining_indices[::step]:
                if current_size + len(page_blocks[i]) > max_chars:
                    break
                selected_indices.append(i)
                current_size += len(page_blocks[i]) + 2
    
    selected_indices.sort()
    return "\n\n".join(page_blocks[i] for i in selected_indices)


def safe_json_loads(s: str) -> Any:
    """
    Try to load JSON even if the model wraps it with extra text.
    """
    s = s.strip()
    # Simple direct load
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Try to find the first outer-most {} or []
    start_brace = s.find('{')
    start_bracket = s.find('[')
    
    start = -1
    end = -1
    is_object = False
    
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start = start_brace
        end = s.rfind('}')
        is_object = True
    elif start_bracket != -1:
        start = start_bracket
        end = s.rfind(']')
        is_object = False
        
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Last ditch: try to fix common trailing comma issues (simple regex)
            candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
                
    # If all fails, return raw string wrapped in dict to avoid crashing
    return {"_raw_content": s, "_error": "Failed to parse JSON"}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
