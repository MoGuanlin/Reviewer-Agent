
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError:
        PdfReader = None


def clean_text(s: str) -> str:
    s = s.replace("\x00", "")
    # Normalize excessive whitespace
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    if PdfReader is None:
        raise ImportError("pypdf or PyPDF2 is required to read PDF files.")
        
    reader = PdfReader(pdf_path)
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
    start_idx = 0
    n = len(pages)
    while start_idx < n:
        end_idx = start_idx
        acc = ""
        start_page = pages[start_idx]["page"]
        while end_idx < n:
            candidate = acc + f"[Page {pages[end_idx]['page']}]\n{pages[end_idx]['text']}\n\n"
            # ensure at least min pages per chunk unless at end
            if (len(candidate) > max_chars_per_chunk) and (end_idx - start_idx) >= min_pages_per_chunk:
                break
            if len(candidate) > max_chars_per_chunk and (end_idx - start_idx) == 0:
                # single huge page; force cut
                acc = candidate[:max_chars_per_chunk]
                end_idx += 1
                break
            acc = candidate
            end_idx += 1

        end_page = pages[end_idx - 1]["page"]
        chunks.append((start_page, end_page, acc.strip()))
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

    must_pages = set()
    for p in range(1, min(3, n) + 1):
        must_pages.add(p)
    for p in range(max(1, n - 1), n + 1):
        must_pages.add(p)

    selected = []
    # pick must pages
    for i in range(1, n + 1):
        if i in must_pages:
            selected.append(i)

    # add evenly spaced pages until reaching budget
    # approximate each page block cost
    page_blocks = {p["page"]: f"[Page {p['page']}]\n{p['text']}\n\n" for p in pages}
    current = "".join(page_blocks[i] for i in selected if i in page_blocks)
    if len(current) >= max_chars:
        return current[:max_chars]

    # candidate pages exclude must pages
    remaining = [i for i in range(1, n + 1) if i not in must_pages]
    if remaining:
        step = max(1, len(remaining) // 20)  # target ~20 additional pages
        for i in remaining[::step]:
            selected.append(i)
            current = "".join(page_blocks[j] for j in sorted(set(selected)) if j in page_blocks)
            if len(current) >= max_chars:
                break

    current = "".join(page_blocks[j] for j in sorted(set(selected)) if j in page_blocks)
    return current[:max_chars]

def safe_json_loads(s: str) -> Any:
    """
    Try to load JSON even if the model wraps it with extra text (shouldn't, but happens).
    """
    s = s.strip()
    # Find first {...} block
    if not s.startswith("{"):
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            s = m.group(0)
    return json.loads(s)

def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
