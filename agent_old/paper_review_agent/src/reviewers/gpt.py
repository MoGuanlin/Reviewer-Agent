
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..utils import (
    extract_pdf_pages,
    join_pages_with_markers,
    chunk_pages,
    build_sampled_context,
    safe_json_loads,
    write_text
)

# -------------------------
# Prompts
# -------------------------

SYSTEM_PROMPT = """You are a meticulous, adversarial technical reviewer for theoretical papers.
Your job is to find errors, inconsistencies, missing assumptions, and ambiguous definitions.
Be strictly objective: do not praise. Do not speculate.
If something is not verifiable from the provided text, mark it explicitly.
"""

RIGOR_TEXT = """RIGOR TEXT (must follow):
- Only label something “Complete Proof” if every nontrivial step is explicitly justified from the excerpt or from a clearly stated external theorem that is quoted with its full conditions.
- If ANY step depends on an unstated assumption, missing lemma, unclear definition, or an external fact not provided, you MUST switch to “Structured Partial Progress”.
- Every time you encounter a gap / missing assumption / unclear inference, tag it with <GAP> and give:
  (i) the exact location (definition/lemma/theorem/line/section/page if available),
  (ii) what is missing,
  (iii) the minimal fix needed (extra assumption / lemma / proof sketch / counterexample).
- Distinguish “definition vs construction” mismatches explicitly when they occur.
- Prefer concrete falsification: provide a minimal counterexample if a claim seems false.
- Do not add new claims beyond the provided text. If uncertain, say “Not verifiable from excerpt” and tag <GAP>.
"""

FIVE_STEP_USER_TEMPLATE = """TASK: Rigorously review the following paper excerpt (may include definitions, lemmas, proofs, and appendices).
You must follow this iterative self-correction protocol:

1) Generate an INITIAL REVIEW that is strictly objective, focusing only on identifying errors and suggesting improvements.
2) SELF-CORRECT your first review by rigorously critiquing your own findings:
   - verify every derivation you relied on,
   - check for hallucinations,
   - ensure any claim of an error is substantive and traceable to the text.
3) Generate a REVISED REVIEW incorporating these corrections.
4) Perform a SECOND ROUND of self-correction to further refine the logic and ensure comprehensive coverage, including appendices if present.
5) Produce a FINAL, VERIFIED REVIEW adhering to strict mathematical standards.

{rigor_text}

OUTPUT FORMAT (exact headings):
A. Initial Review
B. Self-Critique #1 (hallucination/verification audit)
C. Revised Review
D. Self-Critique #2 (coverage + logic audit)
E. Final Verified Review
   - Verdict: [Complete Proof | Structured Partial Progress]
   - Top Issues (bulleted, with <GAP> tags where applicable)
   - Suggested Fixes (paired to issues)

PAPER EXCERPT:
<<<
{paper_text}
>>>

{extra_context}
"""

CHUNK_ISSUE_EXTRACTOR_USER = """You are reviewing ONLY the provided pages of a paper excerpt.
Goal: extract potential technical issues, proof gaps, missing assumptions, unclear definitions, and inconsistencies found in THESE pages ONLY.

Rules:
- Be adversarial and specific.
- For every issue, include a short excerpt (<= 350 chars) copied from the pages to anchor it.
- If something is uncertain, still include it but mark verifiability as "uncertain".
- Output MUST be strict JSON only (no markdown), matching this schema:

{
  "pages": {"start": <int>, "end": <int>},
  "issues": [
    {
      "id": "<string>",
      "severity": "high|medium|low",
      "type": "proof_gap|missing_assumption|definition_ambiguity|inconsistency|notation|citation_dependency|other",
      "location": "page <n> ... (or section/lemma if visible)",
      "excerpt": "<short quote>",
      "claim": "<what seems wrong / unclear>",
      "why_it_matters": "<impact on correctness>",
      "minimal_fix": "<minimal additional assumption/lemma/change>"
    }
  ]
}

PAGES TEXT:
<<<
{chunk_text}
>>>
"""

MERGE_DEDUP_USER = """You are given a list of JSON objects, each containing candidate issues found in different page ranges of a paper.
Task: merge, deduplicate, and standardize them into a single consolidated JSON object.

Rules:
- Deduplicate issues that refer to the same underlying problem; keep the strongest version.
- Preserve page references and excerpts (may include multiple page refs if merged).
- Rewrite issue 'claim' to be crisp and traceable.
- Keep the schema below exactly and output strict JSON only.

Schema:
{
  "issues": [
    {
      "id": "ISSUE-###",
      "severity": "high|medium|low",
      "type": "proof_gap|missing_assumption|definition_ambiguity|inconsistency|notation|citation_dependency|other",
      "location": "<merged locations>",
      "excerpt": "<one or more short quotes>",
      "claim": "<crisp statement>",
      "why_it_matters": "<impact>",
      "minimal_fix": "<minimal fix>"
    }
  ]
}

INPUT JSON LIST:
<<<
{json_list}
>>>
"""


# -------------------------
# OpenRouter Client
# -------------------------

class OpenRouterError(RuntimeError):
    pass


@dataclass
class OpenRouterConfig:
    api_key: str
    model: str = "openai/gpt-5.2-pro"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    temperature: float = 0.2
    max_output_tokens: int = 3000
    timeout_sec: int = 120
    app_title: str = "paper-review-agent"
    http_referer: str = "http://localhost"  # can be any URL you control
    proxy: Optional[str] = None


@retry(
    retry=retry_if_exception_type((requests.RequestException, OpenRouterError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(5),
)
def openrouter_chat(cfg: OpenRouterConfig, messages: List[Dict[str, str]]) -> str:
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
        # OpenRouter best practice headers:
        "X-Title": cfg.app_title,
        "HTTP-Referer": cfg.http_referer,
    }
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_output_tokens,
    }

    proxies = None
    if cfg.proxy:
        proxies = {"http": cfg.proxy, "https": cfg.proxy}

    resp = requests.post(cfg.base_url, headers=headers, json=payload, timeout=cfg.timeout_sec, proxies=proxies)
    if resp.status_code != 200:
        raise OpenRouterError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise OpenRouterError(f"Malformed response: {e}; body={str(data)[:500]}")


# -------------------------
# Pipeline
# -------------------------

def run_chunk_issue_scan(cfg: OpenRouterConfig, chunks: List[Tuple[int, int, str]], sleep_sec: float = 0.0) -> List[Dict[str, Any]]:
    results = []
    for idx, (sp, ep, ctext) in enumerate(chunks, start=1):
        user = CHUNK_ISSUE_EXTRACTOR_USER.format(chunk_text=ctext)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        out = openrouter_chat(cfg, messages)
        try:
            obj = safe_json_loads(out)
            # enforce page range if missing
            if "pages" not in obj:
                obj["pages"] = {"start": sp, "end": ep}
            results.append(obj)
        except Exception as e:
            # keep raw for debugging
            results.append({"pages": {"start": sp, "end": ep}, "issues": [], "_parse_error": str(e), "_raw": out[:2000]})
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return results


def run_merge_dedup(cfg: OpenRouterConfig, issue_jsons: List[Dict[str, Any]]) -> Dict[str, Any]:
    user = MERGE_DEDUP_USER.format(json_list=json.dumps(issue_jsons, ensure_ascii=False))
    messages = [
        {"role": "system", "content": "You are a careful editor. Output strict JSON only."},
        {"role": "user", "content": user},
    ]
    out = openrouter_chat(cfg, messages)
    return safe_json_loads(out)


def run_five_step_review(cfg: OpenRouterConfig, paper_text: str, extra_context: str = "") -> str:
    user = FIVE_STEP_USER_TEMPLATE.format(
        rigor_text=RIGOR_TEXT,
        paper_text=paper_text,
        extra_context=extra_context.strip(),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return openrouter_chat(cfg, messages)


def run_gpt_review(
    pages: List[Dict[str, Any]],
    out_file: Path,
    model: str = "openai/gpt-5.2-pro",
    max_input_chars: int = 180000,
    chunk_chars: int = 35000,
    max_output_tokens: int = 3500,
    temperature: float = 0.2,
    sleep_sec: float = 0.0,
    api_key: Optional[str] = None,
    proxy: Optional[str] = None
):
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    
    if not api_key:
        raise RuntimeError("Please set environment variable OPENROUTER_API_KEY")

    cfg = OpenRouterConfig(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        proxy=proxy
    )

    full_text = join_pages_with_markers(pages)

    stats = {
        "pages": len(pages),
        "extracted_chars": len(full_text),
        "max_input_chars": max_input_chars,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_output_tokens": cfg.max_output_tokens,
    }

    # Decide mode
    if len(full_text) <= max_input_chars:
        mode = "single_call_full_text"
        review = run_five_step_review(cfg, paper_text=full_text, extra_context="")
        run_log = {"mode": mode, "stats": stats}
    else:
        mode = "chunk_scan_then_five_step"
        chunks = chunk_pages(pages, max_chars_per_chunk=chunk_chars)
        issue_jsons = run_chunk_issue_scan(cfg, chunks, sleep_sec=sleep_sec)
        merged = run_merge_dedup(cfg, issue_jsons)

        sampled = build_sampled_context(pages, max_chars=max_input_chars)

        extra_context = (
            "NOTE: The full paper text exceeded the raw-context budget.\n"
            "You are given:\n"
            "1) A SAMPLED subset of raw paper text with page markers.\n"
            "2) A CONSOLIDATED list of candidate issues extracted by scanning all chunks.\n\n"
            "You must use page references and excerpts to stay grounded. If not verifiable from provided context, mark <GAP>.\n\n"
            f"CONSOLIDATED_ISSUES_JSON:\n{json.dumps(merged, ensure_ascii=False, indent=2)}\n"
        )

        review = run_five_step_review(cfg, paper_text=sampled, extra_context=extra_context)
        run_log = {
            "mode": mode,
            "stats": stats,
            "chunking": {
                "chunk_chars": chunk_chars,
                "num_chunks": len(chunks),
            },
            "merged_issue_count": len(merged.get("issues", [])) if isinstance(merged, dict) else None,
        }

    # Write report
    header = (
        "# Paper Review Report\n\n"
        "## Run Metadata\n"
        f"- Pages: {stats['pages']}\n"
        f"- Extracted chars: {stats['extracted_chars']}\n"
        f"- Mode: {mode}\n"
        f"- Model: {stats['model']}\n"
        f"- Temperature: {stats['temperature']}\n"
        f"- Max output tokens: {stats['max_output_tokens']}\n\n"
        "## Review Output\n\n"
    )

    footer = "\n\n---\n\n## Run Log (JSON)\n\n```json\n" + json.dumps(run_log, ensure_ascii=False, indent=2) + "\n```\n"
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(review.strip())
        f.write(footer)

    print(f"OK: wrote {out_file}")
