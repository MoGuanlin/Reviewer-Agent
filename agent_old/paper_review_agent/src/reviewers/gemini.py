
import os
import argparse
from pathlib import Path
from typing import Optional

from tenacity import retry, wait_exponential, stop_after_attempt

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from ..utils import write_text, build_paper_text_gemini

SYSTEM_PROMPT = r"""
You are a meticulous, adversarial, and objective reviewer of a theoretical computer science paper.
Your job is NOT to praise; it is to find mistakes, logical gaps, inconsistent notation, missing assumptions,
and places where claims are not supported by the text.

Hard rules:
- Be strictly evidence-based: every nontrivial claim you make must be grounded in the provided paper text.
- Do not invent missing definitions, lemmas, or citations.
- If you are not fully sure, do NOT state it as an error. Mark it with the GAP_TAG and explain what would need to be checked.
- When you claim an error, provide a reproducible, step-by-step derivation/check that demonstrates the error.

Output format (always):
1) Summary of contributions (2–6 sentences).
2) Errors and Improvements:
   - Major Issues (numbered)
   - Minor Issues (numbered)
3) Minor corrections and typos (bullet list)
4) “Structured Partial Progress” section: items you suspect but could not fully verify, each tagged with GAP_TAG.

Mathematical rigor protocol:
- For each major issue you report, explicitly classify it as either:
  (A) COMPLETE_PROOF: every step checked from the provided text; or
  (B) STRUCTURED_PARTIAL_PROGRESS: there is at least one GAP_TAG.
- Use the tag GAP_TAG to mark any gap / unproven assumption / unverifiable step.
- Do not rely on external folklore knowledge unless it is explicitly stated in the paper text.

GAP_TAG = [GAP]
""".strip()


PROMPT_1 = r"""
TASK: Produce an initial, strictly objective review of the paper.

Input paper (verbatim, includes appendices if provided):
<TITLE>
{paper_title}
</TITLE>

<PAPER_TEXT>
{paper_text}
</PAPER_TEXT>

Constraints:
- Focus ONLY on identifying errors, logical gaps, missing assumptions, inconsistent variables/notation,
  incorrect inequality applications, and places where the proof does not follow.
- Suggest concrete improvements (e.g., “state assumption X”, “clarify definition Y”, “fix variable clash in Lemma 3.2”).
- Do not speculate beyond what is written.
- When referencing evidence, cite as precisely as possible, e.g. "(Page 12, Lines 40-55)" or "(Lemma 3.1, Page 7)".

Return the review in the required output format.
""".strip()


PROMPT_2 = r"""
TASK: Audit your previous review for hallucinations and weak claims.

You must:
- For EACH issue you raised, locate the exact supporting location in the paper (page+line preferred).
- Re-derive or re-check the claim step-by-step using ONLY the provided paper text.
- If an issue cannot be fully verified, downgrade it to STRUCTURED_PARTIAL_PROGRESS and tag the first unverifiable step with [GAP].
- Remove any issue that you cannot ground in the text.

Inputs:
(1) Paper:
<TITLE>
{paper_title}
</TITLE>

<PAPER_TEXT>
{paper_text}
</PAPER_TEXT>

(2) Your previous review:
<REVIEW_V1>
{review_v1}
</REVIEW_V1>

Output:
A) A table-like list (plain text is ok) of issues with fields:
   - Issue ID
   - Original claim
   - Evidence location in paper
   - Verification status: COMPLETE_PROOF or STRUCTURED_PARTIAL_PROGRESS
   - If partial: first [GAP] step and what is needed to resolve it
B) A “patch plan”: concrete edits to apply to the review (add/remove/modify items).
""".strip()


PROMPT_3 = r"""
TASK: Write a revised review that incorporates the audit results.

Inputs:
(1) Paper:
<TITLE>
{paper_title}
</TITLE>

<PAPER_TEXT>
{paper_text}
</PAPER_TEXT>

(2) Original review:
<REVIEW_V1>
{review_v1}
</REVIEW_V1>

(3) Audit + patch plan:
<AUDIT_V1>
{audit_v1}
</AUDIT_V1>

Rules:
- The revised review must remove or downgrade any unverified claims.
- Every Major Issue must be labeled COMPLETE_PROOF or STRUCTURED_PARTIAL_PROGRESS.
- Any gap/unverified step must be tagged with [GAP].

Return the revised review in the required output format.
""".strip()


PROMPT_4 = r"""
TASK: Perform a second-round audit to ensure comprehensive coverage, including appendices.

You must:
- Scan for: inconsistent symbols across sections, missing definitions, lemmas used before stated,
  appendices that contain key proofs, and places where the main theorem depends on appendix claims.
- Re-check the highest-impact theorems/lemmas end-to-end.
- Identify any remaining places where your review may still be missing critical issues.

Inputs:
(1) Paper:
<TITLE>
{paper_title}
</TITLE>

<PAPER_TEXT>
{paper_text}
</PAPER_TEXT>

(2) Revised review:
<REVIEW_V2>
{review_v2}
</REVIEW_V2>

Output:
A) Additional issues found (with evidence locations and verification status).
B) Corrections to existing issues (if any).
C) A “coverage checklist” (what you verified, what remains partial with [GAP]).
""".strip()


PROMPT_5 = r"""
TASK: Produce the final, verified review under strict mathematical standards.

Inputs:
(1) Paper:
<TITLE>
{paper_title}
</TITLE>

<PAPER_TEXT>
{paper_text}
</PAPER_TEXT>

(2) Revised review:
<REVIEW_V2>
{review_v2}
</REVIEW_V2>

(3) Second-round audit:
<AUDIT_V2>
{audit_v2}
</AUDIT_V2>

Strict rules:
- The “Major Issues” section may include COMPLETE_PROOF items only if you have fully verified them.
- Anything not fully verified must go to:
   (i) Minor Issues (if low impact) OR
   (ii) Structured Partial Progress (if potentially high impact),
   and must contain a [GAP] marker at the first missing/uncertain step.
- Keep the tone neutral and technical.

Return the final review in the required output format.
""".strip()


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
def call_gemini(client, model: str, system_instruction: str, user_text: str,
                thinking_level: str, max_output_tokens: int, temperature: float) -> str:
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )
    chunks = []
    # Note: client type hint removed to avoid import error if genai not installed
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=user_text,
        config=cfg,
    ):
        if getattr(chunk, "text", None):
            chunks.append(chunk.text)
            continue

        try:
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for p in chunk.candidates[0].content.parts:
                    if getattr(p, "text", None):
                        chunks.append(p.text)
        except Exception:
            pass

    return "".join(chunks)


def run_gemini_review(
    pages: list,
    paper_title: str,
    outdir: Path,
    model: str = "gemini-3.1-pro-preview",
    thinking_level: str = "high",
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    proxy: Optional[str] = None
):
    if genai is None:
        raise ImportError("google-genai package is required for Gemini reviewer.")

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment.")

    outdir.mkdir(parents=True, exist_ok=True)

    client_kwargs = {"api_key": api_key}
    
    if proxy:
        # Assuming genai.Client accepts http_options for proxy configuration
        http_options = types.HttpOptions(
            timeout=100 * 60 * 1000,  # 100 minutes timeout
            client_args={"proxy": proxy},
            async_client_args={"proxy": proxy},
        )
        client_kwargs["http_options"] = http_options
    
    client = genai.Client(**client_kwargs)

    paper_text = build_paper_text_gemini(pages)

    # Step 1
    step1_in = PROMPT_1.format(paper_title=paper_title, paper_text=paper_text)
    review_v1 = call_gemini(client, model, SYSTEM_PROMPT, step1_in,
                            thinking_level, max_output_tokens, temperature)
    write_text(outdir / "review_v1.md", review_v1)

    # Step 2
    step2_in = PROMPT_2.format(paper_title=paper_title, paper_text=paper_text, review_v1=review_v1)
    audit_v1 = call_gemini(client, model, SYSTEM_PROMPT, step2_in,
                           thinking_level, max_output_tokens, temperature)
    write_text(outdir / "audit_v1.md", audit_v1)

    # Step 3
    step3_in = PROMPT_3.format(paper_title=paper_title, paper_text=paper_text,
                               review_v1=review_v1, audit_v1=audit_v1)
    review_v2 = call_gemini(client, model, SYSTEM_PROMPT, step3_in,
                            thinking_level, max_output_tokens, temperature)
    write_text(outdir / "review_v2.md", review_v2)

    # Step 4
    step4_in = PROMPT_4.format(paper_title=paper_title, paper_text=paper_text, review_v2=review_v2)
    audit_v2 = call_gemini(client, model, SYSTEM_PROMPT, step4_in,
                           thinking_level, max_output_tokens, temperature)
    write_text(outdir / "audit_v2.md", audit_v2)

    # Step 5
    step5_in = PROMPT_5.format(paper_title=paper_title, paper_text=paper_text,
                               review_v2=review_v2, audit_v2=audit_v2)
    review_final = call_gemini(client, model, SYSTEM_PROMPT, step5_in,
                               thinking_level, max_output_tokens, temperature)
    write_text(outdir / "review_final.md", review_final)

    # client.close() # genai.Client might not have close() or it's context manager.
    # In the original code: client.close().
    try:
        client.close()
    except AttributeError:
        pass

    print(f"[OK] Gemini review completed. Outputs written to: {outdir.resolve()}")
