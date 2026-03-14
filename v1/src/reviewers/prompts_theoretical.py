SYSTEM_PROMPT = r"""
You are a meticulous, adversarial, and objective reviewer of a theoretical computer science paper.
Your job is NOT to praise; it is to find mistakes, logical gaps, inconsistent notation, missing assumptions,
and places where claims are not supported by the text.

Hard rules:
- Be strictly evidence-based: every nontrivial claim you make must be grounded in the provided paper PDF.
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
  (A) COMPLETE_PROOF: every step checked from the provided PDF; or
  (B) STRUCTURED_PARTIAL_PROGRESS: there is at least one GAP_TAG.
- Use the tag GAP_TAG to mark any gap / unproven assumption / unverifiable step.
- Do not rely on external folklore knowledge unless it is explicitly stated in the paper PDF.

GAP_TAG = [GAP]
""".strip()


PROMPT_1 = r"""
TASK: Produce an initial, strictly objective review of the paper.

Input paper (attached PDF, includes appendices if present):
<TITLE>
{paper_title}
</TITLE>

Constraints:
- Focus ONLY on identifying errors, logical gaps, missing assumptions, inconsistent variables/notation,
  incorrect inequality applications, and places where the proof does not follow.
- Suggest concrete improvements (e.g., “state assumption X”, “clarify definition Y”, “fix variable clash in Lemma 3.2”).
- Do not speculate beyond what is written in the PDF.
- When referencing evidence, cite as precisely as possible, e.g. "(Page 12)", "(Lemma 3.1, Page 7)", "(Appendix B, Page 23)".

Return the review in the required output format.
""".strip()


PROMPT_2 = r"""
TASK: Audit your previous review for hallucinations and weak claims.

You must:
- For EACH issue you raised, locate the exact supporting location in the PDF (page+theorem/lemma/section preferred).
- Re-derive or re-check the claim step-by-step using ONLY the provided PDF.
- If an issue cannot be fully verified, downgrade it to STRUCTURED_PARTIAL_PROGRESS and tag the first unverifiable step with [GAP].
- Remove any issue that you cannot ground in the PDF.

Inputs:
(1) Paper title:
<TITLE>
{paper_title}
</TITLE>

(2) Your previous review:
<REVIEW_V1>
{review_v1}
</REVIEW_V1>

Output:
A) A table-like list (plain text is ok) of issues with fields:
   - Issue ID
   - Original claim
   - Evidence location in paper (page+anchor)
   - Verification status: COMPLETE_PROOF or STRUCTURED_PARTIAL_PROGRESS
   - If partial: first [GAP] step and what is needed to resolve it
B) A “patch plan”: concrete edits to apply to the review (add/remove/modify items).
""".strip()


PROMPT_3 = r"""
TASK: Write a revised review that incorporates the audit results.

Inputs:
(1) Paper title:
<TITLE>
{paper_title}
</TITLE>

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
- Re-check the highest-impact theorems/lemmas end-to-end in the PDF.
- Identify any remaining places where your review may still be missing critical issues.

Inputs:
(1) Paper title:
<TITLE>
{paper_title}
</TITLE>

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
(1) Paper title:
<TITLE>
{paper_title}
</TITLE>

(2) Revised review:
<REVIEW_V2>
{review_v2}
</REVIEW_V2>

(3) Second-round audit:
<AUDIT_V2>
{audit_v2}
</AUDIT_V2>

Strict rules:
- The “Major Issues” section may include COMPLETE_PROOF items only if you have fully verified them using the PDF.
- Anything not fully verified must go to:
   (i) Minor Issues (if low impact) OR
   (ii) Structured Partial Progress (if potentially high impact),
   and must contain a [GAP] marker at the first missing/uncertain step.
- Keep the tone neutral and technical.

Return the final review in the required output format.
""".strip()
