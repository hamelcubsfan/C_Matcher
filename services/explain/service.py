"""Gemini-powered explanation generation."""
from __future__ import annotations

import json
from typing import List, Literal

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ServerError
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


ReasonCode = Literal[
    "domain_expertise",
    "technical_skills",
    "leadership_experience",
    "product_commercialization",
    "education",
    "patents",
    "seniority_match",
    "cultural_fit",
    "safety_critical_experience",
]


class ExplanationReason(BaseModel):
    code: ReasonCode
    evidence: List[str] = Field(default_factory=list)
    weight: float


class ExplanationPayload(BaseModel):
    summary: str
    reasons: List[ExplanationReason]
    confidence: float


class ScorePayload(BaseModel):
    confidence: float
    reasons: List[ExplanationReason] = Field(default_factory=list)


class ExplanationService:
    def __init__(self):
        from services.shared.config import get_settings

        settings = get_settings()
        self.client = genai.Client(api_key=settings.gemini_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ServerError),
    )
    def score(
        self,
        resume_spans: str,
        job_spans: str,
        must_haves: list[str],
        job_title: str,
    ) -> str:
        """
        Fast scoring pass for ranking only.
        Returns JSON: {"confidence": float, "reasons": [...]}
        """
        sys_prompt = (
            "You are a strict technical recruiter evaluator. Output only valid JSON. "
            "Be conservative with confidence."
        )

        prompt = f"""
Job Title: {job_title}

Resume spans:
{resume_spans}

Job spans:
{job_spans}

Must-haves: {', '.join(must_haves)}

INSTRUCTIONS:
- Return a single confidence score 0.0 to 1.0 for match quality.
- Use ONLY the allowed reason codes.
- Apply strict mismatch penalties:
  - If candidate is Recruiter/Sourcer/Talent and job is Engineering/Science/Product: confidence must be 0.05.
  - If domain mismatch (compiler without LLVM/MLIR, hardware/FPGA without HW, frontend vs backend-only): cap confidence at 0.4.
  - Penalize seniority mismatch and over-qualification.
"""

        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[sys_prompt, prompt],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ScorePayload,
            ),
        )

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return json.dumps(
                {
                    "confidence": float(parsed.confidence),
                    "reasons": [r.model_dump() for r in (parsed.reasons or [])],
                }
            )

        if getattr(response, "text", None):
            data = json.loads(response.text)
            return json.dumps(
                {
                    "confidence": data.get("confidence"),
                    "reasons": data.get("reasons", []),
                }
            )

        raise ValueError("Score response missing parsed payload and text")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ServerError),
    )
    def explain(
        self,
        resume_spans: str,
        job_spans: str,
        must_haves: list[str],
        job_title: str,
    ) -> str:
        """
        Full explanation pass using your long rule block.
        Returns JSON: {"summary": "...", "reasons": [...], "confidence": float}
        """
        sys_prompt = (
            "You are a Principal Technical Recruiter writing a factual briefing for a Hiring Manager. "
            "Focus on evidence-based assessment, not persuasion."
        )

        # This is your long prompt (kept as-is in structure and content, only parsed safely).
        prompt = f"""
Job Title: {job_title}

Resume spans:
{resume_spans}

Job spans:
{job_spans}

Must-haves: {', '.join(must_haves)}

INSTRUCTIONS:
1. **Contextualize for Waymo**: You know this role is at Waymo. Use your internal knowledge of Waymo's stack (Safety-critical C++, real-time systems, large-scale simulation, Bazel) and the AV domain to INFER implicit technical challenges that might be missing from the brief description.
2. **Analyze the Job**: Identify the 3 hardest technical challenges (explicit OR inferred from the Title/Domain).
3. **Analyze the Candidate**: Find specific evidence in the Resume Spans that proves the candidate can solve those EXACT challenges.
4. **Generate the Summary**:
   - You MUST generate a JSON object with a `summary` field.
   - The `summary` field MUST be a Markdown string containing EXACTLY these three sections, in this order:
   
   **Hiring Pitch**
   [2-3 sentences. FACTUAL TONE ONLY. State what the candidate has done and how it relates to the role. NO sales language ("strong contender", "solidify", "brings to bear"). Avoid robotic phrases ("advanced technical capabilities").
    - BAD: "The Candidate is a strong contender for this role and brings 10 years of ADAS experience that solidifies their technical capabilities."
    - GOOD: "The Candidate has 7 years leading ML product development for autonomous vehicles at Motional, including deep learning for 3D object detection. This experience directly applies to the visual reasoning challenges in this role. They hold a PhD in Applied Physics from Caltech."
   ]

   **Why it's a match**
   - [Bullet 1: **Evidence-based mapping**. If you infer a requirement (e.g. "Waymo Planners need MPC"), EXPLAIN THE INFERENCE. e.g., "Waymo Planning requires robust MPC; his PhD thesis on 'Belief-space Guided Navigation' proves deep optimization expertise."]
   - [Bullet 2: Connect specific resume details to job requirements]
   - [Bullet 3]

   **Potential Gaps**
   - [Bullet 1: Mention seniority/domain gaps if any, or "None detected" if perfect fit]

4. **CRITICAL**: Do NOT skip any section. If there are no gaps, write "None detected".
5. **CRITICAL**: Do NOT wrap the content in markdown code blocks (```).

Rules:
- Do NOT include span IDs (e.g. (R1), (J1)) in the output text.
  - **CRITICAL**: Check for Seniority/Career Stage Mismatch.
  - **Year Counting Rules**:
    - Count ONLY full-time industry experience (exclude internships, research roles, and teaching unless they are full-time post-graduation roles).
    - For a candidate who graduated in 2022, they can have at most ~2-3 years of industry experience as of 2025.
    - For each job in the resume, check the date range. If it says "Intern" or is during undergrad/grad school years, it does NOT count.
  - **Seniority Penalties**:
    - If Job title contains "Principal" or "Staff" (e.g. "Principal Software Engineer", "Staff ML Engineer") AND Candidate has < 5 years of full-time industry experience: PENALIZE confidence to max 0.2.
    - If Job title contains "Senior" (but not "Staff" or "Principal") AND Candidate has < 3 years of full-time industry experience: PENALIZE confidence to max 0.4.
    - If Candidate has > 8 years exp and Job is Junior/Entry: PENALIZE confidence to max 0.5.
  - **CRITICAL: Over-Qualification Penalty**:
    - **Level Mapping** (use this for Waymo roles):
      - Job requires "5-7 years" or "7+ years" without "Staff"/"Principal"/"Director" → L4/L5 (Senior SWE)
      - Job requires "8-12 years" or has "Staff" in title → L5/L6 (Staff)
      - Job has "Principal" or "Fellow" in title → L6+ (Principal/Fellow)
      - Job has "Director" or "Head of" in title → Leadership track (not IC)
    - **Over-Qualification Rules**:
      - If Candidate has 20+ years of experience OR previous Director/VP/C-level titles AND Job is L4/L5 (just "Senior SWE", "7+ years req"): PENALIZE confidence to max 0.3.
      - Explain: "Candidate is significantly over-qualified (L6+ profile for an L4/L5 role)."
      - If Candidate has 15+ years experience AND Job is L4/L5: PENALIZE confidence to max 0.4.
      - If Candidate has "Staff" or "Principal" in current title AND Job is standard "Senior" without Staff/Principal prefix: PENALIZE confidence to max 0.5.
  - **PhD + Leadership**: If Candidate has a PhD and 3+ years of leadership/mentorship, they are likely qualified for "Tech Lead" or "Manager" roles. Do NOT penalize for lack of formal "Manager" title if they have this.
  - Explicitly mention seniority mismatch in "Potential Gaps".
- **CRITICAL**: Check for Technical Domain Mismatch.
  - If Job is "Compiler Engineer" and Candidate lacks LLVM/MLIR/Compiler experience: PENALIZE confidence to max 0.4.
  - If Job is "Hardware/FPGA" and Candidate is purely Software: PENALIZE confidence to max 0.4.
  - If Job is "Frontend/Web" and Candidate is purely Backend/ML: PENALIZE confidence to max 0.4.
  - Do NOT let generic skills (Python, C++) override a lack of specific domain expertise.
  - **Practitioner vs. Adjacent Experience Check**:
    - CRITICAL: Distinguish between *doing* the work (verbs: built, coded, designed, implemented, deployed) and *supporting* the work (verbs: hired, sourced, sold, managed project, partnered with).
    - **Title Ambiguity Trap**: If the Candidate's title contains "Talent", "Recruiting", "Sourcing", or "Staffing", they are NOT a Practitioner.
    - If the Job requires a Practitioner and the Candidate is a Recruiter/Sourcer, they are NOT a match.
    - PENALIZE confidence to max 0.1 for this mismatch.

- **Reasons Codes**: You MUST select `code` values ONLY from this list:
  - domain_expertise
  - technical_skills
  - leadership_experience
  - product_commercialization
  - education
  - patents
  - seniority_match
  - cultural_fit
  - safety_critical_experience

- **Weights**: For each reason, assign a `weight` between 0.0 and 1.0.
- **ANONYMIZATION**: NEVER use the candidate's real name. Always refer to them as "The Candidate" or "TC".
"""

        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[sys_prompt, prompt],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExplanationPayload,
            ),
        )

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            summary = parsed.summary or ""
            if summary.strip().startswith("```"):
                summary = summary.replace("```markdown", "").replace("```", "").strip()
            return json.dumps(
                {
                    "summary": summary,
                    "reasons": [r.model_dump() for r in (parsed.reasons or [])],
                    "confidence": float(parsed.confidence),
                }
            )

        if getattr(response, "text", None):
            data = json.loads(response.text)
            summary = data.get("summary", "")
            if summary.strip().startswith("```"):
                summary = summary.replace("```markdown", "").replace("```", "").strip()
            return json.dumps(
                {
                    "summary": summary,
                    "reasons": data.get("reasons", []),
                    "confidence": data.get("confidence"),
                }
            )

        raise ValueError("Explain response missing parsed payload and text")
