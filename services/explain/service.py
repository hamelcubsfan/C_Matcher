"""Gemini-powered explanation generation."""
from __future__ import annotations

import json
from typing import List

from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError


class ExplanationService:
    def __init__(self):
        from services.shared.config import get_settings
        settings = get_settings()
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "reasons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "enum": [
                                    "domain_expertise",
                                    "technical_skills",
                                    "leadership_experience",
                                    "product_commercialization",
                                    "education",
                                    "patents",
                                    "seniority_match",
                                    "cultural_fit",
                                    "safety_critical_experience",
                                ],
                            },
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "weight": {"type": "number"},
                        },
                        "required": ["code", "weight"],
                    },
                },
                "confidence": {"type": "number"},
            },
            "required": ["summary", "reasons", "confidence"],
        }

    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(ServerError)
    )
    def explain(
        self,
        resume_spans: list[str],
        job_spans: list[str],
        must_haves: list[str],
    ) -> str:
        sys_prompt = "You are a Principal Technical Recruiter pitching a candidate to a skeptical Hiring Manager. You hate fluff. You only care about concrete evidence."
        prompt = f"""
Resume spans:
{resume_spans}

Job spans:
{job_spans}

Must-haves: {', '.join(must_haves)}
Must-haves: {', '.join(must_haves)}

INSTRUCTIONS:
1. You MUST generate a JSON object with a `summary` field.
2. The `summary` field MUST be a Markdown string containing EXACTLY these three sections, in this order:
   
   **Hiring Pitch**
   [2-3 sentences. Connect specific resume details to job requirements. NO FLUFF.]

   **Why it's a match**
   - [Bullet 1: Evidence-based mapping]
   - [Bullet 2]

   **Potential Gaps**
   - [Bullet 1: Mention seniority/domain gaps if any, or "None detected" if perfect fit]

3. **CRITICAL**: Do NOT skip any section. If there are no gaps, write "None detected".
4. **CRITICAL**: Do NOT wrap the content in markdown code blocks (```).

Rules:
- Do NOT include span IDs (e.g. (R1), (J1)) in the output text.
- **CRITICAL**: Check for Seniority/Career Stage Mismatch.
  - Evaluate seniority based on **YEARS OF EXPERIENCE**, not just titles (especially for "Founding Engineer" or "Lead" at startups).
  - If Candidate has < 3 years exp and Job is Senior/Staff/Principal: PENALIZE confidence to max 0.5.
  - If Candidate has > 8 years exp and Job is Junior/Entry: PENALIZE confidence to max 0.5.
  - **Overqualification Check**: If Candidate has > 15 years experience (or Director/VP titles) and Job is a mid-level IC role (e.g. just "Software Engineer" or "Machine Learning Engineer" without Senior/Staff/Principal prefix): PENALIZE confidence to max 0.6.
  - **PhD + Leadership**: If Candidate has a PhD and 3+ years of leadership/mentorship, they are likely qualified for "Tech Lead" or "Manager" roles. Do NOT penalize for lack of formal "Manager" title if they have this.
  - Explicitly mention this mismatch in the summary.
- **CRITICAL**: Check for Technical Domain Mismatch.
  - If Job is "Compiler Engineer" and Candidate lacks LLVM/MLIR/Compiler experience: PENALIZE confidence to max 0.4.
  - If Job is "Hardware/FPGA" and Candidate is purely Software: PENALIZE confidence to max 0.4.
  - If Job is "Frontend/Web" and Candidate is purely Backend/ML: PENALIZE confidence to max 0.4.
  - **Recruiting/Talent experience is NOT Engineering experience**.
    - **CRITICAL EXCEPTION**: If Job Title contains "Recruiter", "Talent", "Sourcing", or "Staffing", this IS a relevant role. Do NOT penalize.
    - Otherwise, if Candidate is a Recruiter and Job is an Engineer/Developer/Scientist: PENALIZE confidence to max 0.2.
  - Do NOT let generic skills (Python, C++) override a lack of specific domain expertise.

- **Reasons Codes**: You MUST select `code` values ONLY from this list:
  - `domain_expertise`
  - `technical_skills`
  - `leadership_experience`
  - `product_commercialization`
  - `education`
  - `patents`
  - `seniority_match`
  - `cultural_fit`
  - `safety_critical_experience`

- **ANONYMIZATION**: NEVER use the candidate's real name. Always refer to them as "The Candidate" or "TC".
"""
        try:
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[sys_prompt, prompt],
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=self.schema,
                    ),
                )
            except ServerError:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Gemini 2.5-flash-lite overloaded, falling back to 2.5-flash for explanation")
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[sys_prompt, prompt],
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=self.schema,
                    ),
                )
            data = json.loads(response.text)
            summary = data.get("summary", "")
            # Guard against the model wrapping the inner string in markdown code blocks
            if summary.strip().startswith("```"):
                summary = summary.replace("```markdown", "").replace("```", "").strip()
            
            reasons = data.get("reasons", [])
            confidence = data.get("confidence")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Explanation generation failed: {e}", exc_info=True)
            summary = "Automated explanation unavailable."
            reasons = []
            confidence = None
        return json.dumps(
            {
                "summary": summary,
                "reasons": reasons,
                "confidence": confidence,
            }
        )
