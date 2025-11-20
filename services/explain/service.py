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
                            "code": {"type": "string"},
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "weight": {"type": "number"},
                        },
                        "required": ["code", "weight"],
                    },
                },
                "gaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "note": {"type": "string"},
                        },
                        "required": ["code", "note"],
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
        sys_prompt = "Write a concise recruiter note. Do NOT use citations."
        prompt = f"""
Resume spans:
{resume_spans}

Job spans:
{job_spans}

Must-haves: {', '.join(must_haves)}
Rules:
- 120 to 180 words
- Do NOT include span IDs (e.g. (R1), (J1)) in the output text.
- Call out obvious gaps
- **CRITICAL**: Check for Seniority/Career Stage Mismatch.
  - Evaluate seniority based on **YEARS OF EXPERIENCE**, not just titles (especially for "Founding Engineer" or "Lead" at startups).
  - If Candidate has < 3 years exp and Job is Senior/Staff/Principal: PENALIZE confidence to max 0.5.
  - If Candidate has > 8 years exp and Job is Junior/Entry: PENALIZE confidence to max 0.5.
  - **PhD + Leadership**: If Candidate has a PhD and 3+ years of leadership/mentorship, they are likely qualified for "Tech Lead" or "Manager" roles. Do NOT penalize for lack of formal "Manager" title if they have this.
  - Explicitly mention this mismatch in the summary.
- **CRITICAL**: Check for Technical Domain Mismatch.
  - If Job is "Compiler Engineer" and Candidate lacks LLVM/MLIR/Compiler experience: PENALIZE confidence to max 0.4.
  - If Job is "Hardware/FPGA" and Candidate is purely Software: PENALIZE confidence to max 0.4.
  - If Job is "Frontend/Web" and Candidate is purely Backend/ML: PENALIZE confidence to max 0.4.
  - **Recruiting/Talent experience is NOT Engineering experience**.
    - If Candidate is a Recruiter and Job is an Engineer/Developer/Scientist: PENALIZE confidence to max 0.2.
    - **EXCEPTION**: If Job Title contains "Recruiter", "Talent", "Sourcing", or "Staffing", it is NOT an Engineering role. Do NOT penalize.
  - Do NOT let generic skills (Python, C++) override a lack of specific domain expertise.

 Formatting:
- The `summary` field MUST be a Markdown-formatted string with these exact sections:
  
  **Hiring Pitch**
  [2-3 sentences. Factual, non-salesy summary. MUST mention specific companies, teams, algorithms, or research topics from their background (e.g. "Experience at Amazon Ads SimEx", "PhD research in Multi-Armed Bandits"). Avoid generic fluff.]

  **Why it's a match**
  - [Bullet point 1: Connect specific resume details (e.g. "NDSEG Fellow", "developed ad selection algos") to job requirements]
  - [Bullet point 2]
  - [Bullet point 3]

  **Potential Gaps**
  - [Bullet point 1 (mention seniority/domain gaps if any)]
  - [Bullet point 2]

- Use clear Markdown bullets. Keep it clean and readable.
- Total length under 200 words.
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
            reasons = data.get("reasons", [])
            gaps = data.get("gaps", [])
            confidence = data.get("confidence")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Explanation generation failed: {e}", exc_info=True)
            summary = "Automated explanation unavailable."
            reasons = []
            gaps = []
            confidence = None
        return json.dumps(
            {
                "summary": summary,
                "reasons": reasons,
                "gaps": gaps,
                "confidence": confidence,
            }
        )
