"""Resume parsing helpers."""
from __future__ import annotations

import re


from google import genai
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from .config import get_settings

_settings = get_settings()
_client = genai.Client(api_key=_settings.gemini_api_key)

class SkillsResponse(BaseModel):
    skills: list[str]

class Experience(BaseModel):
    title: str | None
    company: str | None
    duration: str | None
    description: str | None

class Education(BaseModel):
    degree: str | None
    school: str | None
    year: str | None

class CandidateInfo(BaseModel):
    name: str | None
    email: str | None
    summary: str | None
    experience: list[Experience]
    education: list[Education]

import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(12),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(ServerError)
)
def extract_candidate_info(text: str) -> CandidateInfo:
    """Extract structured info from resume text using Gemini."""
    prompt = f"""
    You are an expert resume parser. Extract the following information from the resume text below.
    Return the result as a valid JSON object matching the schema.
    
    Resume Text:
    {text[:20000]}
    """
    try:
        try:
            response = _client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": CandidateInfo,
                },
            )
            return response.parsed
        except ServerError:
            logger.warning("Gemini 2.5-flash-lite overloaded, falling back to 2.5-flash for candidate info")
            response = _client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": CandidateInfo,
                },
            )
            return response.parsed
    except Exception as e:
        print(f"Candidate info extraction failed: {e}")
        # Return empty/default info on failure
        return CandidateInfo(name="Unknown Candidate", email="unknown@example.com", summary="", experience=[], education=[])

@retry(
    stop=stop_after_attempt(12),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(ServerError)
)
def extract_skills(text: str) -> list[str]:
    """Extract skills using Gemini."""
    prompt = (
        "Extract a list of technical and professional skills from the following resume text. "
        "Return only the skills as a list of strings. "
        "Do not include addresses, dates, or company names. "
        "Normalize skills to title case (e.g., 'Python', 'Project Management')."
    )
    
    try:
        # Use the stronger model for skills to ensure we capture everything
        model_name = "gemini-2.5-flash"
        logger.info(f"Extracting skills with model: {model_name}")
        
        try:
            response = _client.models.generate_content(
                model=model_name,
                contents=[prompt, text],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SkillsResponse,
                ),
            )
        except ServerError:
            logger.warning("Gemini 2.5-flash overloaded, falling back to 2.5-flash-lite for skills")
            response = _client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[prompt, text],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SkillsResponse,
                ),
            )

        skills = response.parsed.skills
        if not skills:
            logger.warning("LLM returned empty skills list, using fallback.")
            return _fallback_extract_skills(text)
            
        logger.info(f"Extracted {len(skills)} skills")
        return skills
    except Exception as e:
        logger.error(f"Skill extraction failed: {e}", exc_info=True)
        # Fallback to simple extraction if API fails
        return _fallback_extract_skills(text)

def _fallback_extract_skills(text: str) -> list[str]:
    """Very small heuristic skill extractor based on comma-separated tokens."""
    candidates: set[str] = set()
    for line in text.splitlines():
        if ":" in line:
            _, rhs = line.split(":", 1)
            tokens = re.split(r"[,;/]", rhs)
        else:
            tokens = re.split(r"[,;/]", line)
        for token in tokens:
            cleaned = token.strip().lower()
            if cleaned and len(cleaned) <= 50:
                candidates.add(cleaned)
    return sorted(candidates)


def summarize_text(text: str, max_chars: int = 20000) -> str:
    """Trim the resume to a manageable length for embedding."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def chunk_with_labels(text: str, prefix: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    labeled = [f"{prefix}{idx}:{line}" for idx, line in enumerate(lines, start=1)]
    return "\n".join(labeled)


class SearchQueries(BaseModel):
    queries: list[str]


@retry(
    retry=retry_if_exception_type(ServerError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def generate_search_queries(text: str) -> list[str]:
    """Generate diverse search queries to maximize job retrieval coverage."""
    prompt = """
    Analyze this resume and generate 5 distinct search queries to find the best job matches.
    
    Strategy:
    1.  **Role-based**: "Senior Backend Engineer", "Staff Data Scientist"
    2.  **Skill-based**: "Python Machine Learning Expert", "C++ Robotics Engineer"
    3.  **Domain-based**: "Autonomous Driving Perception", "AdTech Optimization"
    4.  **Seniority-based**: "Principal Engineer", "Engineering Manager"
    
    Return a JSON list of strings.
    """
    
    try:
        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, text],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SearchQueries,
            ),
        )
        queries = response.parsed.queries
        logger.info(f"Generated search queries: {queries}")
        return queries
    except Exception as e:
        logger.error(f"Error generating search queries: {e}")
        # Fallback: just use the summary
        return [summarize_text(text)]
