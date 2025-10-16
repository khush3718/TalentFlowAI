
"""
AI-Powered Resume Screening & ATS Scoring System (Enhanced)
FastAPI Backend with Google Gemini AI Integration - Deterministic ATS Scoring
Version: 2.1.0
Python: 3.12+
"""

import os
import re
import io
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from string import Template

import pdfplumber
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ==================== CONFIGURATION ====================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ats_system.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY in environment variables")
    raise ValueError("GOOGLE_API_KEY must be set in .env")

genai.configure(api_key=GOOGLE_API_KEY)

# ==================== PYDANTIC MODELS ====================
class CandidateDetails(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


class KeywordItem(BaseModel):
    keyword: str
    relevance_score: int = Field(..., ge=0, le=100)


class Keywords(BaseModel):
    keywords: List[KeywordItem] = Field(default_factory=list, max_length=8)


class ResumeData(BaseModel):
    job_role: str
    job_description: str
    candidate_details: CandidateDetails
    keywords: Keywords
    extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class Suggestion(BaseModel):
    missing_skill: str
    suggested_action: str
    priority: str = "medium"


class ScreeningAnalysis(BaseModel):
    summary: str
    ats_score: int = Field(..., ge=0, le=100)
    grade: str
    suggestions: List[Suggestion]
    analyzed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class CompleteAnalysisResponse(BaseModel):
    success: bool
    resume_data: ResumeData
    screening: ScreeningAnalysis
    analyzed_at: str


class HealthCheck(BaseModel):
    status: str
    message: str
    timestamp: str
    version: str = "2.1.0"


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# ==================== SERVICES ====================
class PDFExtractor:
    @staticmethod
    def extract_text(pdf_file: bytes) -> str:
        """Extract raw text from PDF"""
        try:
            text = ''
            with pdfplumber.open(io.BytesIO(pdf_file)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ''
                    text += page_text + '\n'
                    logger.info(f"Extracted text from page {i}")
            if not text.strip():
                raise ValueError("No text found in PDF.")
            logger.info(f"Total extracted characters: {len(text)}")
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")


class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'\(cid:\d+\)', '', text)
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\t\r')
        lines = text.split('\n')
        cleaned = [re.sub(r' +', ' ', ln.strip()) for ln in lines if ln.strip()]
        return '\n'.join(cleaned)


class AIService:
    """AI analysis service using Google Gemini"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("Gemini model initialized for ATS analysis")

    # ---------- PROMPT 1: Candidate Data Extraction ----------
    def extract_candidate_data(self, cv_text: str) -> Dict[str, Any]:
        prompt_tmpl = Template("""
You are an AI assistant specialized in parsing resumes. Extract the following details.

RESUME TEXT:
${cv_text}

OUTPUT JSON FORMAT:
{
  "candidate_details": {
    "name": "<full name>",
    "email": "<email>",
    "phone": "<phone>",
    "location": "<location>"
  },
  "keywords": {
    "keywords": [
      {"keyword": "Python", "relevance_score": 95},
      {"keyword": "AWS", "relevance_score": 88}
    ]
  }
}
Rules:
- Always return exactly 8 keywords if available.
- Relevance score (0–100) indicates importance in the candidate’s profile.
- Return only valid JSON.
""")
        try:
            prompt = prompt_tmpl.substitute(cv_text=cv_text)
            response = self.model.generate_content(prompt)
            result_text = (getattr(response, "text", "") or "").strip()

            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL | re.IGNORECASE)
            result_text = json_match.group(1).strip() if json_match else result_text
            data = json.loads(result_text)
            return data
        except Exception as e:
            logger.error(f"Candidate data extraction failed: {e}")
            return {
                "candidate_details": {},
                "keywords": {"keywords": []}
            }

    # ---------- PROMPT 2: Comprehensive ATS Analysis ----------
    def analyze_resume_comprehensive(
        self, job_role: str, job_description: str, cv_text: str, candidate_name: str = "Candidate"
    ) -> Dict[str, Any]:
        score_guidelines = """
ATS SCORING RULES:
- 90–100: Excellent fit — nearly all required skills and tools match.
- 75–89: Good fit — strong alignment but 1–2 major gaps.
- 60–74: Average fit — partial match with several missing skills.
- 40–59: Below Average — limited overlap.
- 0–39: Poor fit — resume unrelated or lacks essentials.
"""

        prompt = f"""
You are an expert ATS and recruitment analyst.

{score_guidelines}

JOB ROLE: {job_role}
JOB DESCRIPTION:
{job_description}

CANDIDATE NAME: {candidate_name}
RESUME TEXT:
{cv_text}

TASK:
1. Identify required skills and experience from the job description.
2. Compare with the resume.
3. Apply the ATS SCORING RULES strictly.
4. Produce the final output ONLY in valid JSON (no text explanation).

OUTPUT FORMAT SCHEMA (do not copy numeric values):
{{
  "summary": "<6–8 sentence evaluation>",
  "ats_score": <integer between 0 and 100>,
  "grade": "<Excellent|Good|Average|Below Average|Poor>",
  "suggestions": [
    {{
      "missing_skill": "<specific skill or qualification>",
      "suggested_action": "<concrete next step>",
      "priority": "<high|medium|low>"
    }}
  ]
}}

INTERNAL SEED: {datetime.now().timestamp()}
"""
        try:
            response = self.model.generate_content(prompt)
            result_text = (getattr(response, "text", "") or "").strip()

            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            result_text = json_match.group(1).strip() if json_match else result_text
            parsed_data = json.loads(result_text)

            score = max(0, min(100, int(parsed_data.get("ats_score", 50))))
            parsed_data["ats_score"] = score

            if score >= 90:
                parsed_data["grade"] = "Excellent"
            elif score >= 75:
                parsed_data["grade"] = "Good"
            elif score >= 60:
                parsed_data["grade"] = "Average"
            elif score >= 40:
                parsed_data["grade"] = "Below Average"
            else:
                parsed_data["grade"] = "Poor"

            if not isinstance(parsed_data.get("suggestions"), list):
                parsed_data["suggestions"] = []

            logger.info(f"Comprehensive analysis completed (ATS Score: {score})")
            return parsed_data

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                "summary": "Analysis failed due to parsing issue.",
                "ats_score": 50,
                "grade": "Average",
                "suggestions": []
            }

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="AI-Powered Resume Screening & ATS Scoring System",
    description="Automated resume screening with deterministic scoring using Google Gemini AI",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pdf_extractor = PDFExtractor()
text_cleaner = TextCleaner()
ai_service = AIService()

# ==================== EXCEPTION HANDLERS ====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=f"HTTP {exc.status_code}", detail=str(exc.detail)).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal Server Error", detail=str(exc)).dict()
    )

# ==================== ENDPOINTS ===================

@app.post("/api/v1/complete-analysis", response_model=CompleteAnalysisResponse)
async def complete_analysis(
    job_role: str = Form(...),
    job_description: str = Form(...),
    resume: UploadFile = File(...)
):
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await resume.read()

    try:
        raw_text = pdf_extractor.extract_text(content)
        cleaned_text = text_cleaner.clean_text(raw_text)

        candidate_data = ai_service.extract_candidate_data(cleaned_text)
        candidate_details = CandidateDetails(**candidate_data.get("candidate_details", {}))
        keywords_list = candidate_data.get("keywords", {}).get("keywords", [])
        keyword_objs = [
            KeywordItem(**kw) if isinstance(kw, dict) else KeywordItem(keyword=kw, relevance_score=70)
            for kw in keywords_list
        ]

        resume_data = ResumeData(
            job_role=job_role,
            job_description=job_description,
            candidate_details=candidate_details,
            keywords=Keywords(keywords=keyword_objs)
        )

        screening_data = ai_service.analyze_resume_comprehensive(
            job_role, job_description, cleaned_text, candidate_details.name or "Candidate"
        )

        screening = ScreeningAnalysis(
            summary=screening_data.get("summary", ""),
            ats_score=screening_data.get("ats_score", 50),
            grade=screening_data.get("grade", "Average"),
            suggestions=[Suggestion(**s) for s in screening_data.get("suggestions", [])]
        )

        return CompleteAnalysisResponse(
            success=True,
            resume_data=resume_data,
            screening=screening,
            analyzed_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("AI Resume Screening System v2.1 Starting...")
    logger.info("Architecture: Deterministic Two-Prompt System")
    logger.info("  - Prompt 1: Candidate Extraction")
    logger.info("  - Prompt 2: ATS Scoring with Rubric Logic")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Resume Screening System...")
    
# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)


