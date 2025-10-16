# """
# AI-Powered Resume Screening & ATS Scoring System (Simplified)
# FastAPI Backend with Google Gemini AI Integration - Two Prompt Version
# Version: 2.0.0
# Python: 3.12+

# Installation:
# pip install fastapi uvicorn pdfplumber pydantic python-dotenv google-generativeai

# Run:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# """

# import os
# import re
# import io
# import json
# import logging
# from typing import Optional, Dict, Any, List
# from datetime import datetime
# from string import Template

# import pdfplumber
# import google.generativeai as genai
# from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field, field_validator
# from dotenv import load_dotenv

# # ==================== CONFIGURATION ====================
# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('ats_system.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize Google Gemini AI
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     logger.error("GOOGLE_API_KEY not found in environment variables")
#     raise ValueError("GOOGLE_API_KEY must be set in .env file")

# genai.configure(api_key=GOOGLE_API_KEY)

# # ==================== PYDANTIC MODELS ====================
# class CandidateDetails(BaseModel):
#     """Model for candidate personal details"""
#     name: Optional[str] = None
#     email: Optional[str] = None
#     phone: Optional[str] = None
#     location: Optional[str] = None

# class KeywordItem(BaseModel):
#     """Model for individual keyword with relevance score"""
#     keyword: str
#     relevance_score: int = Field(..., ge=0, le=100, description="Relevance score 0-100")

# class Keywords(BaseModel):
#     """Model for extracted keywords with relevance scores"""
#     keywords: List[KeywordItem] = Field(default_factory=list, description="Top 8 most relevant keywords with scores", max_length=8)

# class ResumeData(BaseModel):
#     """Model for extracted resume data"""
#     job_role: str
#     job_description: str
#     candidate_details: CandidateDetails
#     keywords: Keywords
#     extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())

# class Suggestion(BaseModel):
#     """Model for individual suggestion"""
#     missing_skill: str
#     suggested_action: str
#     priority: str = "medium"

# class ScreeningAnalysis(BaseModel):
#     """Model for comprehensive screening analysis"""
#     summary: str
#     ats_score: int = Field(..., ge=0, le=100)
#     grade: str
#     suggestions: List[Suggestion]
#     analyzed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

# class CompleteAnalysisResponse(BaseModel):
#     """Complete analysis response"""
#     success: bool
#     resume_data: ResumeData
#     screening: ScreeningAnalysis
#     analyzed_at: str

# class HealthCheck(BaseModel):
#     """Health check response model"""
#     status: str
#     message: str
#     timestamp: str
#     version: str = "2.0.0"

# class ErrorResponse(BaseModel):
#     """Error response model"""
#     error: str
#     detail: str
#     timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# # ==================== SERVICES ====================
# class PDFExtractor:
#     """Service for extracting text from PDF files"""
    
#     @staticmethod
#     def extract_text(pdf_file: bytes) -> str:
#         """Extract text from PDF file"""
#         try:
#             text = ''
#             with pdfplumber.open(io.BytesIO(pdf_file)) as pdf:
#                 for page_num, page in enumerate(pdf.pages, 1):
#                     page_text = page.extract_text()
#                     if page_text:
#                         text += page_text + '\n'
#                     logger.info(f"Extracted text from page {page_num}")
            
#             if not text.strip():
#                 raise ValueError("No text could be extracted from PDF")
            
#             logger.info(f"Successfully extracted {len(text)} characters from PDF")
#             return text
#         except Exception as e:
#             logger.error(f"PDF extraction failed: {str(e)}")
#             raise ValueError(f"Failed to extract text from PDF: {str(e)}")

# class TextCleaner:
#     """Service for cleaning and normalizing text"""
    
#     @staticmethod
#     def clean_text(text: str) -> str:
#         """Clean and normalize text"""
#         if not text:
#             return ""
        
#         # Remove (cid:XXX) artifacts
#         text = re.sub(r'\(cid:\d+\)', '', text)
        
#         # Remove problematic control characters but keep newlines and tabs
#         text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
        
#         # Normalize multiple spaces to single space
#         lines = text.split('\n')
#         cleaned_lines = [re.sub(r' +', ' ', line.strip()) for line in lines]
#         text = '\n'.join(line for line in cleaned_lines if line)
        
#         text = text.strip()
#         logger.info(f"Text cleaned: {len(text)} characters")
#         return text

# class AIService:
#     """Service for AI-powered analysis using Google Gemini"""
    
#     def __init__(self):
#         """Initialize AI service with Gemini model"""
#         self.model = genai.GenerativeModel('gemini-2.0-flash')
#         logger.info("AI Service initialized with Gemini model")
    
#     def extract_candidate_data(self, cv_text: str) -> Dict[str, Any]:
#         """
#         PROMPT 1: Extract candidate details and keywords from resume
        
#         Args:
#             cv_text: Candidate CV text
            
#         Returns:
#             Dictionary with candidate details and keywords
#         """
#         try:
#             # Use Template to avoid f-string brace conflicts with JSON examples
#             prompt_tmpl = Template("""
# You are an AI assistant specialized in parsing resumes. Extract the following information from the resume below.

# RESUME TEXT:
# ${cv_text}

# TASK:
# Extract and return the following information in JSON format:

# 1. **Candidate Details:**
#    - name: Full name of the candidate
#    - email: Email address
#    - phone: Phone number (with country code if available)
#    - location: location (city, state, country)

# 2. **Keywords:**
#    - keywords: A list of the TOP 8 most important and relevant keywords from the resume. Each keyword should have:
#      - keyword: The technology, tool, skill, or certification name
#      - relevance_score: A score from 0-100 indicating how critical this keyword is (based on prominence in resume, market demand, and relevance to typical roles)
#    - Choose keywords that are most significant for the candidate's profile
#    - Prioritize: core programming languages, major frameworks, key tools/platforms, critical certifications
#    - Examples: [{"keyword": "Python", "relevance_score": 95}, {"keyword": "React", "relevance_score": 88}]

# IMPORTANT:
# - If any field is not found, use null for strings or empty array [] for lists
# - Extract EXACTLY 8 keywords - no more, no less (if resume has fewer than 8, extract what's available)
# - Rank keywords by importance: primary tech stack > frameworks > tools > certifications > soft skills
# - Relevance score should reflect how critical the keyword is to the candidate's profile
# - Return ONLY valid JSON, no additional text

# OUTPUT FORMAT:
# {
#   "candidate_details": {
#     "name": "John Doe",
#     "email": "john@example.com",
#     "phone": "+1234567890",
#     "location": "Ahmedabad, Gujarat, India"
#   },
#   "keywords": {
#     "keywords": [
#       {"keyword": "Python", "relevance_score": 95},
#       {"keyword": "React", "relevance_score": 90},
#       {"keyword": "Node.js", "relevance_score": 85},
#       {"keyword": "Docker", "relevance_score": 80},
#       {"keyword": "AWS", "relevance_score": 78},
#       {"keyword": "Machine Learning", "relevance_score": 75},
#       {"keyword": "PostgreSQL", "relevance_score": 70},
#       {"keyword": "Git", "relevance_score": 65}
#     ]
#   }
# }
# """)
#             prompt = prompt_tmpl.substitute(cv_text=cv_text)

#             response = self.model.generate_content(prompt)
#             result_text = (getattr(response, "text", "") or "").strip()
            
#             # Extract JSON from response (handle markdown code blocks)
#             json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL | re.IGNORECASE)
#             if json_match:
#                 result_text = json_match.group(1).strip()
#             else:
#                 # Try to find the first JSON object in the text
#                 json_match = re.search(r'\{(?:[^{}]|(?R))*\}', result_text, re.DOTALL)
#                 if json_match:
#                     result_text = json_match.group(0).strip()

#             # Parse JSON
#             parsed_data = json.loads(result_text)
#             logger.info("Candidate data extraction completed successfully")
#             return parsed_data
            
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to parse JSON response: {e}")
#             logger.error(f"Response text (first 500 chars): {result_text[:500]!r}")
#             # Return empty structure on parse failure
#             return {
#                 "candidate_details": {
#                     "name": None,
#                     "email": None,
#                     "phone": None,
#                     "location": None
#                 },
#                 "keywords": {
#                     "keywords": []
#                 }
#             }
#         except Exception as e:
#             logger.error(f"Candidate data extraction failed: {str(e)}", exc_info=True)
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Candidate data extraction failed: {str(e)}"
#             )   
#     def analyze_resume_comprehensive(self, job_role: str, job_description: str, 
#                                     cv_text: str, candidate_name: str = "Candidate") -> Dict[str, Any]:
#         """
#         PROMPT 2: Comprehensive screening analysis with ATS score and suggestions
        
#         Args:
#             job_role: Job role title
#             job_description: Job description
#             cv_text: Candidate CV text
#             candidate_name: Name of candidate for personalized summary
            
#         Returns:
#             Dictionary with comprehensive analysis
#         """
#         try:
#             prompt = f"""
# You are an expert ATS (Applicant Tracking System) and recruitment analyst. Perform a comprehensive analysis of the candidate's resume against the job requirements.

# JOB ROLE: {job_role}

# JOB DESCRIPTION:
# {job_description}

# CANDIDATE RESUME:
# {cv_text}

# CANDIDATE NAME: {candidate_name}

# TASK:
# Provide a comprehensive analysis in JSON format with the following:

# 1. **summary**: A detailed 6-8 sentence analysis covering:
#    - Overall fit for the role (good fit / partial fit / not a good fit)
#    - Key strengths that align with the job requirements
#    - Major gaps or missing qualifications
#    - Experience level assessment
#    - Notable achievements or projects relevant to the role
#    - Final recommendation (Strongly Recommend / Recommend / Consider / Not Recommended)

# 2. **ats_score**: An ATS score from 0-100 based on how well the resume matches the job description.

# 3. **grade**: Text grade based on ATS score:
#    - 90-100: "Excellent"
#    - 75-89: "Good"
#    - 60-74: "Average"
#    - 40-59: "Below Average"
#    - 0-39: "Poor"

# 4. **suggestions**: Array of 5-8 actionable improvement suggestions. Each suggestion should have:
#    - missing_skill: Specific skill, technology, or qualification missing or underrepresented
#    - suggested_action: Concrete action the candidate can take (one sentence)
#    - priority: "high" (critical for role), "medium" (important), or "low" (nice to have)

# IMPORTANT:
# - Be objective and thorough in your analysis
# - Base suggestions on actual gaps between resume and job requirements
# - Don't suggest skills already present in the resume
# - Make suggestions specific and actionable
# - Consider both technical and soft skills
# - Return ONLY valid JSON, no additional text

# OUTPUT FORMAT:
# {{
#   "summary": "Detailed analysis text here...",
#   "ats_score": 75,
#   "grade": "Good",
#   "suggestions": [
#     {{
#       "missing_skill": "Kubernetes",
#       "suggested_action": "Complete a Kubernetes certification course and deploy a sample microservices application",
#       "priority": "high"
#     }}
#   ]
# }}
# """
            
#             response = self.model.generate_content(prompt)
#             result_text = response.text.strip()
            
#             # Extract JSON from response
#             json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
#             if json_match:
#                 result_text = json_match.group(1)
#             else:
#                 json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
#                 if json_match:
#                     result_text = json_match.group(0)
            
#             # Parse JSON
#             parsed_data = json.loads(result_text)
            
#             # Validate and normalize ATS score
#             ats_score = parsed_data.get('ats_score', 50)
#             ats_score = max(0, min(100, int(ats_score)))
#             parsed_data['ats_score'] = ats_score
            
#             # Validate grade
#             if parsed_data.get('grade') not in ["Excellent", "Good", "Average", "Below Average", "Poor"]:
#                 if ats_score >= 90:
#                     parsed_data['grade'] = "Excellent"
#                 elif ats_score >= 75:
#                     parsed_data['grade'] = "Good"
#                 elif ats_score >= 60:
#                     parsed_data['grade'] = "Average"
#                 elif ats_score >= 40:
#                     parsed_data['grade'] = "Below Average"
#                 else:
#                     parsed_data['grade'] = "Poor"
            
#             # Ensure suggestions is a list
#             if not isinstance(parsed_data.get('suggestions'), list):
#                 parsed_data['suggestions'] = []
            
#             logger.info(f"Comprehensive analysis completed: ATS Score = {ats_score}")
#             return parsed_data
            
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to parse JSON response: {str(e)}")
#             logger.error(f"Response text: {result_text}")
#             # Return default structure on parse failure
#             return {
#                 "summary": "Analysis could not be completed due to a parsing error. Please try again.",
#                 "ats_score": 50,
#                 "grade": "Average",
#                 "suggestions": []
#             }
#         except Exception as e:
#             logger.error(f"Comprehensive analysis failed: {str(e)}")
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Comprehensive analysis failed: {str(e)}"
#             )

# # ==================== FASTAPI APP ====================
# app = FastAPI(
#     title="AI-Powered Resume Screening & ATS Scoring System (Simplified)",
#     description="Automated resume screening with two-prompt architecture using Google Gemini AI",
#     version="2.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize services
# pdf_extractor = PDFExtractor()
# text_cleaner = TextCleaner()
# ai_service = AIService()

# # ==================== EXCEPTION HANDLERS ====================
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     """Handle HTTP exceptions"""
#     logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
#     return JSONResponse(
#         status_code=exc.status_code,
#         content=ErrorResponse(
#             error=f"HTTP {exc.status_code}",
#             detail=str(exc.detail)
#         ).dict()
#     )

# @app.exception_handler(Exception)
# async def general_exception_handler(request, exc):
#     """Handle general exceptions"""
#     logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
#     return JSONResponse(
#         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         content=ErrorResponse(
#             error="Internal Server Error",
#             detail="An unexpected error occurred. Please try again."
#         ).dict()
#     )

# # ==================== ENDPOINTS ====================
# @app.get("/", response_model=HealthCheck, tags=["Health"])
# async def health_check():
#     """Health check endpoint"""
#     logger.info("Health check requested")
#     return HealthCheck(
#         status="healthy",
#         message="AI-Powered Resume Screening & ATS Scoring System (v2.0) is running",
#         timestamp=datetime.now().isoformat()
#     )

# @app.post("/api/v1/complete-analysis", response_model=CompleteAnalysisResponse, tags=["AI Analysis"])
# async def complete_analysis(
#     job_role: str = Form(...),
#     job_description: str = Form(...),
#     resume: UploadFile = File(...)
# ):
#     """
#     Complete end-to-end resume analysis using two AI prompts:
    
#     1. Extract candidate details (name, email, phone, location) and keywords
#     2. Perform comprehensive screening with detailed summary, ATS score, and suggestions
    
#     Returns:
#     - resume_data: Contains candidate details and extracted keywords
#     - screening: Contains detailed summary, ATS score, grade, and improvement suggestions
#     """
#     logger.info(f"Complete analysis requested for role: {job_role}")
    
#     if not resume.filename.endswith('.pdf'):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Only PDF files are supported"
#         )
    
#     contents = await resume.read()
    
#     try:
#         # Step 1: Extract and clean resume text
#         resume_text = pdf_extractor.extract_text(contents)
#         cleaned_text = text_cleaner.clean_text(resume_text)
        
#         # Step 2: PROMPT 1 - Extract candidate details and keywords
#         candidate_data = ai_service.extract_candidate_data(cleaned_text)
        
#         candidate_details = CandidateDetails(**candidate_data.get('candidate_details', {}))
        
#         # Process keywords - convert dict format to KeywordItem objects
#         keywords_data = candidate_data.get('keywords', {})
#         keywords_list = keywords_data.get('keywords', [])
        
#         # Ensure keywords are in correct format
#         processed_keywords = []
#         for kw in keywords_list:
#             if isinstance(kw, dict):
#                 processed_keywords.append(KeywordItem(**kw))
#             elif isinstance(kw, str):
#                 # Fallback if string format is returned
#                 processed_keywords.append(KeywordItem(keyword=kw, relevance_score=70))
        
#         keywords = Keywords(keywords=processed_keywords)
        
#         resume_data = ResumeData(
#             job_role=job_role.strip(),
#             job_description=job_description.strip(),
#             candidate_details=candidate_details,
#             keywords=keywords
#         )
        
#         # Step 3: PROMPT 2 - Comprehensive screening analysis
#         candidate_name = candidate_details.name or "Candidate"
#         screening_data = ai_service.analyze_resume_comprehensive(
#             job_role=job_role.strip(),
#             job_description=job_description.strip(),
#             cv_text=cleaned_text,
#             candidate_name=candidate_name
#         )
        
#         screening_analysis = ScreeningAnalysis(
#             summary=screening_data.get('summary', ''),
#             ats_score=screening_data.get('ats_score', 50),
#             grade=screening_data.get('grade', 'Average'),
#             suggestions=[Suggestion(**s) for s in screening_data.get('suggestions', [])]
#         )
        
#         logger.info("Complete analysis finished successfully")
        
#         return CompleteAnalysisResponse(
#             success=True,
#             resume_data=resume_data,
#             screening=screening_analysis,
#             analyzed_at=datetime.now().isoformat()
#         )
        
#     except Exception as e:
#         logger.error(f"Complete analysis failed: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Analysis failed: {str(e)}"
#         )

# # ==================== STARTUP/SHUTDOWN ====================
# @app.on_event("startup")
# async def startup_event():
#     """Startup event handler"""
#     logger.info("=" * 50)
#     logger.info("AI Resume Screening System v2.0 Starting...")
#     logger.info("Architecture: Two-Prompt System")
#     logger.info("  - Prompt 1: Candidate Details & Keywords Extraction")
#     logger.info("  - Prompt 2: Comprehensive Screening Analysis")
#     logger.info(f"Gemini AI Model: gemini-2.5-flash")
#     logger.info("All services initialized successfully")
#     logger.info("=" * 50)

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Shutdown event handler"""
#     logger.info("AI Resume Screening System Shutting Down...")

# # ==================== MAIN ====================
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )

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

# ==================== ENDPOINTS ====================
@app.get("/", response_model=HealthCheck)
async def health_check():
    return HealthCheck(
        status="healthy",
        message="AI Resume Screening & ATS Scoring System v2.1 is operational",
        timestamp=datetime.now().isoformat()
    )


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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
