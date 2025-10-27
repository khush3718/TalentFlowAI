# AI-Powered Resume Screening & ATS Scoring System

A full‑stack application for automated resume screening with ATS-style scoring. The backend is built with FastAPI and integrates Google Gemini for two-stage analysis: candidate data extraction and comprehensive ATS scoring. The frontend is a Next.js app that provides an interactive UI for uploading resumes and viewing results.

Version: 2.1.0

## Features
- PDF resume ingestion with robust text extraction (pdfplumber)
- Text cleaning/normalization for better LLM parsing
- Two‑prompt analysis with Google Gemini:
  - Prompt 1: Extract candidate details and top 8 keywords with relevance scores
  - Prompt 2: Comprehensive ATS analysis with summary, score, grade, and suggestions
- Deterministic score normalization and grade mapping
- Typed response models with Pydantic
- CORS-enabled API suitable for browser clients
- Health check endpoint and structured error handling

## Project Structure
- api/
  - main.py — FastAPI app with endpoints and Gemini prompts
  - requirements.txt — Python dependencies
- UI/ — Next.js frontend (app router) and components

## Prerequisites
- Python 3.12+
- Node.js 18+
- A Google Generative AI API key (GOOGLE_API_KEY)

## Setup

### 1) Backend (FastAPI)
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   pip install -r api/requirements.txt
3. Create a .env file in the repository root (same folder as api/) with:
   GOOGLE_API_KEY=your_api_key_here
4. Start the API server from the api directory path:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

API will run at http://localhost:8000. Interactive docs at /docs and /redoc.

Key Python packages: fastapi, uvicorn, pdfplumber, pydantic, python-dotenv, google-generativeai

### 2) Frontend (Next.js)
1. From the UI directory, install dependencies:
   npm install
2. Run the dev server:
   npm run dev

Frontend will run at http://localhost:3000.

Note: Ensure CORS is allowed (already configured in the API) and the frontend calls the API base URL (e.g., http://localhost:8000).

## API Overview

Base URL: http://localhost:8000

- GET / — Health check
  - Response example:
    {
      "status": "healthy",
      "message": "AI Resume Screening & ATS Scoring System v2.1 is operational",
      "timestamp": "2024-01-01T12:00:00.000Z",
      "version": "2.1.0"
    }

- POST /api/v1/complete-analysis — Resume analysis
  - Form fields (multipart/form-data):
    - job_role: string (required)
    - job_description: string (required)
    - resume: file (PDF, required)
  - Response shape:
```
{
  "success": true,
  "resume_data": {
    "job_role": "...",
    "job_description": "...",
    "candidate_details": {
      "name": "...",
      "email": "...",
      "phone": "...",
      "location": "..."
    },
    "keywords": {
      "keywords": [
        {"keyword": "Python", "relevance_score": 95},
        "... up to 8 items ..."
      ]
    },
    "extracted_at": "ISO8601"
  },
  "screening": {
    "summary": "...",
    "ats_score": 0-100,
    "grade": "Excellent|Good|Average|Below Average|Poor",
    "suggestions": [
      {
        "missing_skill": "...",
        "suggested_action": "...",
        "priority": "high|medium|low"
      }
    ],
    "analyzed_at": "ISO8601"
  },
  "analyzed_at": "ISO8601"
}
```


## How It Works
- PDFExtractor: Reads PDF bytes and extracts text with pdfplumber
- TextCleaner: Normalizes whitespace, removes artifacts, preserves structure
- AIService:
  - extract_candidate_data: LLM prompt to parse candidate details and return exactly 8 key skills if available
  - analyze_resume_comprehensive: LLM prompt with a rubric, returns summary, score, grade, and suggestions; result normalized in code

## Environment Variables
- GOOGLE_API_KEY — Required. Put in .env at the repository root.

Example .env:
GOOGLE_API_KEY=your_api_key_here

## Running with Python directly
From api/:
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

Or run main as a script:
python main.py

## Logging
- API logs to ats_system.log and stdout with INFO level by default.

## Notes and Limitations
- Only PDF resumes are accepted by the API.
- LLM outputs are parsed as JSON; defensive parsing and normalization applied.
- Ensure your API key has access to the specified Gemini models.
