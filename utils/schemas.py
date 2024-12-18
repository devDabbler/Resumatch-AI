from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class InterviewQuestion(BaseModel):
    category: str = Field(..., description="Category of the question")
    question: str = Field(..., description="Detailed multi-part question")
    context: str = Field(..., description="Relevant context from resume")

class AnalysisResult(BaseModel):
    technical_match_score: int = Field(..., ge=0, le=100)
    recommendation: str = Field(..., pattern="^(STRONG_MATCH|GOOD_MATCH|POTENTIAL_MATCH|NO_MATCH)$")
    interview_questions: List[InterviewQuestion]
    technical_gaps: List[str]
    key_findings: List[str]
    concerns: List[str]
    confidence_score: float = Field(..., ge=0, le=1)

class Experience(BaseModel):
    duration: float
    type: str  # professional, internship, academic
    location: str  # US, non-US
    description: str
    skills: List[str]

class ResumeData(BaseModel):
    raw_text: str
    experiences: List[Experience]
    skills: Dict[str, List[str]]  # skill -> contexts
    linkedin_profile: Optional[str]
    education: List[str]
