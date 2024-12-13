from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Union
from enum import Enum
from datetime import datetime

class MatchStrength(str, Enum):
    STRONG = "STRONG_MATCH"
    GOOD = "GOOD_MATCH"
    POTENTIAL = "POTENTIAL_MATCH"
    NO_MATCH = "NO_MATCH"

    @classmethod
    def from_score(cls, score: int) -> 'MatchStrength':
        """Enhanced score to strength mapping with configurable thresholds"""
        if score >= 85:
            return cls.STRONG
        elif score >= 70:
            return cls.GOOD
        elif score >= 50:
            return cls.POTENTIAL
        else:
            return cls.NO_MATCH

class SkillAssessment(BaseModel):
    skill: str
    proficiency: str
    years: float
    context: Optional[str] = None
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)  # Add confidence score
    last_used: Optional[str] = None  # Add recency information
    
    @field_validator('years')
    @classmethod
    def validate_years(cls, v):
        return round(float(v), 1) if v else 0.0
    
    @field_validator('proficiency')
    @classmethod
    def validate_proficiency(cls, v):
        valid_levels = ['Expert', 'Advanced', 'Intermediate', 'Beginner']
        return v if v in valid_levels else 'Intermediate'

class InterviewQuestion(BaseModel):
    """Model for structured interview questions"""
    category: str = Field(..., description="Category of the interview question")
    question: str = Field(..., description="Full text of the interview question")
    context: Optional[str] = Field(default=None, description="Context from which the question was derived")

class ExperienceDetails(BaseModel):
    us_experience_years: float = Field(default=0.0)
    non_us_experience_years: float = Field(default=0.0)
    total_professional_years: float = Field(default=0.0)
    internship_count: int = Field(default=0)
    experience_breakdown: List[str] = Field(default_factory=list)
    experience_strength: str = Field(default="UNKNOWN")
    experience_flags: List[str] = Field(default_factory=list)
    last_position_date: Optional[datetime] = None  # Add last position date
    employment_gaps: List[Dict[str, datetime]] = Field(default_factory=list)  # Track gaps

class AnalysisResult(BaseModel):
    technical_match_score: int = Field(..., ge=0, le=100)
    recommendation: MatchStrength
    skills_assessment: List[SkillAssessment] = Field(default_factory=list)
    technical_gaps: List[str] = Field(default_factory=list)
    interview_questions: List[Union[str, InterviewQuestion]] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    experience_details: Optional[ExperienceDetails] = None
    analysis_timestamp: datetime = Field(default_factory=datetime.now)  # Add timestamp
    confidence_score: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)  # Add confidence

    @field_validator('technical_match_score')
    @classmethod
    def validate_score(cls, v):
        return max(0, min(100, v))

    @field_validator('recommendation')
    @classmethod
    def validate_recommendation(cls, v):
        if isinstance(v, str):
            return MatchStrength(v)
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
