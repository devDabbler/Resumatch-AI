import pytest
import yaml
from utils.llm import LLMAnalyzer
from utils.pdf import PDFProcessor
from utils.schemas import MatchStrength
import os
from pathlib import Path

@pytest.fixture
def llm_analyzer():
    return LLMAnalyzer()

@pytest.fixture
def pdf_processor():
    return PDFProcessor()

@pytest.fixture
def job_config():
    with open('config/jobs.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_analyze_strong_match_resume(llm_analyzer, pdf_processor, job_config):
    """Test analysis with a known strong match resume"""
    # Get resume text
    resume_path = 'tests/test_data/software_development_engineer/strong_match/strong_match_resume1.pdf'
    resume_text = pdf_processor.extract_text_from_path(resume_path)
    
    # Get job requirements
    role = "Software Development Engineer"
    role_config = job_config['job_roles'][role]
    required_skills = [
        skill['name'] if isinstance(skill, dict) else skill 
        for skill in role_config['required_skills']
    ]
    
    # Analyze resume
    result = llm_analyzer.analyze(resume_text, role, required_skills)
    
    # Verify result structure
    assert isinstance(result, dict)
    assert 'score' in result
    assert 'skill_score' in result
    assert 'experience_score' in result
    assert 'location_score' in result
    assert 'recommendation' in result
    assert 'analysis' in result
    assert isinstance(result['analysis'], list)
    
    # Verify scores are within valid ranges
    assert 0 <= result['score'] <= 100
    assert 0 <= result['skill_score'] <= 100
    assert 0 <= result['experience_score'] <= 100
    assert 0 <= result['location_score'] <= 100
    
    # Since this is a strong match resume, verify high scores
    assert result['score'] >= 85  # Strong match threshold
    assert result['recommendation'] == MatchStrength.STRONG_MATCH
    
    # Verify skills assessment
    assert 'skills_assessment' in result
    assert isinstance(result['skills_assessment'], list)
    if result['skills_assessment']:
        skill = result['skills_assessment'][0]
        assert 'skill' in skill
        assert 'proficiency' in skill
        assert 'years' in skill
        
    # Verify experience details
    assert 'experience_details' in result
    exp_details = result['experience_details']
    assert exp_details['total_professional_years'] > 0
    
    # Verify interview questions
    assert 'interview_questions' in result
    assert isinstance(result['interview_questions'], list)

def test_analyze_with_invalid_input(llm_analyzer):
    """Test error handling with invalid inputs"""
    # Test with empty resume
    result = llm_analyzer.analyze("", "Software Development Engineer", ["Python"])
    assert result['score'] == 0
    assert result['recommendation'] == MatchStrength.NO_MATCH
    assert isinstance(result['analysis'], list)
    
    # Test with invalid role
    result = llm_analyzer.analyze("Sample resume", "", ["Python"])
    assert result['score'] == 0
    assert result['recommendation'] == MatchStrength.NO_MATCH
    assert isinstance(result['analysis'], list)
    
    # Test with empty skills
    result = llm_analyzer.analyze("Sample resume", "Software Development Engineer", [])
    assert result['score'] == 0
    assert result['recommendation'] == MatchStrength.NO_MATCH
    assert isinstance(result['analysis'], list)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
