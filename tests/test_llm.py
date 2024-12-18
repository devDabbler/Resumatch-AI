import pytest
from utils.llm import LLMAnalyzer
from utils.schemas import AnalysisResult, MatchStrength
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture
def llm_analyzer():
    return LLMAnalyzer()

def test_llm_initialization(llm_analyzer):
    """Test that LLM analyzer initializes correctly with API keys"""
    assert llm_analyzer.groq_api_key is not None
    assert llm_analyzer.gemini_api_key is not None
    assert llm_analyzer.groq_client is not None
    assert llm_analyzer.gemini_client is not None

def test_analyze_resume(llm_analyzer):
    """Test resume analysis with a sample resume"""
    sample_resume = """
    Software Engineer with 5 years of experience
    Skills: Python, JavaScript, React, Node.js
    Experience:
    - Senior Software Engineer at Tech Corp (2020-Present)
    - Software Developer at StartUp Inc (2018-2020)
    Education:
    - BS in Computer Science
    """
    
    role = "Software Engineer"
    required_skills = ["Python", "JavaScript", "React"]
    
    result = llm_analyzer.analyze(sample_resume, role, required_skills)
    
    # Verify result structure
    assert isinstance(result, dict)
    assert 'score' in result
    assert 'skill_score' in result
    assert 'experience_score' in result
    assert 'location_score' in result
    assert 'recommendation' in result
    assert 'analysis' in result
    assert isinstance(result['analysis'], list)  # Ensure analysis is a list
    
    # Verify score ranges
    assert 0 <= result['score'] <= 100
    assert 0 <= result['skill_score'] <= 100
    assert 0 <= result['experience_score'] <= 100
    assert 0 <= result['location_score'] <= 100
    
    # Verify recommendation is valid
    assert result['recommendation'] in [e.value for e in MatchStrength]

def test_llm_response_format(llm_analyzer):
    """Test LLM response formatting"""
    messages = [
        {"role": "system", "content": "You are a technical recruiter."},
        {"role": "user", "content": "Analyze this resume."}
    ]
    
    response = llm_analyzer.execute_request(messages)
    assert response is not None
    assert hasattr(response, 'choices')
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], 'message')
    assert 'content' in response.choices[0].message

def test_error_handling(llm_analyzer):
    """Test error handling with invalid input"""
    # Test with empty resume
    result = llm_analyzer.analyze("", "Software Engineer", ["Python"])
    assert result['score'] == 0
    assert result['recommendation'] == "NO_MATCH"
    assert isinstance(result['analysis'], list)
    
    # Test with invalid role
    result = llm_analyzer.analyze("Sample resume", "", ["Python"])
    assert result['score'] == 0
    assert result['recommendation'] == "NO_MATCH"
    assert isinstance(result['analysis'], list)
    
    # Test with empty skills
    result = llm_analyzer.analyze("Sample resume", "Software Engineer", [])
    assert result['score'] == 0
    assert result['recommendation'] == "NO_MATCH"
    assert isinstance(result['analysis'], list)

if __name__ == "__main__":
    pytest.main([__file__])
