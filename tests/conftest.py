import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def test_data_paths():
    """Provide paths to test data directories"""
    base_path = Path(__file__).parent / "test_data"
    return {
        "data_scientist": {
            "strong_match": base_path / "data_scientist/strong_match",
            "good_match": base_path / "data_scientist/good_match",
            "partial_match": base_path / "data_scientist/partial_match",
            "no_match": base_path / "data_scientist/no_match"
        }
    }

@pytest.fixture
def mock_llm_responses():
    """Provide mock LLM responses for testing"""
    return {
        "no_match": {
            "technical_match_score": 25,
            "recommendation": "NO_MATCH",
            "skills_assessment": [],
            "technical_gaps": ["Missing core ML skills", "No data science experience"],
            "interview_questions": [],
            "key_findings": ["Software engineering background", "Limited DS exposure"],
            "concerns": ["No ML/Stats background", "Career transition needed"]
        }
    }
