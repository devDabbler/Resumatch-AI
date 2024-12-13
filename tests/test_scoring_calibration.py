import pytest
import os
import json
from utils.llm import LLMAnalyzer
from utils.matcher import JobMatcher
from utils.pdf import PDFProcessor
import logging
from typing import Dict, List, Tuple
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class TestScoringCalibration:
    @pytest.fixture
    def matcher(self):
        return JobMatcher('config/jobs.yaml')

    @pytest.fixture
    def llm_analyzer(self):
        return LLMAnalyzer()

    def load_test_resume(self, category: str, filename: str) -> dict:
        path = f'tests/test_data/data_scientist/{category}/{filename}'
        with open(path, 'r') as f:
            return json.load(f)

    def test_score_distribution(self, matcher):
        """Test score distribution across different resume types"""
        test_cases = [
            {
                'case': "Strong match",
                'required': ['Python', 'Machine Learning', 'Statistics', 'Data Visualization', 'Git', 'Linear Algebra'],
                'preferred': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy'],  # Missing one preferred
                'experience': 8,
                'expected_range': (80, 95)
            },
            {
                'case': "Good match",
                'required': ['Python', 'Machine Learning', 'Statistics', 'Data Visualization', 'Git'],  # Missing one required
                'preferred': ['TensorFlow', 'PyTorch', 'Scikit-learn'],  # Fewer preferred
                'experience': 5,
                'expected_range': (65, 85)
            },
            {
                'case': "Partial match",
                'required': ['Python', 'Machine Learning', 'Statistics', 'Git'],  # Missing two required
                'preferred': ['TensorFlow', 'PyTorch'],  # Minimal preferred
                'experience': 3,
                'expected_range': (50, 70)
            }
        ]
        
        for case in test_cases:
            matched_skills = {
                'required': case['required'],
                'preferred': case['preferred'],
                'experience_matches': {'years': case['experience']}
            }
            
            score = matcher.calculate_match_score(
                role_name="Data Scientist, C3",
                matched_skills=matched_skills,
                experience_years=case['experience']
            )
            
            print(f"\nTest Case: {case['case']}")
            print(f"Required Skills: {len(case['required'])}/{6}")
            print(f"Preferred Skills: {len(case['preferred'])}/{13}")
            print(f"Experience: {case['experience']} years")
            print(f"Expected Range: {case['expected_range']}")
            print(f"Actual Score: {score['technical_match_score']}")
            
            assert score['technical_match_score'] in range(*case['expected_range']), \
                f"Score {score['technical_match_score']} not in expected range {case['expected_range']} for {case['case']}"