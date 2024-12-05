import unittest
import json
from pathlib import Path
from app import load_config
from utils.matcher import JobMatcher
from utils.pdf import PDFProcessor
from utils.llm import LLMAnalyzer

class TestScoringCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.config = load_config()
        cls.job_matcher = JobMatcher('config/jobs.yaml')
        cls.pdf_processor = PDFProcessor()
        cls.llm_analyzer = LLMAnalyzer()
        
        # Define expected score ranges for each category
        cls.score_ranges = {
            'strong_match': (85, 100),
            'good_match': (70, 84),
            'potential_match': (50, 69),
            'no_match': (0, 49)
        }
        
        # Base path for test data
        cls.test_data_path = Path('tests/test_data/data_scientist')
        
    def validate_score_range(self, score, category):
        """Validate if a score falls within the expected range."""
        min_score, max_score = self.score_ranges[category]
        msg = (
            f"Score {score} for {category} outside range "
            f"({min_score}-{max_score})"
        )
        self.assertTrue(min_score <= score <= max_score, msg)
        
    def load_json_score(self, json_path):
        """Load pre-configured score from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data.get('expected_score', 0)
            
    def process_resume(self, resume_file, role_name):
        """Process a resume file and return score data."""
        # Extract text from PDF
        with open(resume_file, 'rb') as pdf_file:
            self.pdf_processor.extract_text(pdf_file)
            
        # Extract skills and experience from JSON
        json_file = resume_file.with_suffix('.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
            experience_years = data.get('years_experience', 0)
            expected_skills = data.get('expected_skills', {})
            matched_skills = {
                'required': expected_skills.get('required_skills', []),
                'preferred': expected_skills.get('preferred_skills', [])
            }
            
        # Get score details
        score_details = self.job_matcher.calculate_match_score(
            role_name=role_name,
            matched_skills=matched_skills,
            experience_years=experience_years
        )
        
        # Validate score structure
        self.assertIsInstance(score_details, dict)
        self.assertIn('technical_match_score', score_details)
        self.assertIn('skills_score', score_details)
        self.assertIn('experience_score', score_details)
        self.assertIn('skills_breakdown', score_details)
        
        # Validate score ranges
        self.assertGreaterEqual(score_details['technical_match_score'], 0)
        self.assertLessEqual(score_details['technical_match_score'], 100)
        self.assertGreaterEqual(score_details['skills_score'], 0)
        self.assertLessEqual(score_details['skills_score'], 100)
        self.assertGreaterEqual(score_details['experience_score'], 0)
        self.assertLessEqual(score_details['experience_score'], 100)
        
        # Return technical match score for existing test logic
        return score_details['technical_match_score']
            
    def test_strong_match_resumes(self):
        """Test scoring for strong match resumes."""
        roles = ['Data Scientist', 'Software Development Engineer']
        
        for role in roles:
            strong_match_dir = self.test_data_path / role.lower().replace(' ', '_') / 'strong_match'
            for resume_file in strong_match_dir.glob('*.pdf'):
                json_file = resume_file.with_suffix('.json')
                actual_score = self.process_resume(resume_file, role)
                expected_score = self.load_json_score(json_file)
                
                # Validate score is in strong match range
                self.validate_score_range(actual_score, 'strong_match')
                
                # Score should be within 5 points of expected
                self.assertAlmostEqual(
                    actual_score, 
                    expected_score, 
                    delta=5,
                    msg=f"Significant score deviation for {resume_file.name} ({role})"
                )
            
    def test_good_match_resumes(self):
        """Test scoring for good match resumes."""
        roles = ['Data Scientist', 'Software Development Engineer']
        
        for role in roles:
            good_match_dir = self.test_data_path / role.lower().replace(' ', '_') / 'good_match'
            for resume_file in good_match_dir.glob('*.pdf'):
                json_file = resume_file.with_suffix('.json')
                actual_score = self.process_resume(resume_file, role)
                expected_score = self.load_json_score(json_file)
                
                self.validate_score_range(actual_score, 'good_match')
                self.assertAlmostEqual(
                    actual_score, 
                    expected_score, 
                    delta=5,
                    msg=f"Significant score deviation for {resume_file.name} ({role})"
                )
            
    def test_potential_match_resumes(self):
        """Test scoring for potential match resumes."""
        roles = ['Data Scientist', 'Software Development Engineer']
        
        for role in roles:
            potential_match_dir = self.test_data_path / role.lower().replace(' ', '_') / 'potential_match'
            for resume_file in potential_match_dir.glob('*.pdf'):
                json_file = resume_file.with_suffix('.json')
                actual_score = self.process_resume(resume_file, role)
                expected_score = self.load_json_score(json_file)
                
                self.validate_score_range(actual_score, 'potential_match')
                self.assertAlmostEqual(
                    actual_score, 
                    expected_score, 
                    delta=5,
                    msg=f"Significant score deviation for {resume_file.name} ({role})"
                )
            
    def test_no_match_resumes(self):
        """Test scoring for no match resumes."""
        roles = ['Data Scientist', 'Software Development Engineer']
        
        for role in roles:
            no_match_dir = self.test_data_path / role.lower().replace(' ', '_') / 'no_match'
            for resume_file in no_match_dir.glob('*.pdf'):
                json_file = resume_file.with_suffix('.json')
                actual_score = self.process_resume(resume_file, role)
                expected_score = self.load_json_score(json_file)
                
                self.validate_score_range(actual_score, 'no_match')
                self.assertAlmostEqual(
                    actual_score, 
                    expected_score, 
                    delta=5,
                    msg=f"Significant score deviation for {resume_file.name} ({role})"
                )
            
    def test_score_distribution(self):
        """Test that scores maintain proper distribution across categories."""
        roles = ['Data Scientist', 'Software Development Engineer']
        categories = ['strong_match', 'good_match', 'potential_match', 'no_match']
        
        for role in roles:
            all_scores = []
            
            for category in categories:
                category_dir = self.test_data_path / role.lower().replace(' ', '_') / category
                for resume_file in category_dir.glob('*.pdf'):
                    score = self.process_resume(resume_file, role)
                    all_scores.append((category, score))
            
            # Verify category separation
            for i in range(len(categories) - 1):
                current_cat = categories[i]
                next_cat = categories[i + 1]
                
                current_scores = [s for cat, s in all_scores if cat == current_cat]
                next_scores = [s for cat, s in all_scores if cat == next_cat]
                
                if current_scores and next_scores:
                    min_current = min(current_scores)
                    max_next = max(next_scores)
                    self.assertGreater(
                        min_current,
                        max_next,
                        f"Score overlap between {current_cat} and {next_cat} for {role}"
                    )

    def test_required_skills_threshold(self):
        """Test that required skills threshold is enforced."""
        # Test with insufficient required skills
        test_data = {
            'experience_years': 5,
            'matched_skills': {
                'required': ['Python'],  # Only 1 of 6 required skills
                'preferred': ['TensorFlow', 'PyTorch', 'Docker']
            }
        }
        
        score_details = self.job_matcher.calculate_match_score(
            role_name="Data Scientist",
            matched_skills=test_data['matched_skills'],
            experience_years=test_data['experience_years']
        )
        
        # Should fail threshold and return 0
        self.assertEqual(score_details['technical_match_score'], 0)
        self.assertEqual(score_details['analysis'], 'Failed required skills threshold')

    def test_experience_scoring(self):
        """Test experience score calculation."""
        test_cases = [
            # Below minimum years
            {
                'years': 1,
                'expected_ratio': 0.25  
            },
            # At minimum years
            {
                'years': 2,
                'expected_ratio': 1.0
            },
            # Above minimum, below maximum
            {
                'years': 5,
                'expected_ratio': 1.0
            },
            # Above maximum years
            {
                'years': 10,
                'expected_ratio': 1.0
            }
        ]
        
        for case in test_cases:
            score_details = self.job_matcher.calculate_match_score(
                role_name="Data Scientist",
                matched_skills={
                    'required': ['Python', 'Machine Learning', 'Statistics'],
                    'preferred': ['TensorFlow']
                },
                experience_years=case['years']
            )
            
            expected_score = int(case['expected_ratio'] * 100)
            self.assertEqual(
                score_details['experience_score'],
                expected_score,
                f"Experience score incorrect for {case['years']} years"
            )

    def test_skills_weighting(self):
        """Test proper weighting between required and preferred skills."""
        test_data = {
            'experience_years': 5,
            'matched_skills': {
                'required': [
                    'Python', 'Machine Learning', 'Statistics',
                    'Data Visualization', 'Deep Learning', 'Big Data'
                ],
                'preferred': [
                    'TensorFlow', 'PyTorch', 'Docker', 'MLflow',
                    'Databricks', 'Airflow'
                ]
            }
        }
        
        score_details = self.job_matcher.calculate_match_score(
            role_name="Data Scientist",
            matched_skills=test_data['matched_skills'],
            experience_years=test_data['experience_years']
        )
        
        # Get weights from config
        weights = self.job_matcher.scoring_config['skill_weights']
        required_weight = weights['required']
        preferred_weight = weights['preferred']
        
        # Calculate expected scores
        required_ratio = len(test_data['matched_skills']['required']) / 6  # Total required
        preferred_ratio = len(test_data['matched_skills']['preferred']) / 12  # Total preferred
        
        expected_skills_score = int(
            (required_ratio * required_weight + 
             preferred_ratio * preferred_weight) * 100
        )
        
        self.assertAlmostEqual(
            score_details['skills_score'],
            expected_skills_score,
            delta=5,
            msg="Skills weighting incorrect"
        )

    def test_score_components(self):
        """Test that final score properly combines skills and experience."""
        test_data = {
            'experience_years': 5,
            'matched_skills': {
                'required': ['Python', 'Machine Learning', 'Statistics'],
                'preferred': ['TensorFlow', 'PyTorch']
            }
        }
        
        score_details = self.job_matcher.calculate_match_score(
            role_name="Data Scientist",
            matched_skills=test_data['matched_skills'],
            experience_years=test_data['experience_years']
        )
        
        # Get component scores
        skills_score = score_details['skills_score']
        experience_score = score_details['experience_score']
        technical_score = score_details['technical_match_score']
        
        # Get weights from config
        weights = self.job_matcher.scoring_config['weights']['technical']
        
        # Calculate expected final score
        expected_score = int(
            skills_score * weights['skills'] +
            experience_score * weights['experience']
        )
        
        self.assertAlmostEqual(
            technical_score,
            expected_score,
            delta=5,
            msg="Final score calculation incorrect"
        )

    def test_edge_cases(self):
        """Test scoring behavior for edge cases."""
        edge_cases = [
            # No experience, no skills
            {
                'case': 'empty',
                'years': 0,
                'skills': {'required': [], 'preferred': []},
                'expected_score': 0
            },
            # No experience but all skills
            {
                'case': 'skills_no_exp',
                'years': 0,
                'skills': {
                    'required': [
                        'Python', 'Machine Learning', 'Statistics',
                        'Data Visualization', 'Deep Learning', 'Big Data'
                    ],
                    'preferred': [
                        'TensorFlow', 'PyTorch', 'Docker', 'MLflow',
                        'Databricks', 'Airflow'
                    ]
                },
                'expected_score': 46  # 50% reduction for no experience
            },
            # Max experience but no skills
            {
                'case': 'exp_no_skills',
                'years': 20,
                'skills': {'required': [], 'preferred': []},
                'expected_score': 0  # Should fail required skills threshold
            }
        ]
        
        for case in edge_cases:
            score_details = self.job_matcher.calculate_match_score(
                role_name="Data Scientist",
                matched_skills=case['skills'],
                experience_years=case['years']
            )
            
            if case['case'] == 'skills_no_exp':
                self.assertEqual(
                    score_details['technical_match_score'],
                    case['expected_score'],
                    f"Edge case failed: {case['case']}"
                )
            else:
                self.assertEqual(
                    score_details['technical_match_score'],
                    case['expected_score'],
                    f"Edge case failed: {case['case']}"
                )

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        error_cases = [
            # Invalid role name
            {
                'case': 'invalid_role',
                'role': 'NonexistentRole',
                'years': 5,
                'skills': {'required': ['Python'], 'preferred': []}
            },
            # Negative experience years
            {
                'case': 'negative_years',
                'role': 'Data Scientist',
                'years': -1,
                'skills': {'required': ['Python'], 'preferred': []}
            },
            # Missing required skills key
            {
                'case': 'missing_required',
                'role': 'Data Scientist',
                'years': 5,
                'skills': {'preferred': []}
            },
            # None values
            {
                'case': 'none_values',
                'role': 'Data Scientist',
                'years': None,
                'skills': None
            }
        ]
        
        for case in error_cases:
            score_details = self.job_matcher.calculate_match_score(
                role_name=case['role'],
                matched_skills=case['skills'],
                experience_years=case['years']
            )
            
            # All error cases should return a valid score structure with 0 scores
            self.assertEqual(score_details['technical_match_score'], 0)
            self.assertIn(case['case'], score_details['analysis'])  # Check for case name in error message
            self.assertIn('Error', score_details['analysis'])  # Check for 'Error' keyword

    def test_score_normalization(self):
        """Test that scores are properly normalized and rounded."""
        test_data = {
            'experience_years': 3,
            'matched_skills': {
                'required': ['Python', 'Machine Learning', 'Statistics'],
                'preferred': ['TensorFlow', 'PyTorch']
            }
        }
        
        score_details = self.job_matcher.calculate_match_score(
            role_name="Data Scientist",
            matched_skills=test_data['matched_skills'],
            experience_years=test_data['experience_years']
        )
        
        # Check all scores are integers between 0 and 100
        scores_to_check = [
            'technical_match_score',
            'skills_score',
            'experience_score'
        ]
        
        for score_key in scores_to_check:
            score = score_details[score_key]
            self.assertIsInstance(score, int)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
            
        # Check ratios in skills_breakdown are floats between 0 and 1
        for ratio_key in ['required_score', 'preferred_score']:
            ratio = score_details['skills_breakdown'][ratio_key]
            self.assertIsInstance(ratio, float)
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLessEqual(ratio, 1.0)

    def test_skill_pattern_matching(self):
        """Test that skill pattern matching handles variations correctly."""
        test_patterns = [
            # Python variations
            {
                'skill': 'Python',
                'variations': [
                    'python development',
                    'python3',
                    'py3',
                    'python programming',
                    'python implementation'
                ],
                'non_matches': [
                    'typescript',
                    'javascript',
                    'pythonic thinking'  # Too vague
                ]
            },
            # Machine Learning variations
            {
                'skill': 'Machine Learning',
                'variations': [
                    'ML',
                    'machine learning models',
                    'deep learning',
                    'neural networks',
                    'ML development'
                ],
                'non_matches': [
                    'learning management',
                    'learned skills',
                    'machine operator'
                ]
            }
        ]
        
        for pattern in test_patterns:
            # Test positive matches
            for text in pattern['variations']:
                result = self.job_matcher.check_skill(pattern['skill'], text)
                self.assertTrue(
                    result['matched'],
                    f"Failed to match {pattern['skill']} in: {text}"
                )
                
            # Test negative matches
            for text in pattern['non_matches']:
                result = self.job_matcher.check_skill(pattern['skill'], text)
                self.assertFalse(
                    result['matched'],
                    f"Incorrectly matched {pattern['skill']} in: {text}"
                )

    def test_experience_extraction(self):
        """Test experience extraction from resume text."""
        test_cases = [
            {
                'text': '5+ years of Python development experience',
                'expected_years': 5
            },
            {
                'text': 'Over 7 years in machine learning and data science',
                'expected_years': 7
            },
            {
                'text': '''
                Software Engineer | ABC Corp | Jan 2018 - Present
                Data Scientist | XYZ Inc | Mar 2015 - Dec 2017
                ''',
                'expected_years': 9  # Calculated from date spans
            },
            {
                'text': 'Recent graduate with internship experience',
                'expected_years': 0  # No explicit years mentioned
            }
        ]
        
        for case in test_cases:
            result = self.job_matcher.extract_experience(case['text'])
            self.assertEqual(
                result['years'],
                case['expected_years'],
                f"Experience extraction failed for: {case['text'][:50]}..."
            )

    def test_skill_context_validation(self):
        """Test that skills are validated with proper context."""
        test_cases = [
            {
                'text': 'Developed machine learning models using Python and TensorFlow',
                'skill': 'Python',
                'expect_match': True,
                'expect_context': True
            },
            {
                'text': 'Python listed in skills section',
                'skill': 'Python',
                'expect_match': True,
                'expect_context': False  # No implementation context
            },
            {
                'text': 'Implemented deep learning solutions with PyTorch',
                'skill': 'Deep Learning',
                'expect_match': True,
                'expect_context': True
            }
        ]
        
        for case in test_cases:
            result = self.job_matcher.check_skill(case['skill'], case['text'])
            self.assertEqual(
                result['matched'],
                case['expect_match'],
                f"Skill match failed for: {case['skill']}"
            )
            has_context = bool(result.get('context', []))
            self.assertEqual(
                has_context,
                case['expect_context'],
                f"Context validation failed for: {case['skill']}"
            )

if __name__ == '__main__':
    unittest.main() 