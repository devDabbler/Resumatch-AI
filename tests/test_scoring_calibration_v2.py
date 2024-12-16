import unittest
import json
from pathlib import Path
from utils.llm import LLMAnalyzer
from utils.matcher import JobMatcher
from utils.pdf import PDFProcessor

class TestScoringCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.llm_analyzer = LLMAnalyzer()
        cls.job_matcher = JobMatcher('config/jobs.yaml')
        cls.pdf_processor = PDFProcessor()
        
        # Define score ranges and categories
        cls.score_ranges = {
            'strong_match': (85, 100),
            'good_match': (70, 84),
            'potential_match': (50, 69),
            'no_match': (0, 49)
        }
        
        # Base paths for test data
        cls.test_data_path = Path('tests/test_data')
        cls.roles = ['data_scientist', 'software_development_engineer']
        cls.categories = ['strong_match', 'good_match', 'partial_match', 'no_match']
        
    def get_expected_score(self, json_path):
        """Get expected score and category from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
            return {
                'score': data.get('expected_score', 0),
                'category': data.get('expected_category', 'no_match'),
                'skills': data.get('expected_skills', []),
                'experience': data.get('expected_experience', {})
            }
            
    def analyze_resume(self, pdf_path, role_name):
        """Analyze a resume and return the analysis results."""
        # Extract text from PDF
        with open(pdf_path, 'rb') as pdf_file:
            resume_text = self.pdf_processor.extract_text(pdf_file)
            
        # Get skill matches
        skill_matches = self.job_matcher.match_skills(resume_text, role_name)
        
        # Analyze resume
        return self.llm_analyzer.analyze_resume(resume_text, role_name, skill_matches)
        
    def test_all_roles_scoring(self):
        """Test scoring calibration for all roles and categories."""
        for role in self.roles:
            role_path = self.test_data_path / role
            
            for category in self.categories:
                category_path = role_path / category
                
                # Process each resume in the category
                for resume_num in [1, 2]:
                    pdf_file = category_path / f'resume{resume_num}.pdf'
                    json_file = category_path / f'resume{resume_num}.json'
                    
                    if not pdf_file.exists() or not json_file.exists():
                        continue
                        
                    # Get expected results
                    expected = self.get_expected_score(json_file)
                    
                    # Analyze resume
                    analysis = self.analyze_resume(pdf_file, role)
                    actual_score = analysis.get('technical_match_score', 0)
                    
                    # Test name for better error reporting
                    test_name = f"{role}/{category}/resume{resume_num}"
                    
                    # Validate score range
                    min_score, max_score = self.score_ranges[category]
                    self.assertTrue(
                        min_score <= actual_score <= max_score,
                        f"{test_name}: Score {actual_score} outside range ({min_score}-{max_score})"
                    )
                    
                    # Validate skills match if expected skills are provided
                    if expected['skills']:
                        matched_skills = {
                            skill['skill'] 
                            for skill in analysis.get('skills_assessment', [])
                        }
                        expected_skills = set(expected['skills'])
                        missing_skills = expected_skills - matched_skills
                        
                        self.assertTrue(
                            len(missing_skills) == 0,
                            f"{test_name}: Missing expected skills: {missing_skills}"
                        )
                        
    def test_experience_impact(self):
        """Test that experience properly impacts scoring."""
        # Test strong matches with different experience levels
        role = 'software_development_engineer'
        category = 'strong_match'
        
        # Compare scores of resume1 and resume2
        pdf1 = self.test_data_path / role / category / 'resume1.pdf'
        pdf2 = self.test_data_path / role / category / 'resume2.pdf'
        
        analysis1 = self.analyze_resume(pdf1, role)
        analysis2 = self.analyze_resume(pdf2, role)
        
        # Get experience years
        exp1 = analysis1.get('experience_details', {}).get('experience_summary', {}).get('total_professional_years', 0)
        exp2 = analysis2.get('experience_details', {}).get('experience_summary', {}).get('total_professional_years', 0)
        
        # Compare scores if there's a significant experience difference
        if abs(exp1 - exp2) >= 2:
            score1 = analysis1.get('technical_match_score', 0)
            score2 = analysis2.get('technical_match_score', 0)
            
            # More experienced resume should have higher score
            if exp1 > exp2:
                self.assertGreaterEqual(score1, score2)
            else:
                self.assertGreaterEqual(score2, score1)

if __name__ == '__main__':
    unittest.main()
