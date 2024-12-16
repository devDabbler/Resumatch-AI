import unittest
import json
from pathlib import Path
from utils.llm import LLMAnalyzer
from utils.pdf import PDFProcessor

class TestYearExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.llm_analyzer = LLMAnalyzer()
        cls.pdf_processor = PDFProcessor()
        
        # Load test resumes
        cls.mixed_exp_resume = cls._load_resume(Path('Jubin_Soni_Resume_SDE (1).pdf'))
        cls.us_exp_resume = cls._load_resume(Path('Anne Castrillon_SDE.pdf'))
        
    @classmethod
    def _load_resume(cls, pdf_path):
        """Load resume text from PDF file."""
        with open(pdf_path, 'rb') as pdf_file:
            return cls.pdf_processor.extract_text(pdf_file)
    
    def test_mixed_experience_extraction(self):
        """Test extraction of mixed US and non-US work experience."""
        # Test with Jubin's resume which has both US and non-US experience
        analysis = self.llm_analyzer._gemini_experience_analysis(
            self.mixed_exp_resume, 
            "software_development_engineer"
        )
        
        # Validate experience summary exists
        self.assertIn('experience_summary', analysis)
        exp_summary = analysis['experience_summary']
        
        # Known experience values for Jubin's resume
        expected_total = 11.2
        expected_us = 7.4
        expected_non_us = 3.8
        
        # Allow for some variation in calculation (±0.5 years)
        margin = 0.5
        
        # Validate total years
        total_years = exp_summary.get('total_professional_years', 0)
        self.assertAlmostEqual(
            total_years, 
            expected_total, 
            delta=margin,
            msg=f"Total years {total_years} not close to expected {expected_total}"
        )
        
        # Validate US experience
        us_years = exp_summary.get('us_experience_years', 0)
        self.assertAlmostEqual(
            us_years, 
            expected_us, 
            delta=margin,
            msg=f"US years {us_years} not close to expected {expected_us}"
        )
        
        # Validate non-US experience
        non_us_years = exp_summary.get('non_us_experience_years', 0)
        self.assertAlmostEqual(
            non_us_years, 
            expected_non_us, 
            delta=margin,
            msg=f"Non-US years {non_us_years} not close to expected {expected_non_us}"
        )
        
    def test_us_only_experience_extraction(self):
        """Test extraction of US-only work experience."""
        # Test with Anne's resume which has only US experience
        analysis = self.llm_analyzer._gemini_experience_analysis(
            self.us_exp_resume, 
            "software_development_engineer"
        )
        
        # Validate experience summary exists
        self.assertIn('experience_summary', analysis)
        exp_summary = analysis['experience_summary']
        
        # Known experience values for Anne's resume
        expected_total = 6.0
        expected_us = 6.0
        expected_non_us = 0.0
        
        # Allow for some variation in calculation (±0.5 years)
        margin = 0.5
        
        # Validate total years
        total_years = exp_summary.get('total_professional_years', 0)
        self.assertAlmostEqual(
            total_years, 
            expected_total, 
            delta=margin,
            msg=f"Total years {total_years} not close to expected {expected_total}"
        )
        
        # Validate US experience
        us_years = exp_summary.get('us_experience_years', 0)
        self.assertAlmostEqual(
            us_years, 
            expected_us, 
            delta=margin,
            msg=f"US years {us_years} not close to expected {expected_us}"
        )
        
        # Validate non-US experience
        non_us_years = exp_summary.get('non_us_experience_years', 0)
        self.assertAlmostEqual(
            non_us_years, 
            expected_non_us, 
            delta=margin,
            msg=f"Non-US years {non_us_years} not close to expected {expected_non_us}"
        )
        
    def test_experience_details(self):
        """Test extraction of detailed position information."""
        # Test with mixed experience resume
        analysis = self.llm_analyzer._gemini_experience_analysis(
            self.mixed_exp_resume, 
            "software_development_engineer"
        )
        
        # Validate position details exist
        self.assertIn('position_details', analysis)
        positions = analysis['position_details']
        
        # Validate position structure
        for position in positions:
            self.assertIn('title', position)
            self.assertIn('company', position)
            self.assertIn('location', position)
            self.assertIn('duration', position)
            self.assertIn('is_us_based', position)
            
    def test_experience_validation(self):
        """Test validation of experience calculations."""
        # Test with both resumes
        for resume_text in [self.mixed_exp_resume, self.us_exp_resume]:
            analysis = self.llm_analyzer._gemini_experience_analysis(
                resume_text, 
                "software_development_engineer"
            )
            exp_summary = analysis['experience_summary']
            
            # Validate total years is sum of US and non-US years
            total_years = exp_summary.get('total_professional_years', 0)
            us_years = exp_summary.get('us_experience_years', 0)
            non_us_years = exp_summary.get('non_us_experience_years', 0)
            
            # Allow for small floating point differences
            self.assertAlmostEqual(
                total_years, 
                us_years + non_us_years, 
                places=1,
                msg="Total years does not match sum of US and non-US years"
            )

if __name__ == '__main__':
    unittest.main()
