import os
import json
import yaml
import sys
import logging
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.matcher import JobMatcher
from utils.llm import LLMAnalyzer
from utils.pdf import PDFProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeScoringCalibrator:
    def __init__(self, config_path='config/jobs.yaml'):
        """Initialize calibrator with job configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.job_matcher = JobMatcher(config_path)
        self.llm_analyzer = LLMAnalyzer()
        self.pdf_processor = PDFProcessor()

    def analyze_resume(self, pdf_path: str, role: str, expected_json_path: str) -> Dict[str, Any]:
        """Analyze a resume and compare with expected results."""
        # Load expected results
        with open(expected_json_path, 'r') as f:
            expected_results = json.load(f)

        # Extract text from PDF
        with open(pdf_path, 'rb') as f:
            resume_text = self.pdf_processor.extract_text(f)

        # Match skills
        skill_matches = self.job_matcher.match_skills(resume_text, role)

        # Extract experience
        experience_matches = self.job_matcher.extract_experience(resume_text)

        # Get role configuration
        role_config = next(
            (config for name, config in self.config['job_roles'].items() 
             if name.lower() == role.lower()), 
            None
        )

        if not role_config:
            raise ValueError(f"Role {role} not found in configuration")

        # Analyze resume
        analysis_result = self.llm_analyzer.analyze_resume(
            resume_text, 
            role, 
            skill_matches, 
            experience_matches.get('matches', [])
        )

        return {
            'predicted_score': analysis_result.technical_match_score,
            'expected_score': expected_results['expected_score'],
            'score_difference': abs(analysis_result.technical_match_score - expected_results['expected_score']),
            'recommendation': analysis_result.recommendation.value,
            'skills_matched': len(skill_matches.get('required', [])),
            'skills_total_required': len(role_config.get('required_skills', [])),
            'skills_matched_percentage': len(skill_matches.get('required', [])) / len(role_config.get('required_skills', [])) * 100
        }

def test_data_scientist_scoring():
    """Test scoring for Data Scientist role."""
    calibrator = ResumeScoringCalibrator()
    
    # Define test categories and their acceptable score differences
    categories = {
        'strong_match': {'max_diff': 5, 'min_score': 90, 'max_score': 100},
        'good_match': {'max_diff': 7, 'min_score': 80, 'max_score': 89},
        'partial_match': {'max_diff': 10, 'min_score': 70, 'max_score': 79},
        'no_match': {'max_diff': 15, 'min_score': 0, 'max_score': 69}
    }
    
    results = {}
    
    # Iterate through test data
    base_path = 'tests/test_data/data_scientist'
    for category in categories.keys():
        category_path = os.path.join(base_path, category)
        
        # Skip if category directory doesn't exist
        if not os.path.exists(category_path):
            logger.warning(f"Category {category} directory not found")
            continue
        
        category_results = []
        
        # Process each resume in the category
        for filename in os.listdir(category_path):
            if filename.endswith('.pdf'):
                base_name = filename.replace('.pdf', '')
                pdf_path = os.path.join(category_path, filename)
                json_path = os.path.join(category_path, f'{base_name}.json')
                
                if os.path.exists(json_path):
                    try:
                        result = calibrator.analyze_resume(pdf_path, 'Data Scientist', json_path)
                        category_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
        
        # Validate category results
        if category_results:
            category_scores = [r['predicted_score'] for r in category_results]
            category_diffs = [r['score_difference'] for r in category_results]
            
            results[category] = {
                'results': category_results,
                'avg_score': sum(category_scores) / len(category_scores),
                'avg_diff': sum(category_diffs) / len(category_diffs),
                'max_score': max(category_scores),
                'min_score': min(category_scores)
            }
            
            # Validate score ranges and differences
            assert results[category]['avg_diff'] <= categories[category]['max_diff'], \
                f"{category} category: Average score difference too high"
            
            assert all(
                categories[category]['min_score'] <= score <= categories[category]['max_score'] 
                for score in category_scores
            ), f"{category} category: Scores outside expected range"
    
    # Print detailed results
    for category, data in results.items():
        logger.info(f"\n{category.upper()} Category:")
        logger.info(f"Average Score: {data['avg_score']:.2f}")
        logger.info(f"Average Difference: {data['avg_diff']:.2f}")
        logger.info(f"Min Score: {data['min_score']}")
        logger.info(f"Max Score: {data['max_score']}")
        
        for result in data['results']:
            logger.info(
                f"Resume: Predicted {result['predicted_score']}, "
                f"Expected {result['expected_score']}, "
                f"Difference {result['score_difference']:.2f}"
            )
    
    return results

if __name__ == '__main__':
    test_data_scientist_scoring()
