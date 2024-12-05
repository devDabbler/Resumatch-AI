from utils.logging_config import setup_logging
import yaml
import regex as re
from typing import Dict, List, Any
import logging
from datetime import datetime

# Initialize logging configuration
setup_logging()
logger = logging.getLogger('matcher')

class JobMatcher:
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        logger.info(f"Initializing JobMatcher with config: {config_path}")
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Initialize patterns from config
            self.patterns = {
                'experience': self.config.get('analysis_patterns', {}).get('experience', []),
                'skills': self.config.get('analysis_patterns', {}).get('skills', {}),
                'context': self.config.get('analysis_patterns', {}).get('context', [])
            }
            
            # Initialize skill variations map
            self.skill_variations = self.config.get('skill_variations', {})
            
            # Initialize current role as None
            self.current_role = None
            
            self.scoring_config = self.config.get('scoring_config', {})

        except Exception as e:
            logger.error(f"Failed to initialize JobMatcher: {str(e)}")
            raise

    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience details with enhanced year detection."""
        try:
            # Look for explicit year mentions with improved patterns
            year_patterns = [
                r'(\d+)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',  # Matches "5+ years experience"
                r'(?:over|more than)\s+(\d+)\s*(?:years?|yrs?)',  # Matches "over 7 years"
                r'(\d+)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:work|professional)',  # Matches "5 years work"
                r'experience\s+(?:of|for|with)?\s*(\d+)\s*\+?\s*(?:years?|yrs?)',  # Additional patterns
                r'(\d+)\s*\+\s*(?:years?|yrs?)',  # Simple year mentions
                r'(?:^|\s)(\d+)\+' # Just numbers with plus
            ]
            
            # Look for date ranges
            date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\s*(?:-|to|–)\s*(?:Present|Current|Now|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4}))'
            
            max_years = 0
            text_lower = text.lower()
            
            # Check explicit year mentions first
            for pattern in year_patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    try:
                        years = int(match.group(1))
                        max_years = max(max_years, years)
                    except (IndexError, ValueError):
                        continue
            
            # If no explicit years found, check date ranges and sum up total experience
            if max_years == 0:
                current_year = datetime.now().year
                total_years = 0
                matches = re.finditer(date_pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        start_year = int(match.group(1))
                        end_year = int(match.group(2)) if match.group(2) else current_year
                        years = end_year - start_year + 1  # Add 1 to include both start and end years
                        total_years += years
                    except (IndexError, ValueError):
                        continue
                max_years = total_years
            
            return {
                'years': max_years,
                'matches': [],
                'all_years': [max_years] if max_years > 0 else [],
                'locations': {'us': [], 'india': [], 'other': []},
                'us_experience_ratio': 0,
                'india_experience_ratio': 0,
                'short_stints': [],
                'gaps': []
            }
            
        except Exception as e:
            logger.error(f"Error extracting experience: {str(e)}")
            return {
                'years': 0,
                'matches': [],
                'all_years': [],
                'locations': {'us': [], 'india': [], 'other': []},
                'us_experience_ratio': 0,
                'india_experience_ratio': 0,
                'short_stints': [],
                'gaps': []
            }

    def match_skills(self, text: str, role_name: str) -> Dict[str, Any]:
        """Match required and preferred skills with context analysis."""
        # Set current role for skill matching
        self.current_role = role_name
        
        role_config = self.config.get('job_roles', {}).get(role_name)
        if not role_config:
            logger.error(f"Role {role_name} not found in configuration")
            return {'required': [], 'preferred': [], 'context': {}}
        
        matches = {
            'required': [],
            'preferred': [],
            'context': {}
        }

        # Get role skills as lowercase for case-insensitive comparison
        required_skills = {skill.lower(): skill for skill in role_config.get('required_skills', [])}
        preferred_skills = {skill.lower(): skill for skill in role_config.get('preferred_skills', [])}

        logger.info(f"Checking required skills: {list(required_skills.values())}")
        logger.info(f"Checking preferred skills: {list(preferred_skills.values())}")

        # Process all potential skills in the text
        for skill in role_config.get('required_skills', []) + role_config.get('preferred_skills', []):
            logger.info(f"\nAnalyzing skill: {skill}")
            result = self.check_skill(skill, text)
            
            if result['matched']:
                skill_lower = skill.lower()
                if skill_lower in required_skills:
                    matches['required'].append(skill)
                    logger.info(f"[PASS] Found required skill: {skill}")
                elif skill_lower in preferred_skills:
                    matches['preferred'].append(skill)
                    logger.info(f"[PASS] Found preferred skill: {skill}")
                
                if result['context']:
                    matches['context'][skill] = result['context']
                    clean_context = ''.join(char for char in result['context'] if ord(char) < 128)
                    logger.info(f"  Context: {clean_context}")
            else:
                logger.info(f"[FAIL] Skill not found: {skill}")

        logger.info("\nMatched Skills Summary:")
        logger.info(f"Required: {matches['required']}")
        logger.info(f"Preferred: {matches['preferred']}")
        
        return matches

    def check_skill(self, skill: str, text: str) -> Dict[str, Any]:
        """Check for skill presence with context validation."""
        skill_lower = skill.lower()
        text_lower = text.lower()
        
        try:
            # Get variations for the skill from config
            variations = self.skill_variations.get(skill_lower.replace(' ', '_'), [])
            variations.extend([
                skill_lower,
                skill_lower.replace(' ', ''),
                skill_lower.replace(' ', '-'),
                skill_lower.replace(' ', '_')
            ])
            
            logger.info(f"  Checking variations: {variations}")
            
            # Consider skills section headers as valid context
            skills_headers = [
                'skills', 'technical skills', 'programming languages',
                'technologies', 'tech stack', 'languages', 'proficiencies',
                'technical proficiencies', 'core competencies', 'tools',
                'frameworks', 'platforms', 'expertise'  # Added more common headers
            ]
            
            # First check if we're in a skills section
            in_skills_section = any(
                re.search(rf'\b{re.escape(header)}\b', text_lower, re.IGNORECASE)
                for header in skills_headers
            )
            
            # Get the pattern from the config if it exists, otherwise build a default pattern
            skill_pattern = self.patterns.get('skills', {}).get(skill_lower)
            if not skill_pattern:
                # Create default pattern from variations
                pattern = r'\b(?:' + '|'.join(map(re.escape, variations)) + r')\b'
            else:
                pattern = skill_pattern
            
            # Search for skill with word boundaries
            found = re.search(pattern, text_lower, re.IGNORECASE)
            
            if found:
                # Get surrounding context with larger window
                start = max(0, found.start() - 150)  # Increased context window
                end = min(len(text), found.end() + 150)
                context = text[start:end].strip()
                
                # Clean context of problematic characters before logging
                clean_context = ''.join(char for char in context if ord(char) < 128)
                logger.info(f"  Found match in: {clean_context}")
                
                # If we're in a skills section, that's enough context
                if in_skills_section:
                    logger.info("  [PASS] Found in skills section")
                    return {
                        'matched': True,
                        'context': context
                    }
                
                # Otherwise check for implementation context
                implementation_indicators = self.config.get('analysis_patterns', {}).get('context', [])
                has_implementation_context = any(
                    indicator.lower() in text_lower
                    for indicator in implementation_indicators
                )
                
                # Return matched=True regardless of context if we found the skill
                return {
                    'matched': True,
                    'context': context
                }
                
            return {'matched': False, 'context': None}
            
        except re.error as e:
            logger.error(f"Invalid regex pattern for skill {skill}: {e}")
            return {'matched': False, 'context': None}

    def calculate_match_score(self, role_name: str, matched_skills: Dict, experience_years: int) -> Dict:
        try:
            role_config = self.config.get('job_roles', {}).get(role_name)
            if not role_config:
                return self._error_response("Invalid role")

            # Get scoring constraints with defaults
            scoring_constraints = role_config.get('scoring_constraints', {})
            max_score = scoring_constraints.get('max_score', 100)
            required_skills_threshold = scoring_constraints.get('required_skills_threshold', 0.85)
            minimum_skills_match = scoring_constraints.get('minimum_skills_match', 0.60)

            # Calculate required skills match
            total_required = len(role_config['required_skills'])
            matched_required = len(matched_skills.get('required', []))
            required_ratio = matched_required / total_required if total_required > 0 else 0

            # Calculate preferred skills match  
            total_preferred = len(role_config['preferred_skills'])
            matched_preferred = len(matched_skills.get('preferred', []))
            preferred_ratio = matched_preferred / total_preferred if total_preferred > 0 else 0

            # Get weights from config
            weights = self.config.get('scoring_config', {}).get('weights', {}).get('technical', {})
            skill_weight = weights.get('skills', 0.7)
            experience_weight = weights.get('experience', 0.3)

            # Calculate skills score (0-100) with adjusted weights
            skills_score = int((required_ratio * 0.85 + preferred_ratio * 0.15) * 100)

            # Calculate experience score (0-100) 
            min_years = role_config.get('min_years_experience', 2)
            if experience_years <= 0:
                experience_score = 0
            elif experience_years >= min_years:
                experience_score = 100
            else:
                experience_score = int((experience_years / min_years) * 100)
                experience_score = max(25, experience_score)

            # Calculate technical match score
            technical_score = int(
                skills_score * skill_weight + 
                experience_score * experience_weight
            )

            # Apply thresholds with adjusted bonuses/penalties
            if required_ratio < minimum_skills_match:
                technical_score = int(technical_score * 0.9)
            elif required_ratio < required_skills_threshold:
                technical_score = int(technical_score * 0.95)
            
            # Apply bonuses for strong matches
            if required_ratio >= 0.67:
                if preferred_ratio >= 0.3:
                    technical_score = int(technical_score * 1.1)
                if experience_years >= min_years:
                    technical_score = int(technical_score * 1.05)

            # Ensure strong matches get appropriate scores
            if required_ratio >= 0.85 and experience_years >= min_years:
                technical_score = max(technical_score, 85)
            elif required_ratio >= 0.67 and experience_years >= min_years:
                technical_score = max(technical_score, 75)

            return {
                'technical_match_score': min(technical_score, max_score),
                'skills_score': skills_score,
                'experience_score': experience_score,
                'analysis': self._generate_analysis(technical_score),
                'skills_breakdown': {
                    'required_match': required_ratio,
                    'preferred_match': preferred_ratio,
                    'required_score': required_ratio,
                    'preferred_score': preferred_ratio
                }
            }

        except Exception as e:
            logger.error(f"Error calculating match score: {str(e)}")
            return self._error_response(str(e))

    def _generate_analysis(self, technical_score: int) -> str:
        """Generate analysis message based on technical score."""
        if technical_score >= self.scoring_config.get('thresholds', {}).get('strong_match', 0):
            return "Strong technical match"
        elif technical_score >= self.scoring_config.get('thresholds', {}).get('good_match', 0):
            return "Good technical match"
        elif technical_score >= self.scoring_config.get('thresholds', {}).get('potential_match', 0):
            return "Potential match with training"
        elif technical_score >= self.scoring_config.get('thresholds', {}).get('minimum_match', 0):
            return "Below expectations"
        else:
            return "Failed required skills threshold"

    def _error_response(self, message: str, case: str = "") -> Dict[str, Any]:
        """Helper method to generate error response with consistent format."""
        analysis = f"Error: {message}"
        if case:
            analysis = f"Error: {case} - {message}"
        return {
            'technical_match_score': 0,
            'skills_score': 0,
            'experience_score': 0,
            'analysis': analysis,
            'skills_breakdown': {
                'required_match': 0.0,
                'preferred_match': 0.0,
                'required_score': 0.0,
                'preferred_score': 0.0
            }
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
