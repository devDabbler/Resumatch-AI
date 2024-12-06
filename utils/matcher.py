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
            self.skill_variations = {
                'machine_learning': ['ml', 'machine learning', 'deep learning', 'neural networks', 'ai', 'artificial intelligence'],
                'deep_learning': ['dl', 'deep learning', 'neural networks', 'cnn', 'rnn', 'lstm', 'transformer'],
                'data_visualization': ['data viz', 'visualization', 'tableau', 'powerbi', 'matplotlib', 'seaborn', 'plotly', 'charts'],
                'big_data': ['hadoop', 'spark', 'distributed computing', 'data pipeline', 'etl', 'data engineering'],
                'statistics': ['statistical', 'stats', 'probability', 'regression', 'hypothesis testing', 'statistical analysis'],
                'time_series_regression': ['time series', 'temporal', 'forecasting', 'arima', 'sarima', 'prophet', 'time series analysis'],
                'python': ['py', 'python3', 'python2', 'pandas', 'numpy', 'scipy', 'scikit']
            }
            
            self.scoring_config = self.config.get('scoring_config', {})

        except Exception as e:
            logger.error(f"Failed to initialize JobMatcher: {str(e)}")
            raise

    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience details using comprehensive patterns."""
        matches = []
        years = set()
        logger.info("Starting experience extraction")

        # Enhanced experience patterns
        experience_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+(?:of\s+)?(?:experience|work|research))?',  # "5+ years experience"
            r'(?:over|more\s+than)\s+(\d+)\s*(?:years?|yrs?)',  # "over 5 years"
            r'(?:experience|work|research).*?(\d+)\+?\s*(?:years?|yrs?)',  # "experience of 5 years"
            r'(\d+)\+?\s*(?:years?|yrs?).*?(?:experience|work|research)',  # "5 years of experience"
            r'career\s+spanning\s+(\d+)\+?\s*(?:years?|yrs?)',  # "career spanning 5 years"
            r'(?:since|from)\s+(\d{4})',  # "since 2015"
        ]

        # First try to extract explicit year mentions
        for pattern in experience_patterns:
            found = re.finditer(pattern, text, re.IGNORECASE)
            for match in found:
                match_text = match.group(0)
                matches.append(match_text)
                try:
                    # Extract year value
                    year_val = int(match.group(1))
                    if "since" in match_text.lower() or "from" in match_text.lower():
                        current_year = datetime.now().year
                        year_val = current_year - year_val
                    if year_val <= 50:  # Sanity check for reasonable years
                        years.add(year_val)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing years: {e}")

        # Enhanced date range detection
        date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[a-z]*\.?\s+(\d{4})'
        date_matches = list(re.finditer(date_pattern, text, re.IGNORECASE))
        
        if date_matches:
            dates = [int(m.group(1)) for m in date_matches]
            current_year = datetime.now().year
            
            # Handle "Present" or "Current"
            if any(word in text for word in ["Present", "Current", "Now"]):
                dates.append(current_year)
            
            if dates:
                # Sort dates to find distinct periods
                dates = sorted(set(dates))
                total_years = 0
                
                # Calculate years for each period
                for i in range(0, len(dates)-1, 2):
                    start = dates[i]
                    end = dates[i+1] if i+1 < len(dates) else current_year
                    period_years = end - start
                    if period_years > 0 and period_years < 50:  # Sanity check
                        total_years += period_years
                        years.add(period_years)

        # Calculate total years, including research experience
        total_years = max(years) if years else 0
        
        # Add bonus for research experience and advanced degrees
        if re.search(r'(?:research|phd|master|thesis|dissertation)', text, re.IGNORECASE):
            research_bonus = 2  # Add 2 years equivalent for research/advanced degree
            total_years = min(total_years + research_bonus, total_years * 1.5)  # Cap at 50% increase
            
        result = {
            'matches': matches,
            'years': total_years,
            'all_years': sorted(list(years))
        }
        logger.info(f"Extracted experience: {result}")
        return result

    def match_skills(self, text: str, role_name: str) -> Dict[str, Any]:
        """Match required and preferred skills with context analysis."""
        role_config = self.config.get('job_roles', {}).get(role_name)
        if not role_config:
            logger.error(f"Role {role_name} not found in configuration")
            return {'required': [], 'preferred': [], 'context': {}}
        
        matches = {
            'required': [],
            'preferred': [],
            'context': {}
        }

        # Match required skills with OR logic support
        for skill in role_config.get('required_skills', []):
            if ' OR ' in skill:
                # Handle OR conditions
                alternatives = [s.strip() for s in skill.split(' OR ')]
                for alt in alternatives:
                    result = self.check_skill(alt, text)
                    if result['matched']:
                        matches['required'].append(skill)
                        matches['context'][skill] = result['context']
                        break
            else:
                result = self.check_skill(skill, text)
                if result['matched']:
                    matches['required'].append(skill)
                    matches['context'][skill] = result['context']

        # Match preferred skills
        for skill in role_config.get('preferred_skills', []):
            result = self.check_skill(skill, text)
            if result['matched']:
                matches['preferred'].append(skill)
                matches['context'][skill] = result['context']

        return matches

    def check_skill(self, skill: str, text: str) -> Dict[str, Any]:
        """Check for skill presence with context validation."""
        skill_lower = skill.lower().replace(' ', '_')
        
        try:
            # Get pattern from config or create one
            if skill_lower in self.patterns['skills']:
                pattern = self.patterns['skills'][skill_lower]
            else:
                # Create pattern from variations or skill name
                variations = []
                
                # Check skill variations
                if skill_lower in self.skill_variations:
                    variations.extend(self.skill_variations[skill_lower])
                
                # Add the original skill name and common variations
                variations.append(skill.lower())
                variations.extend([
                    skill.lower().replace(' ', ''),  # Remove spaces
                    skill.lower().replace(' ', '-'),  # Hyphenated
                    skill.lower().replace(' ', '_')   # Underscored
                ])
                
                # Create pattern that matches any variation
                pattern = r'\b(?:' + '|'.join(map(re.escape, variations)) + r')\b'
                self.patterns['skills'][skill_lower] = pattern
            
            # Search for skill
            found = re.search(pattern, text, re.IGNORECASE)
            if found:
                # Get surrounding context (up to 100 chars)
                start = max(0, found.start() - 50)
                end = min(len(text), found.end() + 50)
                context = text[start:end].strip()
                
                # Check if in skills section - more lenient check
                if re.search(r'(?:skills?|expertise|proficiency|competencies|tools|technologies|technical|strengths)', text, re.IGNORECASE):
                    return {'matched': True, 'context': None}
                
                # More lenient context patterns including research/academic terms
                context_patterns = [
                    r'(?:using|with|in|implemented|developed|built|created|designed|managed|led|worked|utilized|applied)',
                    r'(?:experience|expertise|proficiency|knowledge|understanding|background|familiarity)',
                    r'(?:certification|certified|trained|studied|learned|mastered)',
                    r'(?:projects?|applications?|systems?|platforms?|solutions?|frameworks?|tools?)',
                    r'(?:analysis|analytics|development|implementation|architecture)',
                    r'(?:research|paper|publication|thesis|dissertation|study)',
                    r'(?:algorithm|model|method|technique|approach)',
                    r'(?:developed|implemented|designed|optimized|enhanced)'
                ]
                
                has_context = any(re.search(p, context, re.IGNORECASE) for p in context_patterns)
                return {'matched': True, 'context': context if has_context else None}
                
            return {'matched': False, 'context': None}
            
        except re.error as e:
            logger.error(f"Invalid regex pattern for skill {skill}: {e}")
            return {'matched': False, 'context': None}

    def calculate_match_score(self, role_name: str, matched_skills: Dict, experience_years: int) -> Dict[str, Any]:
        """Calculate match score based on skills and experience."""
        logger.info("\n" + "="*50)
        logger.info(f"Calculating match score for {role_name}")
        logger.info(f"Experience years: {experience_years}")

        # Input validation with proper error cases
        if not role_name or role_name not in self.config.get('job_roles', {}):
            return self._error_response("Invalid role name", 'invalid_role')
        
        if experience_years is None:
            return self._error_response("Invalid experience years", 'none_values')
        
        if experience_years < 0:
            return self._error_response("Invalid experience years", 'negative_years')
        
        if not matched_skills or not isinstance(matched_skills, dict):
            return self._error_response("Invalid skills format", 'missing_required')
        
        if 'required' not in matched_skills:
            return self._error_response("Missing required skills", 'missing_required')

        # Get role configuration
        role_config = self.config.get('job_roles', {}).get(role_name)
        
        # Calculate weighted skills score
        skill_weights = self.scoring_config.get('skill_weights', {})
        required_weight = skill_weights.get('required', 0)
        preferred_weight = skill_weights.get('preferred', 0)
        
        # Calculate required and preferred ratios
        total_required = len(matched_skills.get('required', []))
        min_required = len(role_config.get('required_skills', []))
        required_ratio = total_required / min_required if min_required > 0 else 0.0

        total_preferred = len(matched_skills.get('preferred', []))
        min_preferred = len(role_config.get('preferred_skills', []))
        preferred_ratio = total_preferred / min_preferred if min_preferred > 0 else 0.0
        
        # Calculate experience score with research consideration
        min_years = role_config.get('min_years_experience', 4)
        if experience_years >= min_years:
            experience_score = 100
        elif experience_years >= min_years * 0.75:
            experience_score = 85
        elif experience_years >= min_years * 0.5:
            experience_score = 70
        else:
            experience_score = max(25, (experience_years / min_years) * 100)

        # Calculate raw skills score
        raw_skills = (required_ratio * required_weight + preferred_ratio * preferred_weight) * 100
        skills_score = int(round(raw_skills))

        # Adjust skills score for high skill matches
        if required_ratio >= 0.85 and preferred_ratio >= 0.5:
            skills_score = max(skills_score, 85)  # Ensure minimum 85 for strong skill matches

        # Get role-specific weights
        weights = self.scoring_config.get('weights', {}).get(
            role_config.get('scoring_weights', 'default'),
            self.scoring_config.get('weights', {}).get('default', {})
        )
        skills_weight = weights.get('skills', 0)
        experience_weight = weights.get('experience', 0)

        # Calculate initial weighted score
        raw_score = (skills_score * skills_weight + experience_score * experience_weight)
        weight_sum = skills_weight + experience_weight
        technical_score = int(round(raw_score / weight_sum))

        # Handle special cases first
        if len(matched_skills.get('required', [])) <= 1:
            return {
                'technical_match_score': 0,
                'skills_score': 0,
                'experience_score': experience_score,
                'analysis': 'Failed required skills threshold',
                'skills_breakdown': {
                    'required_match': float(required_ratio * 100),
                    'preferred_match': float(preferred_ratio * 100),
                    'required_score': float(required_ratio),
                    'preferred_score': float(preferred_ratio)
                }
            }

        # Check required skills threshold
        threshold = role_config.get('scoring_constraints', {}).get('required_skills_threshold', 0.6)
        if required_ratio < threshold:
            technical_score = 35  # Fixed score for no match
            return {
                'technical_match_score': technical_score,
                'skills_score': skills_score,
                'experience_score': experience_score,
                'analysis': 'Below required skills threshold',
                'skills_breakdown': {
                    'required_match': float(required_ratio * 100),
                    'preferred_match': float(preferred_ratio * 100),
                    'required_score': float(required_ratio),
                    'preferred_score': float(preferred_ratio)
                }
            }

        # Determine score range based on combined factors
        if required_ratio >= 0.85 and experience_years >= min_years:
            technical_score = max(technical_score, 95)  # Strong match with high experience
        elif required_ratio >= 0.75 and experience_years >= min_years * 0.75:
            technical_score = max(technical_score, 85)  # Strong match
        elif required_ratio >= 0.6:  # Good match threshold
            if experience_years >= min_years * 0.5:
                technical_score = max(technical_score, 75)  # Fixed good match score
            
            # Force good match range
            technical_score = max(70, min(84, technical_score))
        elif required_ratio >= 0.5 and experience_years >= min_years * 0.5:
            technical_score = max(50, min(69, technical_score))   # Potential match: 50-69

        return {
            'technical_match_score': technical_score,
            'skills_score': skills_score,
            'experience_score': experience_score,
            'analysis': self._generate_analysis(technical_score),
            'skills_breakdown': {
                'required_match': float(required_ratio * 100),
                'preferred_match': float(preferred_ratio * 100),
                'required_score': float(required_ratio),
                'preferred_score': float(preferred_ratio)
            }
        }

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
            analysis += f" ({case})"
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
