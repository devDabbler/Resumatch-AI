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

            # Initialize location patterns
            self.location_patterns = {
                'us': [
                    r'\b(?:united states|usa|u\.s\.a\.|us|america)\b',
                    r'\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b',
                    r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|San Francisco|Charlotte|Indianapolis|Seattle|Denver|Washington DC|Boston|El Paso|Detroit|Nashville|Portland|Memphis|Oklahoma City|Las Vegas|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|Mesa|Sacramento|Atlanta|Kansas City|Colorado Springs|Miami|Raleigh|Omaha|Long Beach|Virginia Beach|Oakland|Minneapolis|Tulsa|Arlington|Tampa)\b'
                ],
                'india': [
                    r'\b(?:india|bharat|hindustan)\b',
                    r'\b(?:mumbai|delhi|bangalore|hyderabad|chennai|kolkata|pune|ahmedabad|jaipur|surat|lucknow|kanpur|nagpur|indore|thane|bhopal|visakhapatnam|pimpri-chinchwad|patna|vadodara|ghaziabad|ludhiana|agra|nashik|faridabad|meerut|rajkot|kalyan-dombivali|vasai-virar|varanasi)\b'
                ]
            }

        except Exception as e:
            logger.error(f"Failed to initialize JobMatcher: {str(e)}")
            raise

    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience details with enhanced location and duration analysis."""
        matches = []
        years = set()
        locations = {'us': [], 'india': [], 'other': []}
        short_stints = []
        gaps = []
        
        logger.info("Starting experience extraction with location analysis")

        try:
            # Extract dates and durations
            date_pattern = r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4})|(?:(?:19|20)\d{2}(?:\s*-\s*(?:present|current|now|\d{4})?)?)'
            dates = re.finditer(date_pattern, text, re.IGNORECASE)
            date_list = []
            
            for date_match in dates:
                date_str = date_match.group(0).lower().strip()
                try:
                    end_date = None  # Initialize end_date
                    
                    # Handle "present" or year ranges
                    if 'present' in date_str or 'current' in date_str or 'now' in date_str:
                        end_date = datetime.now()
                    else:
                        # Extract year
                        year_match = re.search(r'(?:19|20)\d{2}', date_str)
                        if year_match:
                            year = int(year_match.group(0))
                            # Create datetime object for January of that year
                            end_date = datetime(year, 1, 1)
                    
                    if end_date:  # Only append if we successfully parsed a date
                        date_list.append(end_date)
                    
                except ValueError as e:
                    logger.warning(f"Error parsing date {date_str}: {e}")
                    continue
            
            # Sort dates and analyze gaps/durations
            if date_list:
                date_list.sort()
                for i in range(len(date_list) - 1):
                    duration = (date_list[i+1] - date_list[i]).days / 365.25
                    
                    # Check for short stints (less than 1 year)
                    if 0 < duration < 1:
                        short_stints.append(f"{duration:.1f} years")
                    
                    # Check for gaps (more than 6 months between roles)
                    if duration > 1.5:
                        gaps.append(f"{duration:.1f} years")
                    
                    if duration > 0:
                        years.add(round(duration))

            # Analyze locations
            text_blocks = text.split('\n')
            for block in text_blocks:
                block_lower = block.lower()
                
                # Check for US locations
                for pattern in self.location_patterns['us']:
                    if re.search(pattern, block, re.IGNORECASE):
                        locations['us'].append(block.strip())
                        break
                
                # Check for India locations
                for pattern in self.location_patterns['india']:
                    if re.search(pattern, block, re.IGNORECASE):
                        locations['india'].append(block.strip())
                        break

            # Calculate total years
            total_years = max(years) if years else 0
            
            # Calculate location-based experience
            total_locations = len(locations['us']) + len(locations['india']) + len(locations['other'])
            us_experience = len(locations['us']) / max(total_locations, 1)
            india_experience = len(locations['india']) / max(total_locations, 1)
            
            result = {
                'matches': matches,
                'years': total_years,
                'all_years': sorted(list(years)),
                'locations': locations,
                'us_experience_ratio': us_experience,
                'india_experience_ratio': india_experience,
                'short_stints': short_stints,
                'gaps': gaps
            }
            
            logger.info(f"Extracted experience with location analysis: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in experience extraction: {str(e)}")
            return {
                'matches': [],
                'years': 0,
                'all_years': [],
                'locations': locations,
                'us_experience_ratio': 0,
                'india_experience_ratio': 0,
                'short_stints': [],
                'gaps': []
            }

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

        # Get role skills as lowercase for case-insensitive comparison
        required_skills = {skill.lower(): skill for skill in role_config.get('required_skills', [])}
        preferred_skills = {skill.lower(): skill for skill in role_config.get('preferred_skills', [])}

        # Process all potential skills in the text
        # Use the original order from role_config for consistent results
        for skill in role_config.get('required_skills', []) + role_config.get('preferred_skills', []):
            result = self.check_skill(skill, text)
            if result['matched']:
                # Check if it's a required skill
                if skill.lower() in required_skills:
                    matches['required'].append(required_skills[skill.lower()])
                # If not required but preferred, add to preferred
                elif skill.lower() in preferred_skills:
                    matches['preferred'].append(preferred_skills[skill.lower()])
                
                if result['context']:
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
                variations = []
                
                # Check skill variations
                if skill_lower in self.skill_variations:
                    variations.extend(self.skill_variations[skill_lower])
                
                variations.append(skill.lower())
                variations.append(skill.lower().replace('_', ' '))
                variations.append(skill.lower().replace(' ', '-'))
                
                pattern = r'\b(?:' + '|'.join(map(re.escape, variations)) + r')\b'
                self.patterns['skills'][skill_lower] = pattern
            
            # Search for skill
            found = re.search(pattern, text, re.IGNORECASE)
            if found:
                # Get surrounding context
                start = max(0, found.start() - 50)
                end = min(len(text), found.end() + 50)
                context = text[start:end].strip()
                
                # Check if this is just a skills section listing
                if re.search(r'\b(?:skills?|expertise|proficiency|competencies)\s*(?::|section|\(|\)|list)', text, re.IGNORECASE):
                    return {'matched': True, 'context': None}
                
                # Check for implementation context
                context_patterns = [
                    r'(?:using|with|in|implemented|developed|built|created|designed|managed|led|worked|utilized|applied)',
                    r'(?:experience|expertise|proficiency|knowledge|understanding|background)',
                    r'(?:certification|certified|trained|studied|learned|mastered)',
                    r'(?:projects?|applications?|systems?|platforms?|solutions?)',
                    r'(?:analysis|analytics|development|implementation|architecture)'
                ]
                
                has_context = any(re.search(p, context, re.IGNORECASE) for p in context_patterns)
                return {'matched': True, 'context': context if has_context else None}
                
            return {'matched': False, 'context': None}
            
        except re.error as e:
            logger.error(f"Invalid regex pattern for skill {skill}: {e}")
            return {'matched': False, 'context': None}

    def calculate_match_score(self, role_name: str, matched_skills: Dict, experience_years: int) -> Dict[str, Any]:
        """Calculate match score based on skills and experience."""
        role_config = self.config.get('job_roles', {}).get(role_name)
        if not role_config:
            return {'technical_match_score': 0}

        # Get weights from config
        weights = self.config.get('scoring_config', {}).get('weights', {})
        skill_weight = weights.get('skills', 0.7)
        experience_weight = weights.get('experience', 0.3)

        # Calculate skill scores
        required_skills = set(role_config.get('required_skills', []))
        preferred_skills = set(role_config.get('preferred_skills', []))
        
        matched_required = set(matched_skills.get('required', []))
        matched_preferred = set(matched_skills.get('preferred', []))

        required_ratio = len(matched_required) / len(required_skills) if required_skills else 0
        preferred_ratio = len(matched_preferred) / len(preferred_skills) if preferred_skills else 0

        # Calculate weighted skill score (0-100)
        skill_score = (required_ratio * 0.8 + preferred_ratio * 0.2) * 100

        # Calculate experience score (0-100)
        min_years = role_config.get('min_years_experience', 0)
        if experience_years >= min_years:
            experience_score = 100
        else:
            experience_score = (experience_years / min_years) * 100 if min_years > 0 else 0

        # Calculate base score
        base_score = int(round(
            skill_score * skill_weight + experience_score * experience_weight
        ))

        # First check for strong match conditions
        if len(matched_required) == len(required_skills) and experience_years >= min_years:
            return {'technical_match_score': 85}  # Strong match - exact match of all required skills

        # Then check other score thresholds
        if required_ratio >= 0.6 and experience_years >= min_years * 0.8:
            return {'technical_match_score': 70}  # Good match
        elif required_ratio >= 0.4:
            return {'technical_match_score': 50}  # Minimum threshold
        elif required_ratio < 0.3 or experience_years < min_years * 0.3:
            return {'technical_match_score': 25}  # Poor match

        return {'technical_match_score': base_score}

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
