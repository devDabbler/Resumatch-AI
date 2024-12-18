import re
from typing import List, Dict, Set, Tuple, Any
from fuzzywuzzy import fuzz
import logging

logger = logging.getLogger(__name__)

class PatternMatcher:
    def __init__(self, config: Dict):
        self.config = config
        self.compile_patterns()

    def compile_patterns(self):
        """Compile regex patterns from config for better performance."""
        self.experience_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config['analysis_patterns']['experience']
        ]
        self.context_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config['analysis_patterns']['context']
        ]
        self.location_patterns = {
            location: re.compile(pattern, re.IGNORECASE)
            for location, pattern in self.config['analysis_patterns']['location'].items()
        }

    def extract_experience_years(self, text: str) -> List[float]:
        """Extract years of experience from text using regex patterns."""
        years = []
        for pattern in self.experience_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                try:
                    year = float(match.group(1))
                    if 0 < year < 50:  # Sanity check
                        years.append(year)
                except (IndexError, ValueError):
                    continue
        return years

    def extract_skills_with_context(self, text: str) -> Dict[str, List[str]]:
        """Extract skills and their context from the text with improved accuracy."""
        skills_dict = {}
        paragraphs = re.split(r'\n+', text)
        
        # Load skill variations from config
        skill_variations = self.config.get('skill_variations', {})
        
        for skill, variations in skill_variations.items():
            all_patterns = [skill.lower()]
            
            # Add aliases
            if 'aliases' in variations:
                all_patterns.extend([alias.lower() for alias in variations['aliases']])
            
            # Add related terms
            if 'related_terms' in variations:
                all_patterns.extend([term.lower() for term in variations['related_terms']])
            
            # Add context indicators
            context_indicators = variations.get('context_indicators', [])
            
            for paragraph in paragraphs:
                paragraph_lower = paragraph.lower()
                
                # Check for skill patterns
                if any(re.search(rf'\b{re.escape(pattern)}\b', paragraph_lower) for pattern in all_patterns):
                    # Extract context
                    contexts = []
                    
                    # Look for specific context indicators
                    for indicator in context_indicators:
                        if indicator.lower() in paragraph_lower:
                            context_match = self._extract_context(paragraph, indicator)
                            if context_match:
                                contexts.append(context_match)
                    
                    # If no specific context found, get general context
                    if not contexts:
                        general_context = self._extract_context(paragraph)
                        if general_context:
                            contexts.append(general_context)
                    
                    # Add to skills dictionary
                    if contexts:
                        if skill not in skills_dict:
                            skills_dict[skill] = []
                        skills_dict[skill].extend(contexts)
        
        return skills_dict

    def _extract_context(self, text: str, skill: str = "", context_window: int = 100) -> str:
        """Extract context around a skill mention with improved accuracy."""
        try:
            # If specific skill provided, find its position
            if skill:
                skill_index = text.lower().find(skill.lower())
                if skill_index == -1:
                    return ""
                
                # Extract window around skill mention
                start = max(0, skill_index - context_window)
                end = min(len(text), skill_index + len(skill) + context_window)
                
                # Expand to sentence boundaries
                while start > 0 and text[start] not in '.!?\n':
                    start -= 1
                while end < len(text) and text[end] not in '.!?\n':
                    end += 1
                
                context = text[start:end].strip()
            else:
                # Use context patterns for general context extraction
                for pattern in self.context_patterns:
                    match = pattern.search(text)
                    if match:
                        start = max(0, match.start() - context_window)
                        end = min(len(text), match.end() + context_window)
                        context = text[start:end].strip()
                        break
                else:
                    return ""
            
            # Clean and normalize context
            context = re.sub(r'\s+', ' ', context)
            context = context.strip('.,;')
            
            return context if len(context) >= 10 else ""  # Ensure meaningful context
            
        except Exception as e:
            logger.error(f"Error extracting context: {str(e)}")
            return ""

    def detect_location(self, text: str) -> str:
        """Detect location (US vs non-US) from text."""
        for location, pattern in self.location_patterns.items():
            if pattern.search(text):
                return location
        return "unknown"

    def validate_linkedin_profile(self, linkedin_url: str, resume_text: str) -> Tuple[bool, List[str]]:
        """Validate LinkedIn profile against resume content."""
        discrepancies = []
        
        # Basic URL validation
        if not re.match(r'https?://(?:www\.)?linkedin\.com/in/[\w-]+/?$', linkedin_url):
            return False, ["Invalid LinkedIn URL format"]

        # Extract name from LinkedIn URL
        linkedin_name = linkedin_url.split('/')[-1].replace('-', ' ').strip()
        
        # Find potential name in resume (simple heuristic)
        first_line = resume_text.split('\n')[0]
        name_similarity = fuzz.ratio(linkedin_name.lower(), first_line.lower())
        
        if name_similarity < 70:  # Threshold for name matching
            discrepancies.append("LinkedIn profile name may not match resume name")

        return len(discrepancies) == 0, discrepancies

    def match_skills(self, text: str, job_role: str) -> Dict[str, List[str]]:
        """Match skills with improved semantic understanding and context validation."""
        logger.info(f"\n{'='*50}\nEnhanced Skill Matching Process for {job_role}\n{'='*50}")
        
        if job_role not in self.config['job_roles']:
            logger.error(f"Unknown job role: {job_role}")
            raise ValueError(f"Unknown job role: {job_role}")
            
        role_config = self.config['job_roles'][job_role]
        required_skills = role_config['required_skills']
        preferred_skills = role_config['preferred_skills']
        
        # Convert skills to list of strings if they're dictionaries
        if required_skills and isinstance(required_skills[0], dict):
            required_skills = [skill['name'] for skill in required_skills]
        if preferred_skills and isinstance(preferred_skills[0], dict):
            preferred_skills = [skill['name'] for skill in preferred_skills]
        
        logger.info(f"Required Skills: {required_skills}")
        logger.info(f"Preferred Skills: {preferred_skills}")
        
        text_lower = text.lower()
        matched_required = []
        matched_required_context = {}
        matched_preferred = []
        matched_preferred_context = {}
        
        # Cache for fuzzy matching results to avoid duplicate checks
        skill_match_cache = {}
        
        def fuzzy_match_skill(skill: str) -> Tuple[bool, str, float]:
            if skill in skill_match_cache:
                return skill_match_cache[skill]
                
            skill_lower = skill.lower()
            logger.info(f"\nChecking skill: {skill}")
            
            # Get skill variations and context from config
            variations = self.config.get('skill_variations', {}).get(skill_lower, {})
            skill_patterns = [skill_lower] + [v.lower() for v in variations.get('aliases', [])]
            
            # Direct pattern matching
            for pattern in skill_patterns:
                if re.search(rf'\b{re.escape(pattern)}\b', text_lower):
                    context = self._extract_context(text, pattern)
                    result = (True, context, 1.0)
                    skill_match_cache[skill] = result
                    return result
            
            # Context-based matching
            context_indicators = variations.get('context_indicators', [])
            for indicator in context_indicators:
                if indicator.lower() in text_lower:
                    context = self._extract_context(text, indicator)
                    if context:
                        result = (True, context, 0.9)
                        skill_match_cache[skill] = result
                        return result
            
            # Fuzzy matching as fallback
            ratio = fuzz.partial_ratio(skill_lower, text_lower)
            if ratio > 85:
                context = self._extract_context(text, skill_lower)
                result = (True, context, ratio / 100)
                skill_match_cache[skill] = result
                return result
            
            result = (False, "", 0.0)
            skill_match_cache[skill] = result
            return result
        
        # Match required skills
        for skill in required_skills:
            matched, context, confidence = fuzzy_match_skill(skill)
            if matched:
                matched_required.append(skill)
                if context:
                    matched_required_context[skill] = {
                        'context': context,
                        'confidence': confidence
                    }
        
        # Match preferred skills
        for skill in preferred_skills:
            matched, context, confidence = fuzzy_match_skill(skill)
            if matched:
                matched_preferred.append(skill)
                if context:
                    matched_preferred_context[skill] = {
                        'context': context,
                        'confidence': confidence
                    }
        
        logger.info(f"\nMatched Required Skills ({len(matched_required)}/{len(required_skills)}): {matched_required}")
        logger.info(f"Matched Preferred Skills ({len(matched_preferred)}/{len(preferred_skills)}): {matched_preferred}")
        
        return {
            'required': matched_required,
            'preferred': matched_preferred,
            'required_context': matched_required_context,
            'preferred_context': matched_preferred_context
        }
    
    def _extract_context(self, text: str, skill: str = "", context_window: int = 100) -> str:
        """Extract context around a skill mention with improved accuracy."""
        try:
            # If specific skill provided, find its position
            if skill:
                skill_index = text.lower().find(skill.lower())
                if skill_index == -1:
                    return ""
                
                # Extract window around skill mention
                start = max(0, skill_index - context_window)
                end = min(len(text), skill_index + len(skill) + context_window)
                
                # Expand to sentence boundaries
                while start > 0 and text[start] not in '.!?\n':
                    start -= 1
                while end < len(text) and text[end] not in '.!?\n':
                    end += 1
                
                context = text[start:end].strip()
            else:
                # Use context patterns for general context extraction
                for pattern in self.context_patterns:
                    match = pattern.search(text)
                    if match:
                        start = max(0, match.start() - context_window)
                        end = min(len(text), match.end() + context_window)
                        context = text[start:end].strip()
                        break
                else:
                    return ""
            
            # Clean and normalize context
            context = re.sub(r'\s+', ' ', context)
            context = context.strip('.,;')
            
            return context if len(context) >= 10 else ""  # Ensure meaningful context
            
        except Exception as e:
            logger.error(f"Error extracting context: {str(e)}")
            return ""

    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience details from text."""
        experience_details = {
            'total_years': 0,
            'positions': [],
            'companies': [],
            'skills_context': {},
            'education': [],
            'certifications': []
        }
        
        # Extract years of experience
        years = self.extract_experience_years(text)
        if years:
            experience_details['total_years'] = max(years)
        
        # Extract positions and companies
        text_blocks = text.split('\n\n')
        for block in text_blocks:
            # Look for position patterns
            if any(word in block.lower() for word in ['engineer', 'developer', 'manager', 'architect', 'analyst']):
                position = block.split('\n')[0].strip()
                if len(position) < 100:  # Sanity check
                    experience_details['positions'].append(position)
            
            # Look for company patterns
            if any(word in block.lower() for word in ['inc', 'corp', 'ltd', 'llc']):
                company = block.split('\n')[0].strip()
                if len(company) < 100:  # Sanity check
                    experience_details['companies'].append(company)
        
        return experience_details
