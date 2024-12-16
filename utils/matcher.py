from utils.logging_config import setup_logging
import yaml
import regex as re
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import math
import traceback

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
            
            self.scoring_config = self.config.get('scoring_config', {})

        except Exception as e:
            logger.error(f"Failed to initialize JobMatcher: {str(e)}")
            raise

    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience details with enhanced context awareness."""
        try:
            matches = []
            years = set()
            dates = []
            
            # Extract date ranges with improved context
            date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[a-z]*\.?\s+(\d{4})'
            date_matches = list(re.finditer(date_pattern, text, re.IGNORECASE))
            
            if date_matches:
                dates = [int(m.group(1)) for m in date_matches]
                current_year = datetime.now().year
                
                # Handle "Present" or "Current"
                if any(word in text for word in ["Present", "Current", "Now"]):
                    dates.append(current_year)
                
                if dates:
                    dates = sorted(set(dates))
                    if len(dates) >= 2:
                        total_years = dates[-1] - dates[0]
                        if total_years > 0 and total_years < 50:
                            years.add(total_years)

            # Extract explicit year mentions with context
            for pattern in self.patterns['experience']:
                found = re.finditer(pattern, text, re.IGNORECASE)
                for match in found:
                    match_text = match.group(0)
                    context = self._extract_context(text, match.start(), 100)
                    matches.append({
                        'text': match_text,
                        'context': context
                    })
                    try:
                        year_val = int(match.group(1))
                        if year_val <= 50:
                            years.add(year_val)
                    except (ValueError, IndexError):
                        continue

            # Calculate total experience with education consideration
            total_years = max(years) if years else 0
            if not total_years and dates:
                total_years = dates[-1] - dates[0] if len(dates) >= 2 else 0

            # Add research/education experience bonus with reduced weight
            education_bonus = self._calculate_education_bonus(text)
            if education_bonus > 0:
                # Apply academic multiplier from config
                academic_multiplier = self.scoring_config['experience_weights'].get('academic_multiplier', 0.4)
                total_years = total_years + (education_bonus * academic_multiplier)

            return {
                'matches': matches,
                'years': total_years,
                'all_years': sorted(list(years)),
                'education_bonus': education_bonus,
                'date_ranges': dates
            }
            
        except Exception as e:
            logger.error(f"Experience extraction failed: {str(e)}")
            return {'matches': [], 'years': 0, 'all_years': [], 'education_bonus': 0}

    def match_skills(self, text: str, role_name: str) -> Dict[str, Any]:
        """Match skills with enhanced context analysis."""
        try:
            # Case-insensitive role name matching
            role_config = None
            for config_role_name, config in self.config.get('job_roles', {}).items():
                if config_role_name.lower() == role_name.lower():
                    role_config = config
                    break
                
            if not role_config:
                available_roles = list(self.config.get('job_roles', {}).keys())
                logger.error(f"Role '{role_name}' not found in configuration. Available roles: {available_roles}")
                return {'required': [], 'preferred': [], 'context': {}}
            
            matches = {
                'required': [],
                'preferred': [],
                'context': {},
                'skill_details': []
            }

            text_lower = text.lower()
            
            # Process required skills
            required_skills = role_config.get('required_skills', [])
            for skill in required_skills:
                # Handle both string and dictionary skill formats
                skill_name = skill if isinstance(skill, str) else skill.get('name')
                if not skill_name:
                    continue
                    
                skill_match = self._analyze_skill(skill_name, text_lower, text)
                if skill_match['matched']:
                    matches['required'].append(skill_name)
                    matches['context'][skill_name] = skill_match['context']
                    matches['skill_details'].append({
                        'skill': skill_name,
                        'type': 'required',
                        'context': skill_match['context'],
                        'confidence': skill_match['confidence'],
                        'min_years': skill.get('min_years', 0) if isinstance(skill, dict) else 0,
                        'context_requirement': skill.get('context', '') if isinstance(skill, dict) else ''
                    })

            # Process preferred skills
            for skill in role_config.get('preferred_skills', []):
                # Handle both string and dictionary skill formats
                skill_name = skill if isinstance(skill, str) else skill.get('name')
                if not skill_name:
                    continue
                    
                skill_match = self._analyze_skill(skill_name, text_lower, text)
                if skill_match['matched']:
                    matches['preferred'].append(skill_name)
                    matches['context'][skill_name] = skill_match['context']
                    matches['skill_details'].append({
                        'skill': skill_name,
                        'type': 'preferred',
                        'context': skill_match['context'],
                        'confidence': skill_match['confidence']
                    })

            # Add skill group analysis
            matches['group_analysis'] = self._analyze_skill_groups(
                matches['skill_details'],
                role_config.get('skill_groups', {})
            )

            logger.info(f"Matched required skills: {len(matches['required'])}/{len(role_config.get('required_skills', []))}")
            logger.info(f"Matched preferred skills: {len(matches['preferred'])}/{len(role_config.get('preferred_skills', []))}")
            
            return matches
            
        except Exception as e:
            logger.error(f"Skills matching failed: {str(e)}")
            return {'required': [], 'preferred': [], 'context': {}}

    def _analyze_skill(self, skill: str, text_lower: str, original_text: str) -> Dict[str, Any]:
        """Analyze a single skill with enhanced context awareness."""
        try:
            variations = self._get_skill_variations(skill)
            best_match = None
            highest_confidence = 0
            
            for variation in variations:
                pattern = fr'\b{re.escape(variation)}\b'
                matches = list(re.finditer(pattern, text_lower))
                
                for match in matches:
                    context = self._extract_context(original_text, match.start(), 150)
                    confidence = self._calculate_skill_confidence(context, variation)
                    
                    if confidence > highest_confidence:
                        best_match = {
                            'matched': True,
                            'skill': skill,
                            'variation': variation,
                            'context': context,
                            'confidence': confidence
                        }
                        highest_confidence = confidence

            return best_match if best_match else {
                'matched': False,
                'skill': skill,
                'context': "",
                'confidence': 0.0
            }
            
        except Exception as e:
            logger.error(f"Skill analysis failed for {skill}: {str(e)}")
            return {
                'matched': False,
                'skill': skill,
                'context': "",
                'confidence': 0.0
            }

    def _calculate_skill_confidence(self, context: str, skill: str) -> float:
        """Calculate confidence score for skill match based on context."""
        try:
            confidence = 0.7  # Base confidence for regex match
            
            # Check for strong indicators
            strong_indicators = [
                'developed', 'implemented', 'architected', 'designed',
                'led', 'managed', 'expertise in', 'specialized in'
            ]
            
            # Check for moderate indicators
            moderate_indicators = [
                'used', 'worked with', 'familiar with', 'experience in',
                'knowledge of', 'skilled in'
            ]
            
            # Check for weak indicators
            weak_indicators = [
                'learning', 'basic', 'fundamental', 'exposure to'
            ]
            
            context_lower = context.lower()
            
            # Adjust confidence based on indicators
            for indicator in strong_indicators:
                if indicator in context_lower:
                    confidence = min(1.0, confidence + 0.2)
                    
            for indicator in moderate_indicators:
                if indicator in context_lower:
                    confidence = min(1.0, confidence + 0.1)
                    
            for indicator in weak_indicators:
                if indicator in context_lower:
                    confidence = max(0.3, confidence - 0.2)
            
            return round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.7  # Return base confidence on error

    def _extract_context(self, text: str, position: int, window_size: int) -> str:
        """Extract context around a position with improved formatting."""
        try:
            start = max(0, position - window_size)
            end = min(len(text), position + window_size)
            
            # Get context window
            context = text[start:end].strip()
            
            # Clean up context
            context = re.sub(r'\s+', ' ', context)
            context = re.sub(r'[^\w\s.,;:-]', '', context)
            
            return context
            
        except Exception as e:
            logger.error(f"Context extraction failed: {str(e)}")
            return ""

    def _calculate_education_bonus(self, text: str) -> float:
        """Calculate education-based experience bonus."""
        try:
            bonus = 0.0
            text_lower = text.lower()
            
            # PhD bonus
            if re.search(r'ph\.?d|doctor of philosophy|dissertation', text_lower):
                bonus += 3.0
            # Masters bonus
            elif re.search(r'master|ms|m\.s|thesis', text_lower):
                bonus += 1.5
            # Research experience bonus
            if re.search(r'research\s+(?:assistant|associate|fellow)', text_lower):
                bonus += 0.5
            
            return bonus
            
        except Exception as e:
            logger.error(f"Education bonus calculation failed: {str(e)}")
            return 0.0

    def _analyze_skill_groups(self, skill_details: List[Dict], skill_groups: Dict) -> Dict[str, Any]:
        """Analyze skill groups with enhanced context awareness."""
        try:
            group_analysis = {}
            
            for group_name, group_skills in skill_groups.items():
                matched_skills = []
                total_confidence = 0.0
                
                for skill_detail in skill_details:
                    if skill_detail['skill'] in group_skills:
                        matched_skills.append({
                            'skill': skill_detail['skill'],
                            'confidence': skill_detail['confidence'],
                            'context': skill_detail['context']
                        })
                        total_confidence += skill_detail['confidence']
                
                if matched_skills:
                    avg_confidence = total_confidence / len(matched_skills)
                    coverage = len(matched_skills) / len(group_skills)
                    
                    group_analysis[group_name] = {
                        'matched_skills': matched_skills,
                        'coverage': coverage,
                        'average_confidence': avg_confidence,
                        'strength': self._calculate_group_strength(coverage, avg_confidence)
                    }
            
            return group_analysis
            
        except Exception as e:
            logger.error(f"Skill group analysis failed: {str(e)}")
            return {}

    def _calculate_group_strength(self, coverage: float, confidence: float) -> str:
        """Calculate strength of skill group match."""
        try:
            combined_score = coverage * confidence
            
            if combined_score >= 0.8:
                return "STRONG"
            elif combined_score >= 0.6:
                return "MODERATE"
            elif combined_score >= 0.4:
                return "PARTIAL"
            else:
                return "WEAK"
                
        except Exception as e:
            logger.error(f"Group strength calculation failed: {str(e)}")
            return "UNKNOWN"

    def _get_skill_variations(self, skill: str) -> List[str]:
        """Get skill variations from config with fallback handling."""
        try:
            # Handle both string and dict skill formats
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                skill_key = skill_name.lower().replace(' ', '_')
            else:
                skill_key = skill.lower().replace(' ', '_')
                skill_name = skill
            
            variations = set([skill_name.lower()])  # Start with the original skill
            
            # Add configured variations
            if skill_key in self.skill_variations:
                config_variations = self.skill_variations[skill_key]
                if isinstance(config_variations, dict):
                    variations.update(config_variations.get('aliases', []))
                    
                    # Add forms if they exist
                    base_skill = skill_name.lower()
                    for form in config_variations.get('forms', []):
                        variations.add(f"{base_skill} {form}")
            
            # Add common variations
            variations.update([
                skill_name.lower().replace(' ', ''),
                skill_name.lower().replace(' ', '-'),
                skill_name.lower().replace('-', ' '),
                f"{skill_name.lower()}s",  # Plural
                f"{skill_name.lower()}ing"  # Gerund
            ])
            
            return list(variations)
            
        except Exception as e:
            logger.error(f"Skill variation generation failed for {skill}: {str(e)}")
            return [skill.lower() if isinstance(skill, str) else skill.get('name', '').lower()]

    def _extract_experience_details(self, text: str) -> Dict[str, Any]:
        """Extract detailed experience information including location and type context."""
        try:
            # Initialize result structure
            details = {
                'us_years': 0.0,
                'non_us_years': 0.0,
                'professional_years': 0.0,
                'academic_years': 0.0,
                'internship_years': 0.0,
                'context': 'unknown',
                'total_years': 0.0
            }
            
            # Get patterns from config
            us_pattern = self.config['scoring_config']['location_patterns']['us']
            non_us_pattern = self.config['scoring_config']['location_patterns']['non_us']
            professional_patterns = self.config['scoring_config']['experience_patterns']['professional']
            academic_patterns = self.config['scoring_config']['experience_patterns']['academic']
            internship_patterns = self.config['scoring_config']['experience_patterns']['internship']
            
            # Extract total experience
            exp_info = self.extract_experience(text)
            total_years = exp_info['years']
            
            # Determine location context
            us_matches = len(re.findall(us_pattern, text, re.I))
            non_us_matches = len(re.findall(non_us_pattern, text, re.I))
            
            # Calculate location-based experience
            if us_matches > non_us_matches:
                details['us_years'] = total_years
                details['context'] = 'us'
            else:
                details['non_us_years'] = total_years
                details['context'] = 'non_us'
            
            # Determine experience type context
            professional_matches = sum(len(re.findall(pattern, text, re.I)) for pattern in professional_patterns)
            academic_matches = sum(len(re.findall(pattern, text, re.I)) for pattern in academic_patterns)
            internship_matches = sum(len(re.findall(pattern, text, re.I)) for pattern in internship_patterns)
            
            # Calculate experience type distribution
            total_matches = professional_matches + academic_matches + internship_matches
            if total_matches > 0:
                details['professional_years'] = (professional_matches / total_matches) * total_years
                details['academic_years'] = (academic_matches / total_matches) * total_years
                details['internship_years'] = (internship_matches / total_matches) * total_years
            else:
                # Default to professional if no clear indicators
                details['professional_years'] = total_years
            
            details['total_years'] = total_years
            
            return details
            
        except Exception as e:
            logger.error(f"Experience details extraction failed: {str(e)}")
            return {
                'us_years': 0.0,
                'non_us_years': 0.0,
                'professional_years': 0.0,
                'academic_years': 0.0,
                'internship_years': 0.0,
                'context': 'unknown',
                'total_years': 0.0
            }

    def _calculate_experience_score(self, text: str, role_config: Dict) -> float:
        """Calculate experience score with reduced variance."""
        try:
            min_years = role_config.get('min_years_experience', 0)
            
            # Extract experience details
            exp_details = self._extract_experience_details(text)
            
            # Calculate weighted years based on location and experience type
            weights = self.config['scoring_config']['experience_weights']
            
            # Weight US vs non-US experience with reduced difference
            location_weighted_years = (
                exp_details['us_years'] * weights['us_experience'] +
                exp_details['non_us_years'] * weights['non_us_experience'] * 0.4  # Was 0.2
            )
            
            # Weight professional vs academic/internship experience with reduced penalties
            type_weighted_years = (
                exp_details['professional_years'] * weights.get('professional_multiplier', 1.0) +
                exp_details['academic_years'] * weights.get('academic_multiplier', 0.3) +  # Was 0.1
                exp_details['internship_years'] * weights.get('internship_multiplier', 0.1)  # Was 0.02
            )
            
            # Combine weighted years with reduced emphasis
            total_weighted_years = (location_weighted_years * 0.3) + (type_weighted_years * 0.7)  # Was 0.1/0.9
            
            # Calculate base score with reduced penalties
            if total_weighted_years >= min_years:
                base_score = 1.0
            else:
                # Reduced power for gap ratio penalty
                gap_ratio = total_weighted_years / min_years
                base_score = math.pow(gap_ratio, 2)  # Was 5
            
            # Apply more moderate penalties
            penalties = 1.0
            
            # Significant shortfall penalty with higher floor
            if total_weighted_years < (min_years * 0.7):
                penalties *= 0.8  # Was 0.4
            
            # Professional experience ratio with reduced penalties
            if exp_details['total_years'] > 0:
                professional_ratio = exp_details['professional_years'] / exp_details['total_years']
                if professional_ratio < 0.8:
                    penalties *= 0.9  # Was 0.6
                if professional_ratio < 0.6:
                    penalties *= 0.9  # Was 0.7
            
            # Non-US experience penalty with reduced impact
            if exp_details['us_years'] < (min_years * 0.8):
                penalties *= 0.7  # Was 0.5
            if exp_details['us_years'] < (min_years * 0.5):
                penalties *= 0.8  # Was 0.7
            
            # Calculate final score
            final_score = base_score * penalties
            
            # Apply context multipliers with reduced penalties
            if exp_details['context'] == 'enterprise':
                final_score *= self.config['scoring_config']['context_weights']['enterprise']
            elif exp_details['context'] == 'research':
                final_score *= self.config['scoring_config']['context_weights']['research'] * 0.6  # Was 0.4
            elif exp_details['context'] == 'academic':
                final_score *= self.config['scoring_config']['context_weights']['academic'] * 0.5  # Was 0.3
            
            # Early career and academic focus penalties with higher floors
            if exp_details['total_years'] < 2:
                final_score *= 0.7  # Was 0.5
            
            if exp_details['academic_years'] > exp_details['professional_years']:
                final_score *= 0.8  # Was 0.6
            
            # Floor for strong matches to reduce variance
            if final_score >= 0.9:
                final_score = 0.9 + ((final_score - 0.9) * 0.3)
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Experience score calculation failed: {str(e)}")
            return 0.0

    def _normalize_score(self, score: float, category: str) -> float:
        """Normalize scores to reduce variance within categories."""
        if category == "strong_match":
            if score >= 90:
                # Compress scores between 90-100 less aggressively
                return 90 + ((score - 90) * 0.8)  # Changed from 0.5 to 0.8
        elif category == "good_match":
            if score >= 80:
                return 80 + ((score - 80) * 0.7)
        return score
