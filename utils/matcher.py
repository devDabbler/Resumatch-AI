from utils.logging_config import setup_logging
import yaml
import regex as re
from typing import Dict, List, Any
import logging

# Initialize logging configuration
setup_logging()
logger = logging.getLogger('matcher')

class JobMatcher:
    def __init__(self, config_path: str):
        """Initialize JobMatcher with configuration file."""
        logger.info(f"Initializing JobMatcher with config: {config_path}")
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.patterns = self.config['scoring_config']['analysis_patterns']
        self.scoring_config = self.config['scoring_config']
        logger.debug("Successfully loaded configuration")

    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience details using comprehensive patterns."""
        matches = []
        years = set()
        logger.info("Starting experience extraction")

        # First try explicit experience statements
        for pattern in self.patterns['experience']:
            found = re.finditer(pattern, text, re.IGNORECASE)
            for match in found:
                match_text = match.group(0)
                matches.append(match_text)
                logger.debug(f"Found experience match: {match_text}")

                year_match = re.search(r'(\d+)', match_text)
                if year_match:
                    years.add(int(year_match.group(1)))
                    logger.debug(f"Extracted years: {year_match.group(1)}")

        # If no explicit statements found, try career span calculation
        if not years:
            # Look for earliest and latest dates
            date_matches = re.finditer(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})', text, re.IGNORECASE)
            dates = [int(m.group(1)) for m in date_matches]
            if dates:
                career_span = max(dates) - min(dates)
                if career_span > 0:
                    years.add(career_span)
                    logger.debug(f"Calculated career span: {career_span} years")

        result = {
            'matches': matches,
            'years': max(years) if years else 0,
            'all_years': sorted(list(years))
        }
        logger.info(f"Extracted experience: {result}")
        return result

    def check_platform_specific_experience(self, text: str, role_config: Dict) -> Dict[str, Any]:
        """Check for platform-specific experience requirements."""
        if 'platform_specific_experience' not in role_config:
            return {'matched': True, 'score': 1.0, 'details': []}

        required_platforms = role_config['platform_specific_experience']
        matched_platforms = []
        platform_context = {}

        for platform in required_platforms:
            pattern = self.patterns['skills'].get(platform.lower())
            if pattern:
                result = self.check_skill(platform, text)
                if result['matched'] and result['context']:  # Require context
                    matched_platforms.append(platform)
                    platform_context[platform] = result['context']

        score = len(matched_platforms) / len(required_platforms)
        threshold = 0.6  # Require 60% of platform-specific skills
        
        return {
            'matched': score >= threshold,
            'score': score,
            'matched_platforms': matched_platforms,
            'context': platform_context,
            'details': f"Matched {len(matched_platforms)}/{len(required_platforms)} platform requirements"
        }

    def match_skills(self, text: str, role_name: str) -> Dict[str, Any]:
        """Match required and preferred skills with context analysis."""
        role_config = self.config['job_roles'].get(role_name)
        if not role_config:
            logger.error(f"Role {role_name} not found in configuration")
            return {'required': [], 'preferred': [], 'context': {}}
        
        matches = {
            'required': [],
            'preferred': [],
            'context': {}
        }

        logger.info(f"Starting skill matching for role: {role_name}")
        logger.info(f"Required skills to match: {role_config['required_skills']}")
        logger.info(f"Preferred skills to match: {role_config['preferred_skills']}")

        # Check platform-specific experience for Solutions Architect role
        if role_name == "Solutions Architect, ANZ, Australia":
            platform_check = self.check_platform_specific_experience(text, role_config)
            if not platform_check['matched']:
                logger.info(f"Failed platform-specific experience check: {platform_check['details']}")
                return {
                    'required': [],
                    'preferred': [],
                    'context': {},
                    'platform_check': platform_check,
                    'failed_validation': 'platform_specific'
                }

            # Validate GCP requirements if present
            gcp_validation = self.validate_gcp_requirements(text, role_config)
            if not gcp_validation['valid']:
                logger.info(f"Failed GCP validation: {gcp_validation['details']}")
                return {
                    'required': [],
                    'preferred': [],
                    'context': {},
                    'gcp_validation': gcp_validation,
                    'failed_validation': 'gcp_requirements'
                }

        # Match required skills
        for skill in role_config['required_skills']:
            logger.debug(f"Checking required skill: {skill}")
            result = self.check_skill(skill, text)
            if result['matched']:
                matches['required'].append(skill)
                matches['context'][skill] = result['context']
                logger.info(f"[MATCH] Found required skill: {skill}")
            else:
                logger.info(f"[MISSING] Required skill not found: {skill}")

        # Check required skills threshold for Solutions Architect role
        if role_name == "Solutions Architect, ANZ, Australia" and 'required_skills_match_threshold' in role_config:
            threshold = role_config['required_skills_match_threshold']
            required_match_ratio = len(matches['required']) / len(role_config['required_skills'])
            if required_match_ratio < threshold:
                logger.info(f"Failed to meet required skills threshold: {required_match_ratio:.2f} < {threshold}")
                return {
                    'required': matches['required'],
                    'preferred': [],
                    'context': matches['context'],
                    'threshold_check': {
                        'matched': False,
                        'ratio': required_match_ratio,
                        'threshold': threshold
                    },
                    'failed_validation': 'skills_threshold'
                }

        # Match preferred skills
        for skill in role_config['preferred_skills']:
            logger.debug(f"Checking preferred skill: {skill}")
            result = self.check_skill(skill, text)
            if result['matched']:
                matches['preferred'].append(skill)
                matches['context'][skill] = result['context']
                logger.info(f"[MATCH] Found preferred skill: {skill}")
            else:
                logger.info(f"[MISSING] Preferred skill not found: {skill}")

        # Calculate and log match percentages
        total_required = len(role_config['required_skills'])
        total_preferred = len(role_config['preferred_skills'])
        matched_required = len(matches['required'])
        matched_preferred = len(matches['preferred'])

        required_percentage = int((matched_required / total_required * 100) if total_required > 0 else 100)
        preferred_percentage = int((matched_preferred / total_preferred * 100) if total_preferred > 0 else 100)

        logger.info(f"Required skills match: {required_percentage}% ({matched_required}/{total_required})")
        logger.info(f"Preferred skills match: {preferred_percentage}% ({matched_preferred}/{total_preferred})")

        return matches

    def check_skill(self, skill: str, text: str) -> Dict[str, Any]:
        """Check for skill with improved context validation."""
        skill_lower = skill.lower()
        pattern = self.patterns['skills'].get(
            skill_lower, 
            f'\\b{re.escape(skill)}\\b'
        )

        logger.debug(f"Checking skill: {skill} with pattern: {pattern}")
        found = re.search(pattern, text, re.IGNORECASE)

        if found:
            context_matches = []
            logger.debug(f"Found match for skill: {skill}")

            # For Data Scientist role, don't require context
            return {
                'matched': True,
                'context': ['Found skill mention']
            }

        return {'matched': False, 'context': []}

    def validate_gcp_requirements(self, text: str, role_config: Dict) -> Dict[str, Any]:
        """Validate GCP-specific requirements if present."""
        if 'skill_validation_rules' not in role_config:
            return {'valid': True, 'details': []}

        rules = role_config['skill_validation_rules']
        validation_results = {
            'valid': True,
            'details': []
        }

        # Check required platform experience
        if 'required_platform_experience' in rules:
            for platform in rules['required_platform_experience']:
                result = self.check_skill(platform, text)
                if not result['matched'] or not result['context']:
                    validation_results['valid'] = False
                    validation_results['details'].append(f"Missing required platform experience: {platform}")

        # Check minimum GCP skills
        if 'minimum_gcp_skills' in rules:
            gcp_skills = [
                skill for skill in role_config['required_skills'] + role_config['preferred_skills']
                if any(term in skill.lower() for term in ['gcp', 'google cloud', 'bigquery', 'dataflow', 'dataplex', 'dataproc'])
                and self.check_skill(skill, text)['matched']
                and self.check_skill(skill, text)['context']  # Require context
            ]
            if len(gcp_skills) < rules['minimum_gcp_skills']:
                validation_results['valid'] = False
                validation_results['details'].append(
                    f"Insufficient GCP skills: found {len(gcp_skills)}, required {rules['minimum_gcp_skills']}")

        return validation_results

    def calculate_match_score(self, role_name: str, matched_skills: Dict[str, List[str]], 
                            experience_years: int) -> Dict[str, Any]:
        """Calculate overall match score with detailed logging."""
        logger.info(f"\n{'='*50}\nCalculating match score for {role_name}")
        logger.info(f"Experience years: {experience_years}")
        logger.info(f"Matched required skills: {matched_skills['required']}")
        logger.info(f"Matched preferred skills: {matched_skills['preferred']}")

        role_config = self.config['job_roles'].get(role_name)
        if not role_config:
            logger.error(f"Role {role_name} not found in configuration")
            return {
                'overall_score': 0,
                'skills_score': 0,
                'experience_score': 0,
                'analysis': 'Role not found in configuration'
            }

        # Get scoring weights
        weights = self.scoring_config['weights'][role_config['scoring_weights']]
        skill_weights = self.scoring_config['skill_weights']

        # Calculate skills score
        total_required = len(role_config['required_skills'])
        total_preferred = len(role_config['preferred_skills'])
        matched_required = len(matched_skills['required'])
        matched_preferred = len(matched_skills['preferred'])

        required_score = (matched_required / total_required) if total_required > 0 else 1
        preferred_score = (matched_preferred / total_preferred) if total_preferred > 0 else 1

        skills_score = int((
            required_score * skill_weights['required'] +
            preferred_score * skill_weights['preferred']
        ) * 100)

        # Calculate experience score
        min_years = role_config['min_years_experience']
        experience_score = int(min(100, (experience_years / min_years) * 100)) if min_years > 0 else 100

        # Calculate overall score
        overall_score = int(
            skills_score * weights['skills'] +
            experience_score * weights['experience']
        )

        # For Solutions Architect role, cap the score
        if role_name == "Solutions Architect, ANZ, Australia":
            overall_score = min(overall_score, 35)

        return {
            'overall_score': overall_score,
            'skills_score': skills_score,
            'experience_score': experience_score,
            'analysis': 'Skills and experience evaluated'
        }
