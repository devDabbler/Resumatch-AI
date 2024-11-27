import yaml
import regex as re
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class JobMatcher:
    def __init__(self, config_path: str):
        """Initialize JobMatcher with configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.patterns = self.config['scoring_config']['analysis_patterns']
        self.scoring_config = self.config['scoring_config']
        
    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience details using comprehensive patterns.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            Dict containing years of experience and matched patterns
        """
        matches = []
        years = set()
        
        for pattern in self.patterns['experience']:
            found = re.finditer(pattern, text, re.IGNORECASE)
            for match in found:
                match_text = match.group(0)
                matches.append(match_text)
                
                # Extract years from match
                year_match = re.search(r'(\d+)', match_text)
                if year_match:
                    years.add(int(year_match.group(1)))
        
        return {
            'matches': matches,
            'years': max(years) if years else 0,
            'all_years': sorted(list(years))
        }
    
    def match_skills(self, text: str, role_name: str) -> Dict[str, Any]:
        """Match required and preferred skills with context analysis.
        
        Args:
            text: Resume text to analyze
            role_name: Job role to match against
            
        Returns:
            Dict containing matched skills and their context
        """
        role_config = self.config['job_roles'].get(role_name)
        if not role_config:
            logger.error(f"Role {role_name} not found in configuration")
            return {'required': [], 'preferred': [], 'context': {}}
        
        matches = {
            'required': [],
            'preferred': [],
            'context': {}
        }
        
        # Helper function to check skill with variations
        def check_skill(skill: str, text: str) -> Dict[str, Any]:
            skill_lower = skill.lower()
            # Get skill pattern from config or create default
            pattern = self.patterns['skills'].get(
                skill_lower, 
                f'\\b{re.escape(skill)}\\b'
            )
            
            found = re.search(pattern, text, re.IGNORECASE)
            if found:
                # Look for context around the skill
                context_matches = []
                for context_pattern in self.patterns['context']:
                    # Replace SKILL placeholder with the actual skill pattern
                    full_pattern = context_pattern.replace('SKILL', pattern)
                    context_found = re.finditer(full_pattern, text, re.IGNORECASE)
                    context_matches.extend([m.group(0) for m in context_found])
                
                return {
                    'matched': True,
                    'context': context_matches
                }
            return {'matched': False, 'context': []}
        
        # Check required skills
        for skill in role_config['required_skills']:
            result = check_skill(skill, text)
            if result['matched']:
                matches['required'].append(skill)
                matches['context'][skill] = result['context']
        
        # Check preferred skills
        for skill in role_config['preferred_skills']:
            result = check_skill(skill, text)
            if result['matched']:
                matches['preferred'].append(skill)
                matches['context'][skill] = result['context']
                
        return matches
    
    def calculate_match_score(self, 
                            role_name: str,
                            matched_skills: Dict[str, List[str]],
                            experience_years: int) -> Dict[str, Any]:
        """Calculate overall match score based on skills and experience.
        
        Args:
            role_name: Job role being matched
            matched_skills: Dict of matched required and preferred skills
            experience_years: Years of experience extracted
            
        Returns:
            Dict containing match scores and analysis
        """
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
        
        skills_score = (
            required_score * skill_weights['required'] +
            preferred_score * skill_weights['preferred']
        ) * 100
        
        # Calculate experience score
        min_years = role_config['min_years_experience']
        experience_score = min(100, (experience_years / min_years) * 100) if min_years > 0 else 100
        
        # Calculate overall score
        overall_score = (
            skills_score * weights['skills'] +
            experience_score * weights['experience']
        )
        
        # Determine match level
        thresholds = self.scoring_config['thresholds']
        if overall_score >= thresholds['strong_match']:
            match_level = 'STRONG_MATCH'
        elif overall_score >= thresholds['good_match']:
            match_level = 'GOOD_MATCH'
        elif overall_score >= thresholds['potential_match']:
            match_level = 'POTENTIAL_MATCH'
        else:
            match_level = 'WEAK_MATCH'
        
        return {
            'overall_score': round(overall_score, 2),
            'skills_score': round(skills_score, 2),
            'experience_score': round(experience_score, 2),
            'match_level': match_level,
            'analysis': {
                'required_skills_matched': matched_skills['required'],
                'preferred_skills_matched': matched_skills['preferred'],
                'missing_required': list(
                    set(role_config['required_skills']) - 
                    set(matched_skills['required'])
                ),
                'years_experience': experience_years,
                'min_years_required': min_years
            }
        }
