from typing import List, Dict, Tuple
import re
from datetime import datetime
from dateutil.parser import parse
import logging

logger = logging.getLogger(__name__)

class ExperienceValidator:
    def __init__(self, config: Dict):
        self.config = config
        self.experience_types = {
            'professional': re.compile(r'(?i)' + '|'.join(config['experience_types']['professional'])),
            'internship': re.compile(r'(?i)' + '|'.join(config['experience_types']['internship'])),
            'academic': re.compile(r'(?i)' + '|'.join(config['experience_types']['academic']))
        }
        self.match_score_cache = {}

    def validate_experience(self, text: str) -> Tuple[str, float]:
        """
        Validate and categorize experience entries.
        Returns tuple of (experience_type, duration_in_years)
        """
        # Determine experience type
        exp_type = self._determine_experience_type(text)
        
        # Extract duration
        duration = self._calculate_duration(text)
        
        return exp_type, duration

    def _determine_experience_type(self, text: str) -> str:
        """Determine the type of experience based on keywords."""
        for exp_type, pattern in self.experience_types.items():
            if pattern.search(text):
                return exp_type
        return 'professional'  # Default to professional if no specific type found

    def _calculate_duration(self, text: str) -> float:
        """Calculate duration of experience in years."""
        try:
            # Try to find date ranges in the text
            dates = self._extract_dates(text)
            if len(dates) >= 2:
                start_date, end_date = dates[0], dates[-1]
                return self._years_between(start_date, end_date)
            
            # If no dates found, try to find explicit duration mentions
            duration_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text, re.IGNORECASE)
            if duration_match:
                return float(duration_match.group(1))
            
        except Exception as e:
            logger.warning(f"Error calculating duration: {str(e)}")
        
        return 0.0  # Default to 0 if duration cannot be determined

    def _extract_dates(self, text: str) -> List[datetime]:
        """Extract dates from text."""
        dates = []
        
        # Common date patterns
        patterns = [
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
            r'Dec(?:ember)?)[,]?\s+\d{4}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{4}'  # Year only
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    date_str = match.group()
                    # Handle 'Present' or 'Current'
                    if any(word in text[match.start()-10:match.end()+10].lower() 
                          for word in ['present', 'current']):
                        dates.append(datetime.now())
                    else:
                        dates.append(parse(date_str))
                except (ValueError, TypeError):
                    continue
        
        return sorted(dates)

    def _years_between(self, start: datetime, end: datetime) -> float:
        """Calculate years between two dates."""
        diff = end - start
        return round(diff.days / 365.25, 1)  # Account for leap years

    def validate_minimum_experience(self, total_experience: float, 
                                 required_years: float) -> Tuple[bool, str]:
        """Validate if the total experience meets the minimum requirement."""
        if total_experience >= required_years:
            return True, f"Meets minimum experience requirement of {required_years} years"
        else:
            gap = required_years - total_experience
            return False, f"Falls short of minimum experience by {gap:.1f} years"

    def analyze_resume(self, text: str, job_role: str, skills: Dict, experience: Dict) -> Dict:
        """Analyze resume and provide detailed insights."""
        # Calculate match score once and cache it
        match_score = self._calculate_match_score(skills, job_role)
        
        # Get LinkedIn profile data if available
        linkedin_data = experience.get('linkedin_data', {})
        linkedin_analysis = self._analyze_linkedin_profile(text, linkedin_data) if linkedin_data else {}
        
        analysis = {
            'technical_match_score': match_score,
            'recommendation': self._get_recommendation(skills, experience, job_role, match_score),
            'technical_gaps': self._identify_gaps(skills, job_role),
            'interview_questions': self._generate_questions(skills, experience, job_role, linkedin_analysis),
            'key_findings': self._generate_findings(skills, experience, job_role, linkedin_analysis),
            'concerns': self._identify_concerns(skills, experience, job_role, linkedin_analysis),
            'linkedin_verification': linkedin_analysis.get('verification_status', None),
            'confidence_score': self._calculate_confidence(skills, experience, linkedin_analysis)
        }
        return analysis

    def _analyze_linkedin_profile(self, resume_text: str, linkedin_data: Dict) -> Dict:
        """Compare resume content with LinkedIn profile data."""
        analysis = {
            'verification_status': 'unverified',
            'discrepancies': [],
            'additional_experience': [],
            'missing_experience': [],
            'skill_differences': []
        }
        
        if not linkedin_data:
            return analysis
            
        # Compare experience
        resume_companies = self._extract_companies(resume_text)
        linkedin_companies = set(linkedin_data.get('companies', []))
        
        analysis['missing_experience'] = list(linkedin_companies - resume_companies)
        analysis['additional_experience'] = list(resume_companies - linkedin_companies)
        
        # Compare skills
        resume_skills = set(self._extract_skills(resume_text))
        linkedin_skills = set(linkedin_data.get('skills', []))
        
        analysis['skill_differences'] = {
            'missing_in_resume': list(linkedin_skills - resume_skills),
            'additional_in_resume': list(resume_skills - linkedin_skills)
        }
        
        # Set verification status
        if not analysis['discrepancies'] and not analysis['missing_experience']:
            analysis['verification_status'] = 'verified'
        elif len(analysis['discrepancies']) > 3 or len(analysis['missing_experience']) > 2:
            analysis['verification_status'] = 'major_differences'
        else:
            analysis['verification_status'] = 'minor_differences'
            
        return analysis
        
    def _extract_companies(self, text: str) -> set:
        """Extract company names from text."""
        companies = set()
        experience_section = re.search(r'(?i)(experience|work|employment).*?(?=\n\n|\Z)', text, re.DOTALL)
        if experience_section:
            # Extract company names using common patterns
            company_patterns = [
                r'(?i)(?:at|with|for)\s+([A-Z][A-Za-z0-9\s&]+?)(?=\s+(?:as|where|in|from|,|\.))',
                r'(?m)^([A-Z][A-Za-z0-9\s&]+?)(?=\s+(?:•|\.|,))'
            ]
            for pattern in company_patterns:
                companies.update(re.findall(pattern, experience_section.group(0)))
        return {company.strip() for company in companies if len(company.strip()) > 2}

    def _extract_skills(self, text: str) -> set:
        """Extract skills from text."""
        skills = set()
        skills_section = re.search(r'(?i)(skills|technical skills|expertise).*?(?=\n\n|\Z)', text, re.DOTALL)
        if skills_section:
            # Split by common skill separators and clean up
            skill_list = re.split(r'[,•|]', skills_section.group(0))
            skills.update(skill.strip() for skill in skill_list if len(skill.strip()) > 2)
        return skills

    def _calculate_match_score(self, skills: Dict, job_role: str) -> int:
        """Calculate technical match score based on skills."""
        def dict_to_tuple(d):
            """Convert a dictionary to a hashable tuple recursively."""
            items = []
            for k, v in sorted(d.items()):
                if isinstance(v, dict):
                    items.append((k, dict_to_tuple(v)))
                elif isinstance(v, list):
                    items.append((k, tuple(sorted(v))))
                else:
                    items.append((k, v))
            return tuple(items)
        
        # Convert the entire skills dictionary to a hashable tuple
        cache_key = (job_role, dict_to_tuple(skills))
        
        if cache_key in self.match_score_cache:
            return self.match_score_cache[cache_key]
        
        logger.info(f"\n{'='*50}\nScore Calculation for {job_role}\n{'='*50}")
        
        if job_role not in self.config['job_roles']:
            logger.error(f"Unknown job role: {job_role}")
            return 0
            
        role_config = self.config['job_roles'][job_role]
        required = set(role_config['required_skills'])
        preferred = set(role_config['preferred_skills'])
        
        # Get matched skills from the skills dictionary
        matched_required = set(skills.get('required', []))
        matched_preferred = set(skills.get('preferred', []))
        required_contexts = skills.get('required_context', {})
        preferred_contexts = skills.get('preferred_context', {})
        
        # Log skill matches
        logger.info("\nSkill Coverage:")
        logger.info(f"Required Skills - Found {len(matched_required)}/{len(required)}:")
        logger.info(f"  Matched: {matched_required}")
        logger.info(f"  Missing: {required - matched_required}")
        logger.info(f"\nPreferred Skills - Found {len(matched_preferred)}/{len(preferred)}:")
        logger.info(f"  Matched: {matched_preferred}")
        logger.info(f"  Missing: {preferred - matched_preferred}")
        
        # Calculate skill coverage
        required_coverage = len(matched_required) / len(required) if required else 1.0
        preferred_coverage = len(matched_preferred) / len(preferred) if preferred else 1.0
        
        logger.info(f"\nCoverage Ratios:")
        logger.info(f"  Required Skills Coverage: {required_coverage:.2%}")
        logger.info(f"  Preferred Skills Coverage: {preferred_coverage:.2%}")
        
        # Base score calculation
        if required_coverage < 0.3:
            base_score = required_coverage * 50
            logger.info(f"\nLow Required Skills Coverage - Reduced Scoring Mode")
            logger.info(f"Base Score: {base_score:.1f} (required_coverage * 50)")
        else:
            base_score = (required_coverage * 0.7 + preferred_coverage * 0.3) * 100
            logger.info(f"\nNormal Scoring Mode")
            logger.info(f"Base Score: {base_score:.1f} (required * 0.7 + preferred * 0.3) * 100")
        
        # Context quality scoring
        context_score = 0
        logger.info("\nContext Quality Analysis:")
        
        def score_context(contexts: Dict[str, Dict], skill_type: str) -> float:
            if not contexts:
                logger.info(f"  No context found for {skill_type} skills")
                return 0
                
            quality_indicators = [
                'developed', 'implemented', 'led', 'created', 'designed',
                'years', 'experience', 'expert', 'advanced', 'proficient'
            ]
            
            context_scores = []
            logger.info(f"\n  {skill_type} Skills Context:")
            for skill, context_data in contexts.items():
                context_str = context_data.get('context', '')
                if isinstance(context_str, (list, tuple)):
                    context_str = ' '.join(map(str, context_str))
                elif not isinstance(context_str, str):
                    context_str = str(context_str)
                
                indicators_found = [ind for ind in quality_indicators if ind in context_str.lower()]
                score = min(len(indicators_found) / 2, 1.0)
                # Add confidence boost
                confidence = context_data.get('confidence', 1.0)
                score = score * confidence
                
                context_scores.append(score)
                logger.info(f"    {skill}:")
                logger.info(f"      Context: {context_str[:100]}...")
                logger.info(f"      Indicators found: {indicators_found}")
                logger.info(f"      Confidence: {confidence:.2f}")
                logger.info(f"      Score: {score:.2f}")
            
            avg_score = sum(context_scores) / len(contexts) if context_scores else 0
            logger.info(f"    Average context score: {avg_score:.2f}")
            return avg_score
        
        req_context_score = score_context(required_contexts, "Required") * 15
        pref_context_score = score_context(preferred_contexts, "Preferred") * 10
        context_score = req_context_score + pref_context_score
        
        logger.info(f"\nContext Scores:")
        logger.info(f"  Required Skills Context Score: {req_context_score:.1f}/15")
        logger.info(f"  Preferred Skills Context Score: {pref_context_score:.1f}/10")
        logger.info(f"  Total Context Score: {context_score:.1f}/25")
        
        # Bonus points
        bonus = 0
        logger.info(f"\nBonus Points:")
        logger.info(f"  Total Bonus: {bonus}")
        
        # Final score
        final_score = base_score + context_score + bonus
        logger.info(f"\nFinal Score Calculation:")
        logger.info(f"  Base Score:     {base_score:.1f}")
        logger.info(f"  Context Score:  {context_score:.1f}")
        logger.info(f"  Bonus Points:   {bonus}")
        logger.info(f"  Final Score:    {final_score:.1f}")
        
        self.match_score_cache[cache_key] = min(100, max(0, int(final_score)))
        return self.match_score_cache[cache_key]
        
    def _get_recommendation(self, skills: Dict, experience: Dict, job_role: str, match_score: int = None) -> str:
        """Generate recommendation based on match score and experience."""
        # Use cached score if provided, otherwise calculate
        score = match_score if match_score is not None else self._calculate_match_score(skills, job_role)
        years = experience.get('total_years', 0)
        min_years = self.config['job_roles'][job_role].get('min_years', 0)
        
        # Check for strong matches
        if score >= 85:
            if years >= min_years:
                return "STRONG_MATCH"
            elif years >= min_years * 0.8:
                return "GOOD_MATCH"
        
        # Check for good matches
        if score >= 70:
            if years >= min_years * 0.7:
                return "GOOD_MATCH"
            elif years >= min_years * 0.5:
                return "POTENTIAL_MATCH"
                
        # Check for potential matches
        if score >= 55 or (years >= min_years and score >= 45):
            return "POTENTIAL_MATCH"
            
        return "NO_MATCH"
            
    def _identify_gaps(self, skills: Dict, job_role: str) -> List[str]:
        """Identify missing required and preferred skills."""
        if job_role not in self.config['job_roles']:
            return ["Job role not found in configuration"]
            
        role_config = self.config['job_roles'][job_role]
        required = set(role_config['required_skills'])
        preferred = set(role_config['preferred_skills'])
        
        matched_required = set(skills.get('required', []))
        matched_preferred = set(skills.get('preferred', []))
        
        gaps = []
        
        # Check required skills first
        missing_required = required - matched_required
        if missing_required:
            gaps.extend([f"Missing critical skill: {skill}" for skill in missing_required])
            
        # Then check preferred skills
        missing_preferred = preferred - matched_preferred
        if missing_preferred:
            gaps.extend([f"Could benefit from: {skill}" for skill in missing_preferred])
            
        return gaps
        
    def _generate_questions(self, skills: Dict, experience: Dict, job_role: str, linkedin_analysis: Dict = {}) -> List[Dict]:
        """Generate targeted interview questions based on resume content."""
        from utils.schemas import InterviewQuestion  # Import at function level to avoid circular imports
        
        questions = []
        
        def format_template(template: str, **kwargs) -> str:
            """Format template with provided kwargs, handling missing values."""
            try:
                return template.format(**kwargs)
            except KeyError:
                return None
        
        # Get question templates
        templates = self.config.get('interview_questions', {})
        
        # 1. Technical Implementation Questions
        matched_skills = []
        if isinstance(skills.get('required'), list):
            matched_skills.extend(skills.get('required', []))
        if isinstance(skills.get('preferred'), list):
            matched_skills.extend(skills.get('preferred', []))
            
        for skill in matched_skills:
            skill_context = ""
            if isinstance(skills.get('context'), dict) and skill in skills['context']:
                skill_context = skills['context'][skill]
            elif isinstance(skills.get('context'), dict):
                # Try to find the skill in any of the context values
                for context_key, context_value in skills['context'].items():
                    if skill.lower() in context_key.lower():
                        skill_context = context_value
                        break
            
            if skill_context:
                context_str = skill_context.get('summary', skill_context) if isinstance(skill_context, dict) else str(skill_context)
                for template in templates.get('technical_implementation', []):
                    question = format_template(
                        template['template'],
                        skill=skill,
                        context=context_str[:200]  # Limit context length
                    )
                    if question:
                        questions.append(InterviewQuestion(
                            category='Technical Implementation',
                            question=question,
                            context=f"Based on {skill} usage in resume"
                        ))
        
        # 2. Skill Depth Questions
        for skill in matched_skills:
            skill_context = ""
            if isinstance(skills.get('context'), dict) and skill in skills['context']:
                skill_context = skills['context'][skill]
            elif isinstance(skills.get('context'), dict):
                # Try to find the skill in any of the context values
                for context_key, context_value in skills['context'].items():
                    if skill.lower() in context_key.lower():
                        skill_context = context_value
                        break
            
            confidence = (skill_context.get('confidence', 0) if isinstance(skill_context, dict) 
                        else skills.get('confidence', {}).get(skill, 0))
            
            if skill_context and confidence > 0.7:  # Only for skills with high confidence
                context_str = skill_context.get('summary', skill_context) if isinstance(skill_context, dict) else str(skill_context)
                for template in templates.get('skill_depth', []):
                    question = format_template(
                        template['template'],
                        skill=skill,
                        context=context_str[:200]  # Limit context length
                    )
                    if question:
                        questions.append(InterviewQuestion(
                            category='Technical Depth',
                            question=question,
                            context=f"Deep dive into {skill} expertise"
                        ))
        
        # 3. Gap Assessment Questions
        gaps = self._identify_gaps(skills, job_role)
        if gaps:
            role_config = self.config['job_roles'].get(job_role, {})
            required_skills = role_config.get('required_skills', [])
            
            for gap in gaps:
                missing_skill = gap.replace("Missing critical skill: ", "").replace("Could benefit from: ", "")
                # Find a related skill from their matched skills
                related_skill = next((skill for skill in matched_skills 
                                    if skill in required_skills), 
                                   next(iter(matched_skills)) if matched_skills else None)
                
                if related_skill:
                    for template in templates.get('gap_assessment', []):
                        question = format_template(
                            template['template'],
                            missing_skill=missing_skill,
                            related_skill=related_skill
                        )
                        if question:
                            questions.append(InterviewQuestion(
                                category='Skill Gaps',
                                question=question,
                                context=f"Addressing gap in {missing_skill}"
                            ))
        
        # 4. Project Impact Questions
        if experience.get('positions'):
            positions = experience['positions']
            if isinstance(positions, list):
                recent_positions = positions[:2]  # Focus on recent positions
                for position in recent_positions:
                    if isinstance(position, dict):
                        company = position.get('company', 'your previous company')
                        project = position.get('project', 'your main project')
                    else:
                        company = str(position)
                        project = 'your main project'
                    
                    for template in templates.get('project_impact', []):
                        question = format_template(
                            template['template'],
                            company=company,
                            project=project,
                            skill=next(iter(matched_skills)) if matched_skills else 'your technical skills'
                        )
                        if question:
                            questions.append(InterviewQuestion(
                                category='Project Impact',
                                question=question,
                                context=f"Based on experience at {company}"
                            ))
        
        # Limit to most relevant questions
        return sorted(questions, 
                     key=lambda x: (x.category != 'Technical Implementation', 
                                  x.category != 'Technical Depth'),
                     )[:5]
        
    def _generate_findings(self, skills: Dict, experience: Dict, job_role: str, linkedin_analysis: Dict = {}) -> List[str]:
        """Generate key findings from the analysis."""
        findings = []
        
        # Skills match
        score = self._calculate_match_score(skills, job_role)
        findings.append(f"Technical skill match: {score}%")
        
        # Experience
        years = experience.get('total_years', 0)
        if years > 0:
            findings.append(f"Has {years} years of relevant experience")
            
        # Skill strengths
        matched_required = skills.get('required', [])
        if matched_required:
            findings.append(f"Strong in core skills: {', '.join(matched_required[:3])}")
            
        # LinkedIn verification
        if linkedin_analysis:
            verification_status = linkedin_analysis.get('verification_status', 'unverified')
            findings.append(f"LinkedIn profile verification: {verification_status}")
        
        return findings
        
    def _identify_concerns(self, skills: Dict, experience: Dict, job_role: str, linkedin_analysis: Dict = {}) -> List[str]:
        """Identify potential concerns or red flags."""
        concerns = []
        
        # Check for critical skill gaps
        if job_role in self.config['job_roles']:
            required = set(self.config['job_roles'][job_role]['required_skills'])
            matched = set(skills.get('required', []))
            missing = required - matched
            if len(missing) > len(required) / 2:
                concerns.append("Missing several critical technical skills")
                
        # Check experience
        years = experience.get('total_years', 0)
        if years < 2:
            concerns.append("Limited professional experience")
            
        # Check job history
        positions = experience.get('positions', [])
        if len(positions) > 5:
            concerns.append("Frequent job changes")
            
        # Check LinkedIn discrepancies
        if linkedin_analysis:
            verification_status = linkedin_analysis.get('verification_status', 'unverified')
            if verification_status == 'major_differences':
                concerns.append("Significant discrepancies in LinkedIn profile")
        
        return concerns
        
    def _calculate_confidence(self, skills: Dict, experience: Dict, linkedin_analysis: Dict = {}) -> float:
        """Calculate confidence score for the analysis."""
        confidence = 0.0
        
        # Base confidence on data completeness
        if skills.get('required') or skills.get('preferred'):
            confidence += 0.3
            
        if experience.get('total_years'):
            confidence += 0.2
            
        if experience.get('positions'):
            confidence += 0.2
            
        if experience.get('companies'):
            confidence += 0.2
            
        if skills.get('context'):
            confidence += 0.1
            
        # Adjust confidence based on LinkedIn verification
        if linkedin_analysis:
            verification_status = linkedin_analysis.get('verification_status', 'unverified')
            if verification_status == 'verified':
                confidence += 0.1
            elif verification_status == 'major_differences':
                confidence -= 0.2
        
        return min(confidence, 1.0)
