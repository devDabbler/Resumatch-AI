from typing import Dict, Any, Union, Optional, List
import json
import re
import yaml
import os
from utils.schemas import AnalysisResult, ExperienceDetails, MatchStrength, InterviewQuestion
from pydantic import ValidationError
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PromptHandler:
    @classmethod
    def parse_llm_response(cls, response: Dict[str, Any]) -> AnalysisResult:
        """Parse and validate LLM response"""
        try:
            if not response or 'choices' not in response:
                logger.error("Invalid response format")
                return cls._create_fallback_result()

            content = response['choices'][0]['message']['content']
            
            # Parse JSON content
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                return cls._create_fallback_result()

            # Convert to AnalysisResult
            try:
                result = AnalysisResult(
                    technical_match_score=data.get('technical_match_score', 0),
                    recommendation=data.get('recommendation', 'NO_MATCH'),
                    skills_assessment=data.get('skills_assessment', []),
                    technical_gaps=data.get('technical_gaps', []),
                    interview_questions=data.get('interview_questions', []),
                    key_findings=data.get('key_findings', []),
                    concerns=data.get('concerns', []),
                    confidence_score=data.get('confidence_score', 0.0)
                )
                return result
            except ValidationError as e:
                logger.error(f"Failed to validate response data: {str(e)}")
                return cls._create_fallback_result()

        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return cls._create_fallback_result()

    @classmethod
    def _create_fallback_result(cls) -> AnalysisResult:
        """Create a fallback result when parsing fails"""
        return AnalysisResult(
            technical_match_score=0,
            recommendation=MatchStrength.NO_MATCH,
            skills_assessment=[],
            technical_gaps=["Unable to analyze technical skills"],
            interview_questions=[
                InterviewQuestion(
                    category="General Technical",
                    question="Please describe your technical background and experience",
                    context="Fallback question due to analysis failure"
                )
            ],
            key_findings=["Analysis failed - manual review required"],
            concerns=["Unable to automatically assess qualifications"],
            confidence_score=0.0
        )

    @classmethod
    def create_analysis_prompt(
        cls,
        resume_text: str, 
        role_config: Dict[str, Any],
        matched_skills: Optional[Dict] = None,
        tech_contexts: List[Dict] = [],
        quality_issues: Dict = {},
        questions: Optional[List[Union[str, InterviewQuestion]]] = None
    ) -> List[Dict[str, str]]:
        """Create insight-focused analysis prompt"""
        
        # Load configuration
        config = cls._load_config()
        
        # Normalize role configuration
        role_type = role_config.get('role_type', role_config.get('type', 'technical'))
        required_skills = role_config.get('required_skills', [])
        preferred_skills = role_config.get('preferred_skills', [])
        min_years = role_config.get('min_years_experience', 0)

        # Extract deep insights
        insights = cls._analyze_resume_insights(resume_text, role_config, config)
        
        # Generate dynamic questions if not provided
        if questions is None:
            questions = cls._generate_dynamic_questions(insights, config)
        
        # Create and return the prompt messages
        return [
            {
                "role": "system",
                "content": """Analyze this resume focusing on:
1. Technical depth and implementation details
2. Project impact and business results
3. Architecture decisions and trade-offs
4. Growth areas and skill gaps
5. Red flags or concerns

Generate questions that:
- Probe specific technical implementations
- Explore architectural decisions
- Validate claimed impact
- Address potential concerns
- Help assess culture fit

Questions must be based on actual resume content and help assess candidate fit."""
            },
            {
                "role": "user",
                "content": f"""Analyzing {role_type} candidate.

Key Insights:
{json.dumps(insights, indent=2)}

Suggested Questions:
{json.dumps(questions, indent=2)}

Resume Text:
{resume_text}

Job Requirements:
- Role Type: {role_type}
- Required Skills: {', '.join(map(str, required_skills))}
- Preferred Skills: {', '.join(map(str, preferred_skills))}
- Minimum Experience: {min_years} years

Generate a JSON response with:
1. Technical assessment
2. Experience evaluation
3. Specific follow-up questions
4. Recommendations for interviewers

Focus on helping recruiters/hiring managers understand:
- Technical depth and expertise
- Real-world impact and results
- Potential gaps or concerns
- Cultural and team fit

Return ONLY a valid JSON object."""
            }
        ]

    @classmethod
    def _load_config(cls, config_path: str = 'config/jobs.yaml') -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return {}

    @classmethod
    def _analyze_tech_usage_depth(cls, tech: str, resume_text: str) -> Dict[str, Any]:
        """Analyze depth of technology usage in resume."""
        try:
            # Look for context patterns indicating depth
            depth_patterns = [
                r'developed\s+.*' + tech,
                r'architected\s+.*' + tech,
                r'implemented\s+.*' + tech,
                r'optimized\s+.*' + tech
            ]
            
            instances = []
            depth_signals = []
            challenges = []
            patterns = []
            
            for pattern in depth_patterns:
                matches = list(re.finditer(pattern, resume_text, re.I))
                if matches:
                    for match in matches:
                        instances.append(match.group(0))
                        depth_signals.append(match.group(0).split()[0])
            
            return {
                'tech': tech,
                'instances': instances,
                'depth_signals': depth_signals,
                'challenges': challenges,
                'patterns': patterns
            }
        except Exception as e:
            logger.error(f"Tech usage depth analysis failed: {str(e)}")
            return {'tech': tech, 'instances': [], 'depth_signals': [], 'challenges': [], 'patterns': []}

    @classmethod
    def _extract_context(cls, text: str, position: int, window_size: int = 150) -> str:
        """Extract context around a position in text."""
        try:
            start = max(0, position - window_size // 2)
            end = min(len(text), position + window_size // 2)
            
            # Get context window
            context = text[start:end].strip()
            
            # Clean up context
            context = re.sub(r'\s+', ' ', context)
            context = re.sub(r'[^\w\s.,;:-]', '', context)
            
            return context
            
        except Exception as e:
            logger.error(f"Context extraction failed: {str(e)}")
            return ""

    @classmethod
    def _find_technologies_in_context(cls, context: str, config: Dict[str, Any]) -> List[str]:
        """Find technologies mentioned in a given context."""
        try:
            # Use tech patterns from config if available
            tech_patterns = config.get('analysis_patterns', {}).get('skills', {})
            
            found_techs = []
            for tech, pattern in tech_patterns.items():
                if re.search(pattern, context, re.I):
                    found_techs.append(tech)
            
            return found_techs
        except Exception as e:
            logger.error(f"Technology extraction failed: {str(e)}")
            return []

    @classmethod
    def _find_impact_metrics(cls, context: str) -> str:
        """Find impact metrics in a given context."""
        try:
            # Look for percentage improvements, scaling, etc.
            impact_patterns = [
                r'improved\s+by\s+(\d+)%',
                r'increased\s+by\s+(\d+)%',
                r'reduced\s+by\s+(\d+)%',
                r'scaled\s+to\s+(\d+)\s+(?:users|requests|transactions)'
            ]
            
            for pattern in impact_patterns:
                match = re.search(pattern, context, re.I)
                if match:
                    return match.group(0)
            
            return "No specific impact metrics found"
        except Exception as e:
            logger.error(f"Impact metrics extraction failed: {str(e)}")
            return "Unable to extract impact metrics"

    @classmethod
    def _identify_skill_gaps(cls, role_config: Dict, resume_text: str) -> List[Dict]:
        """Identify skill gaps based on role requirements."""
        try:
            required_skills = role_config.get('required_skills', [])
            gaps = []
            
            for skill in required_skills:
                skill_name = skill if isinstance(skill, str) else skill.get('name', '')
                
                # Check if skill is mentioned in resume
                if not re.search(skill_name, resume_text, re.I):
                    gaps.append({
                        'description': f'Missing or minimal experience with {skill_name}',
                        'missing_element': skill_name,
                        'related_skills': ''  # Could be enhanced to find related skills
                    })
            
            return gaps
        except Exception as e:
            logger.error(f"Skill gap identification failed: {str(e)}")
            return []

    @classmethod
    def _analyze_resume_insights(cls, resume_text: str, role_config: Dict, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract deep insights from resume for question generation"""
        insights = {
            'projects': [],
            'tech_usage': [],
            'impact': [],
            'gaps': [],
            'concerns': []
        }
        
        # Extract project details
        project_patterns = config.get('analysis_patterns', {}).get('project_patterns', [])
        for pattern in project_patterns:
            matches = re.finditer(pattern['role'], resume_text, re.I)
            for match in matches:
                project = {
                    'description': match.group(1),
                    'context': cls._extract_context(resume_text, match.start(), 200),
                    'technologies': cls._find_technologies_in_context(match.group(0), config),
                    'impact': cls._find_impact_metrics(match.group(0))
                }
                insights['projects'].append(project)

        # Analyze technology usage depth
        for tech in role_config['required_skills']:
            usage = cls._analyze_tech_usage_depth(tech, resume_text)
            if usage['instances']:
                insights['tech_usage'].append(usage)

        # Find impact and metrics
        impact_patterns = config.get('analysis_patterns', {}).get('insight_patterns', {}).get('strengths', [])
        for pattern in impact_patterns:
            matches = re.finditer(pattern, resume_text, re.I)
            for match in matches:
                insights['impact'].append({
                    'type': pattern['type'],
                    'metric': match.group(0),
                    'context': cls._extract_context(resume_text, match.start(), 150)
                })

        # Identify gaps and concerns
        gaps = cls._identify_skill_gaps(role_config, resume_text)
        insights['gaps'] = gaps

        return insights

    @classmethod
    def _generate_dynamic_questions(cls, insights: Dict, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate dynamic questions based on resume insights"""
        questions = []
        
        # Project-specific questions
        for project in insights['projects']:
            if project['technologies'] and project['impact']:
                questions.append({
                    'type': 'technical_depth',
                    'question': f"In {project['description']}, you used {', '.join(project['technologies'])} "
                              f"to achieve {project['impact']}. Could you elaborate on:\n"
                              f"- The specific technical challenges you faced\n"
                              f"- Your architecture decisions and trade-offs\n"
                              f"- How you measured and validated the impact?"
                })

        # Technology usage questions
        for usage in insights['tech_usage']:
            if usage['depth_signals']:
                questions.append({
                    'type': 'implementation',
                    'question': f"Your experience with {usage['tech']} shows {usage['depth_signals'][0]}. "
                              f"Can you discuss:\n"
                              f"- How you handled {usage['challenges'][0] if usage['challenges'] else 'technical challenges'}\n"
                              f"- Your approach to {usage['patterns'][0] if usage['patterns'] else 'implementation'}\n"
                              f"- What you learned that you'd do differently?"
                })

        # Gap exploration questions
        for gap in insights['gaps']:
            questions.append({
                'type': 'gap_assessment',
                'question': f"I notice {gap['description']}. Could you address:\n"
                          f"- Your exposure to {gap['missing_element']}\n"
                          f"- How your experience with {gap['related_skills']} applies\n"
                          f"- Your approach to bridging this gap?"
            })

        return questions
