import re
from typing import Dict, List, Tuple, Optional
import yaml
from sentence_transformers import SentenceTransformer
import numpy as np
from .schemas import ResumeData, AnalysisResult, InterviewQuestion, Experience
import logging
import os
from dotenv import load_dotenv
import groq
import fitz  # PyMuPDF
import json

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        
    def analyze_resume(self, resume_data: ResumeData, job_role: str) -> AnalysisResult:
        """Main analysis pipeline for resume evaluation."""
        # Calculate technical match score
        tech_score = self._calculate_technical_score(resume_data, job_role)
        
        # Generate LLM analysis
        llm_analysis = self._get_llm_analysis(resume_data, job_role)
        
        # Combine scores and analysis
        return AnalysisResult(
            technical_match_score=tech_score,
            **llm_analysis
        )

    def _calculate_technical_score(self, resume: ResumeData, job_role: str) -> int:
        """Calculate technical match score based on skills and experience."""
        job_config = self.config['job_roles'][job_role]
        scoring_constraints = job_config.get('scoring_constraints', {})
        
        # Get weights from config
        experience_weight = scoring_constraints.get('experience_weight', 0.35)
        skills_weight = scoring_constraints.get('skills_weight', 0.35)
        location_weight = scoring_constraints.get('location_weight', 0.15)
        context_weight = scoring_constraints.get('context_weight', 0.15)
        
        # Calculate experience score
        exp_score = 0
        required_skills = job_config['required_skills']
        
        # Handle both string and dictionary skill formats
        if required_skills and isinstance(required_skills[0], dict):
            required_skills_dict = {skill['name']: skill for skill in required_skills}
            preferred_skills_dict = {skill['name']: skill for skill in job_config['preferred_skills']}
            total_required_years = sum(skill.get('min_years', 1) for skill in required_skills_dict.values())
        else:
            required_skills_dict = {skill: {'name': skill} for skill in required_skills}
            preferred_skills_dict = {skill: {'name': skill} for skill in job_config['preferred_skills']}
            total_required_years = len(required_skills)  # Default to 1 year per skill
        
        for exp in resume.experiences:
            # Check experience against required skills
            for skill_name, skill_info in required_skills_dict.items():
                if skill_name.lower() in exp.description.lower():
                    min_years = skill_info.get('min_years', 1)
                    exp_score += min(exp.duration, min_years)
            
            # Add location bonus
            if exp.location.lower() == 'us':
                exp_score *= (1 + location_weight)
        
        # Normalize experience score
        exp_score = min(100, (exp_score / total_required_years) * 100) if total_required_years > 0 else 0
        
        # Calculate skills score with semantic matching
        skill_matches = self._match_skills(
            resume.skills.keys(), 
            set(required_skills_dict.keys()), 
            set(preferred_skills_dict.keys())
        )
        
        # Calculate context bonus
        context_bonus = 0
        for skill, contexts in resume.skills.items():
            if skill in required_skills_dict:
                req_context = required_skills_dict[skill].get('context', '').lower()
                if req_context and any(req_context in ctx.lower() for ctx in contexts):
                    context_bonus += 1
            elif skill in preferred_skills_dict:
                pref_context = preferred_skills_dict[skill].get('context', '').lower()
                if pref_context and any(pref_context in ctx.lower() for ctx in contexts):
                    context_bonus += 0.5
        
        context_score = min(100, (context_bonus / (len(required_skills_dict) + len(preferred_skills_dict) * 0.5)) * 100)
        
        # Calculate final skill score with required skills threshold
        required_ratio = skill_matches['required']
        preferred_ratio = skill_matches['preferred']
        required_threshold = scoring_constraints.get('required_skills_threshold', 0.6)
        
        if required_ratio < required_threshold:
            skill_score = required_ratio * 100 * 0.85  # Apply penalty but less severe
        else:
            skill_score = (required_ratio * 0.7 + preferred_ratio * 0.3) * 100
        
        # Combine all scores with weights
        final_score = (
            exp_score * experience_weight +
            skill_score * skills_weight +
            context_score * context_weight
        )
        
        return min(100, max(0, int(final_score)))

    def _match_skills(self, resume_skills: List[str], required: set, preferred: set) -> Dict[str, float]:
        """Match skills using semantic similarity."""
        resume_embeddings = self.model.encode(resume_skills)
        
        def get_match_score(skill_set: set) -> float:
            if not skill_set:
                return 1.0
            skill_embeddings = self.model.encode(list(skill_set))
            similarities = np.max(np.dot(resume_embeddings, skill_embeddings.T), axis=0)
            return np.mean(similarities > self.config['scoring_config']['semantic_threshold'])

        return {
            'required': get_match_score(required),
            'preferred': get_match_score(preferred)
        }

    def _get_llm_analysis(self, resume: ResumeData, job_role: str) -> Dict:
        """Get detailed analysis from LLM with enhanced interview questions."""
        job_config = self.config['job_roles'][job_role]
        required_skills = job_config['required_skills']
        preferred_skills = job_config['preferred_skills']
        
        # Handle both string and dictionary skill formats
        if required_skills and isinstance(required_skills[0], dict):
            required_skills_list = [
                {"name": skill["name"], "context": skill.get("context", ""), "min_years": skill.get("min_years", 1)}
                for skill in required_skills
            ]
            preferred_skills_list = [
                {"name": skill["name"], "context": skill.get("context", ""), "min_years": skill.get("min_years", 1)}
                for skill in preferred_skills
            ]
        else:
            required_skills_list = [
                {"name": skill, "context": "", "min_years": 1}
                for skill in required_skills
            ]
            preferred_skills_list = [
                {"name": skill, "context": "", "min_years": 1}
                for skill in preferred_skills
            ]
        
        # Create detailed context for LLM
        context = {
            "role": job_role,
            "required_skills": required_skills_list,
            "preferred_skills": preferred_skills_list,
            "additional_context": job_config.get("additional_context", {}),
            "skills_found": resume.skills
        }
        
        prompt = f"""Analyze this resume for a {job_role} position and provide:
1. Technical gaps analysis based on required and preferred skills
2. Customized interview questions that:
   - Focus on technical implementation details
   - Verify claimed experience
   - Assess understanding of {job_role} best practices
   - Explore candidate's experience with modern tools and methodologies
3. Key findings about the candidate's experience and potential

Resume Text:
{resume.raw_text}

Role Context:
{json.dumps(context, indent=2)}

Return a JSON object with:
{{
    "technical_gaps": [
        {{
            "skill": "string",
            "gap_type": "missing" | "insufficient_experience" | "context_mismatch",
            "recommendation": "string"
        }}
    ],
    "interview_questions": [
        {{
            "category": "string",
            "question": "string",
            "context": "string",
            "expected_topics": ["string"]
        }}
    ],
    "key_findings": ["string"],
    "concerns": ["string"]
}}"""

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.config['llm_config']['system_prompt']},
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                temperature=0.2,
                max_tokens=2000,
                top_p=0.1
            )
            
            response = json.loads(completion.choices[0].message.content)
            return self._enhance_llm_response(response, job_config)
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return self._get_default_analysis()

    def _enhance_llm_response(self, response: Dict, job_config: Dict) -> Dict:
        """Enhance LLM response with additional context and validation."""
        # Ensure all technical gaps have recommendations
        for gap in response.get('technical_gaps', []):
            if not gap.get('recommendation'):
                skill_name = gap.get('skill', '')
                if skill_name in job_config.get('additional_context', {}):
                    gap['recommendation'] = f"Focus on: {', '.join(job_config['additional_context'][skill_name][:2])}"

        # Enhance interview questions with role-specific context
        for question in response.get('interview_questions', []):
            category = question.get('category', '').lower()
            if category in job_config.get('additional_context', {}):
                context_points = job_config['additional_context'][category]
                question['expected_topics'] = context_points[:3]  # Add top 3 relevant points

        return response

    def _format_skills_context(self, skills: Dict[str, List[str]]) -> str:
        """Format skills and their contexts for LLM prompt."""
        return "\n".join([f"{skill}: {', '.join(contexts)}" for skill, contexts in skills.items()])

    def _format_experience_context(self, experiences: List[Experience]) -> str:
        """Format experiences for LLM prompt."""
        return "\n".join([
            f"{exp.type} ({exp.duration} years, {exp.location}): {exp.description}"
            for exp in experiences
        ])

    def _get_default_analysis(self) -> Dict:
        """Return default analysis in case of LLM failure."""
        return {
            'recommendation': 'POTENTIAL_MATCH',
            'interview_questions': [],
            'technical_gaps': [],
            'key_findings': ["Analysis incomplete due to processing error"],
            'concerns': ["Unable to complete detailed analysis"],
            'confidence_score': 0.5
        }

    def extract_text_from_path(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
