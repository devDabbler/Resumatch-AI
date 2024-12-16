from groq import Groq
import google.generativeai as genai
from typing import Dict, List, Any, Union
import os
from dotenv import load_dotenv
import logging
import json
import time
import re
import yaml
import traceback
from utils.logging_config import setup_logging

# Initialize logging configuration
setup_logging()
logger = logging.getLogger('llm_analyzer')

# Set third-party loggers to WARNING to reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load environment variables at the very start
load_dotenv(override=True)

class LLMAnalyzer:
    def __init__(self):
        """Initialize the analyzer with Mixtral and Gemini support."""
        try:
            self.groq_api_key = os.getenv('GROQ_API_KEY')
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            
            self.mixtral_model = "mixtral-8x7b-32768"
            self.gemini_model = "gemini-pro"
            
            self.groq_client = None
            self.gemini_client = None
            
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
                
            if not self.gemini_api_key:
                logger.warning("GEMINI_API_KEY not found - Gemini features disabled")
            
            self.initialize_clients()
        except Exception as e:
            logger.error(f"LLMAnalyzer initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize LLM services: {str(e)}") from e

    def initialize_clients(self) -> None:
        """Initialize the LLM clients with proper error handling."""
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            logger.info("Successfully initialized Groq client")
            
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(self.gemini_model)
                logger.info("Successfully initialized Gemini client")
        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {str(e)}")
            logger.debug(f"Client initialization error details: {traceback.format_exc()}")
            raise RuntimeError("Failed to initialize one or more LLM clients") from e

    def handle_api_error(self, error: Exception, service: str) -> None:
        """Centralized error handling for API calls."""
        error_msg = str(error)
        if "rate limit" in error_msg.lower():
            logger.error(f"{service} API rate limit exceeded")
            raise RuntimeError(f"{service} rate limit exceeded. Please try again later.")
        elif "invalid api key" in error_msg.lower():
            logger.error(f"Invalid {service} API key")
            raise ValueError(f"Invalid {service} API key. Please check your configuration.")
        elif "timeout" in error_msg.lower():
            logger.error(f"{service} API request timed out")
            raise TimeoutError(f"{service} request timed out. Please try again.")
        else:
            logger.error(f"Unexpected {service} API error: {error_msg}")
            logger.debug(f"API error details: {traceback.format_exc()}")
            raise RuntimeError(f"Unexpected error with {service} API: {error_msg}")

    def execute_request(self, messages: List[Dict[str, str]], max_retries: int = 3, retry_delay: int = 2) -> Dict[str, Any]:
        """Execute a request to Groq with enhanced retry logic."""
        if not self.groq_client:
            logger.info("Client not initialized, initializing now...")
            self.initialize_clients()
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Making API request (attempt {attempt + 1}/{max_retries})")
                
                response = self.groq_client.chat.completions.create(
                    messages=messages,
                    model=self.mixtral_model,
                    temperature=0.01,  # Reduced temperature for more deterministic output
                    max_tokens=2000,   # Increased max tokens to avoid truncation
                    top_p=0.1,         # Added top_p for more focused sampling
                    frequency_penalty=0.0,  # No penalty for token frequency
                    presence_penalty=0.0    # No penalty for token presence
                )
                logger.info("API request successful")
                return response
                
            except Exception as e:
                self.handle_api_error(e, "Groq")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed")
                    raise

    def analyze_resume(self, resume_text: str, role_name: str, matched_skills: Dict, extracted_experience: Union[Dict, List[str]]) -> Dict:
        """
        Dual analysis:
        - Mixtral: Technical skills evaluation 
        - Gemini: Experience validation and classification
        """
        try:
            # Calculate technical score
            technical_score = self._calculate_technical_score(matched_skills, role_name)
            
            # Get technical analysis from Mixtral
            technical_analysis = self._mixtral_technical_analysis(
                resume_text, role_name, matched_skills, 
                extracted_experience, technical_score
            )
            
            # Get experience analysis from Gemini
            experience_analysis = self._gemini_experience_analysis(resume_text, role_name)
            
            # Combine analyses
            technical_analysis['experience_details'] = experience_analysis
            
            return technical_analysis
            
        except Exception as e:
            logger.error(f"Resume analysis failed: {str(e)}")
            return self._get_fallback_response()

    def _clean_text(self, text: str) -> str:
        """Clean text to handle Unicode characters and special formatting"""
        try:
            # Replace problematic Unicode characters
            replacements = {
                '\ufb01': 'fi',
                '\u2013': '-',
                '\u2014': '-',
                '\u2018': "'",
                '\u2019': "'",
                '\u201c': '"',
                '\u201d': '"',
                '\u2022': '*',
                '\ufeff': '',
            }
            
            for old, new in replacements.items():
                text = text.replace(old, new)
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            return text.encode('ascii', 'ignore').decode('ascii')

        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
        
    def _format_experience_dict(self, experience_dict: Dict) -> str:
        """Format experience dictionary into a readable summary string."""
        try:
            summary_parts = []
            
            # Add total years
            if 'years' in experience_dict:
                summary_parts.append(f"Total Experience: {experience_dict['years']:.1f} years")
            
            # Add education bonus if present
            if 'education_bonus' in experience_dict and experience_dict['education_bonus'] > 0:
                summary_parts.append(f"Education Bonus: {experience_dict['education_bonus']:.1f} years")
            
            # Add date ranges if present
            if 'date_ranges' in experience_dict and experience_dict['date_ranges']:
                dates = sorted(experience_dict['date_ranges'])
                if len(dates) >= 2:
                    summary_parts.append(f"Career Span: {dates[0]} - {dates[-1]}")
            
            # Add experience matches if present
            if 'matches' in experience_dict and experience_dict['matches']:
                for match in experience_dict['matches']:
                    if isinstance(match, dict):
                        text = match.get('text', '')
                        context = match.get('context', '')
                        if text and context:
                            summary_parts.append(f"Experience: {text} ({context})")
            
            return "\n".join(summary_parts) if summary_parts else "No detailed experience information available"
        
        except Exception as e:
            logger.error(f"Error formatting experience dictionary: {str(e)}")
            return "Error processing experience information"

    def _clean_json_string(self, json_str: str) -> str:
        """Clean and normalize JSON string."""
        try:
            # Handle code blocks first
            if '```' in json_str:
                # Extract content between code blocks
                matches = re.findall(r'```(?:json)?(.*?)```', json_str, re.DOTALL)
                if matches:
                    json_str = matches[0]

            # Basic cleanup
            json_str = json_str.strip()
            
            # Remove newlines and excess whitespace
            json_str = ' '.join(line.strip() for line in json_str.splitlines())
            
            # Handle escaped characters
            json_str = json_str.replace('\\"', '"')  # Unescape quotes
            json_str = json_str.replace('\\\\', '\\')  # Unescape backslashes
            json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_str)  # Remove invalid escapes
            
            # Fix common JSON formatting issues
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
            json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)  # Quote unquoted keys
            json_str = re.sub(r':\s*undefined\b', ': null', json_str)  # Replace undefined with null
            json_str = re.sub(r':\s*(true|false|null)\b', lambda m: f': {m.group(1).lower()}', json_str)  # Normalize boolean and null
            
            # Normalize recommendation values
            json_str = re.sub(r'"recommendation"\s*:\s*"[^"]*"', lambda m: self._normalize_recommendation(m.group(0)), json_str)
            
            return json_str
            
        except Exception as e:
            logger.error(f"Error cleaning JSON string: {str(e)}")
            return json_str

    def _normalize_recommendation(self, match_str: str) -> str:
        """Normalize recommendation string to match MatchStrength enum values."""
        try:
            # Remove any JSON formatting
            clean_str = match_str.replace('"recommendation": ', '').replace('"', '').strip()
            
            # Map to valid enum values
            recommendation_map = {
                'NO_MATCH': 'NO_MATCH',
                'POTENTIAL_MATCH': 'POTENTIAL_MATCH',
                'GOOD_MATCH': 'GOOD_MATCH',
                'STRONG_MATCH': 'STRONG_MATCH'
            }
            
            return recommendation_map.get(clean_str, 'NO_MATCH')
        except Exception as e:
            logger.error(f"Error normalizing recommendation: {str(e)}")
            return 'NO_MATCH'

    def _convert_skills_to_list(self, skills_dict: Dict) -> List[Dict]:
        """Convert skills dictionary to list of SkillAssessment objects."""
        skills_list = []
        proficiency_map = {
            'Strong': 'Advanced',
            'Intermediate': 'Intermediate',
            'Basic': 'Beginner',
            'Not specified': 'Beginner'
        }
        
        for skill, level in skills_dict.items():
            skills_list.append({
                'skill': skill,
                'proficiency': proficiency_map.get(level, 'Intermediate'),
                'years': 0.0  # Default value as we don't have this information
            })
        return skills_list

    def _convert_gaps_to_list(self, gaps: Union[Dict, List]) -> List[str]:
        """Convert technical gaps to list format."""
        if isinstance(gaps, dict):
            return [f"{k}: {v}" for k, v in gaps.items()]
        return gaps if isinstance(gaps, list) else []

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse and validate the LLM response."""
        try:
            # First try to parse the entire content as JSON
            try:
                result = json.loads(content)
                if isinstance(result, dict):
                    result = self._validate_and_complete_json(result)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                extracted = self._extract_json_from_text(content)
                if isinstance(extracted, dict):
                    result = self._validate_and_complete_json(extracted)
                elif isinstance(extracted, str):
                    try:
                        result = json.loads(extracted)
                        result = self._validate_and_complete_json(result)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse extracted JSON string")
                        return self._get_fallback_json()
                else:
                    logger.error("No valid JSON found in response")
                    return self._get_fallback_json()

            # Convert skills assessment to list format with required fields
            skills_dict = result.get('skills_assessment', [])
            skills_list = []
            
            if isinstance(skills_dict, dict):
                # Handle dictionary format
                for skill_name, proficiency in skills_dict.items():
                    # Map proficiency levels to valid values
                    proficiency_map = {
                        'Expert': 'Advanced',
                        'Strong': 'Advanced',
                        'Advanced': 'Advanced',
                        'Intermediate': 'Intermediate',
                        'Basic': 'Beginner',
                        'Beginner': 'Beginner',
                        'Not specified': 'Beginner'
                    }
                    
                    # Extract years if present in proficiency string
                    years = 0.0
                    if isinstance(proficiency, str):
                        # Look for years in the proficiency string
                        year_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', proficiency.lower())
                        if year_match:
                            try:
                                years = float(year_match.group(1))
                            except (ValueError, AttributeError):
                                years = 0.0
                    elif isinstance(proficiency, dict):
                        # Extract years from dictionary format
                        years = float(proficiency.get('years', 0))
                        proficiency = proficiency.get('proficiency', 'Not specified')
                    
                    # Map the proficiency level
                    mapped_proficiency = proficiency_map.get(
                        proficiency if isinstance(proficiency, str) else 'Not specified',
                        'Intermediate'
                    )
                    
                    skills_list.append({
                        "skill": skill_name,
                        "proficiency": mapped_proficiency,
                        "years": years
                    })
            elif isinstance(skills_dict, list):
                # Handle list format
                for skill_item in skills_dict:
                    if isinstance(skill_item, dict):
                        # Extract skill name from either 'skill' or 'skill_name' field
                        skill_name = skill_item.get('skill') or skill_item.get('skill_name', '')
                        
                        # Map proficiency to valid values
                        raw_proficiency = skill_item.get('proficiency', 'Intermediate')
                        proficiency_map = {
                            'Expert': 'Advanced',
                            'Strong': 'Advanced',
                            'Advanced': 'Advanced',
                            'Intermediate': 'Intermediate',
                            'Basic': 'Beginner',
                            'Beginner': 'Beginner',
                            'Not specified': 'Beginner'
                        }
                        mapped_proficiency = proficiency_map.get(raw_proficiency, 'Intermediate')
                        
                        # Get years, defaulting to 0
                        years = 0.0
                        raw_years = skill_item.get('years')
                        if raw_years is not None:
                            try:
                                if isinstance(raw_years, (int, float)):
                                    years = float(raw_years)
                                elif isinstance(raw_years, str):
                                    # Try to extract years from string format
                                    year_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', raw_years.lower())
                                    if year_match:
                                        years = float(year_match.group(1))
                                    else:
                                        # Try direct conversion if no year pattern found
                                        years = float(raw_years)
                            except (ValueError, AttributeError):
                                years = 0.0
                        
                        if skill_name:  # Only add if we have a valid skill name
                            skills_list.append({
                                "skill": skill_name,
                                "proficiency": mapped_proficiency,
                                "years": years,
                                "context": skill_item.get('context', ''),
                                "confidence": float(skill_item.get('confidence', 1.0)),
                                "last_used": skill_item.get('last_used', 'Recent')
                            })
                    elif isinstance(skill_item, str):
                        # Handle simple string format
                        skills_list.append({
                            "skill": skill_item,
                            "proficiency": "Intermediate",
                            "years": 0.0,
                            "context": "",
                            "confidence": 1.0,
                            "last_used": "Recent"
                        })
            
            result['skills_assessment'] = skills_list

            # Convert technical gaps to list format
            gaps = result.get('technical_gaps', [])
            if isinstance(gaps, dict):
                gaps_list = []
                for gap, status in gaps.items():
                    gaps_list.append(f"{gap}: {status}")
                result['technical_gaps'] = gaps_list
            elif not isinstance(gaps, list):
                result['technical_gaps'] = []

            # Normalize recommendation value
            recommendation = result.get('recommendation', 'NO_MATCH')
            valid_recommendations = {'STRONG_MATCH', 'GOOD_MATCH', 'POTENTIAL_MATCH', 'NO_MATCH'}
            if recommendation not in valid_recommendations:
                if recommendation == 'LOW_MATCH':
                    recommendation = 'NO_MATCH'
                elif 'STRONG' in recommendation.upper():
                    recommendation = 'STRONG_MATCH'
                elif 'GOOD' in recommendation.upper():
                    recommendation = 'GOOD_MATCH'
                elif 'POTENTIAL' in recommendation.upper():
                    recommendation = 'POTENTIAL_MATCH'
                else:
                    recommendation = 'NO_MATCH'
            result['recommendation'] = recommendation

            # Ensure all required fields are present
            required_fields = {
                'technical_match_score': 0.0,
                'recommendation': 'NO_MATCH',
                'skills_assessment': [],
                'technical_gaps': [],
                'interview_questions': [],
                'key_findings': [],
                'concerns': []
            }
            
            for field, default_value in required_fields.items():
                if field not in result or not result[field]:
                    result[field] = default_value
                elif isinstance(result[field], (list, tuple)):
                    result[field] = list(result[field])  # Ensure it's a list

            return result

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return self._get_fallback_json()

    def _get_fallback_response(self) -> Dict[str, Any]:
        """Provide a default response when analysis fails."""
        return {
            "technical_match_score": 0.0,
            "recommendation": "NO_MATCH",
            "skills_assessment": [],
            "technical_gaps": [],
            "interview_questions": [],
            "key_findings": [],
            "concerns": [],
            "experience_details": None
        }

    def _get_fallback_experience_response(self) -> Dict[str, Any]:
        """Provide a default response when experience analysis fails."""
        return {
            "us_experience_years": 0.0,
            "non_us_experience_years": 0.0,
            "total_professional_years": 0.0,
            "internship_count": 0,
            "experience_breakdown": [],
            "experience_strength": "UNKNOWN",
            "experience_flags": ["Experience analysis failed"]
        }

    def _validate_and_complete_json(self, parsed: Dict) -> Dict:
        """Validate and complete JSON with required fields."""
        required_fields = {
            'technical_match_score': 0.0,
            'recommendation': 'NO_MATCH',
            'skills_assessment': [],
            'technical_gaps': [],
            'interview_questions': [],
            'key_findings': [],
            'concerns': [],
            'experience_details': None
        }
        
        # Add missing required fields
        for field, default_value in required_fields.items():
            if field not in parsed:
                parsed[field] = default_value
                
        return parsed

    def _get_fallback_json(self) -> Dict[str, Any]:
        """Provide a default JSON structure when parsing fails."""
        return {
            "technical_match_score": 0.0,
            "recommendation": "NO_MATCH",
            "skills_assessment": [],
            "technical_gaps": ["Unable to parse response"],
            "interview_questions": [],
            "key_findings": ["JSON parsing failed"],
            "concerns": ["Unable to process response format"],
            "experience_details": self._get_fallback_experience_response()
        }

    def _extract_json_from_text(self, text: str) -> Union[str, Dict]:
        """Extract and validate JSON from LLM response text."""
        try:
            # Try to parse the entire text as JSON first
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Find the outermost JSON content
            stack = []
            start = -1
            result = None
            
            for i, char in enumerate(text):
                if char == '{':
                    if not stack:
                        start = i
                    stack.append(char)
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack and start != -1:
                            try:
                                json_str = text[start:i+1]
                                result = json.loads(json_str)
                                break
                            except json.JSONDecodeError:
                                continue

            if result:
                return self._validate_and_complete_json(result)
            
            logger.error("No valid JSON content found in response")
            return self._get_fallback_json()
            
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return self._get_fallback_json()

    def _calculate_technical_score(self, matched_skills: Dict, role_name: str) -> int:
        """Calculate technical match score based on matched skills and role-specific constraints."""
        try:
            with open('config/jobs.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            role_config = config['job_roles'].get(role_name)
            if not role_config:
                logger.error(f"Role configuration not found for: {role_name}")
                return 0
            
            scoring_constraints = role_config.get('scoring_constraints', {})
            max_score = scoring_constraints.get('max_score', 100)
            required_skills_threshold = scoring_constraints.get('required_skills_threshold', 0.7)
            
            # Calculate required skills match
            total_required = len(role_config['required_skills'])
            matched_required = len(matched_skills.get('required', []))
            required_ratio = matched_required / total_required if total_required > 0 else 0
            
            # Calculate preferred skills match
            total_preferred = len(role_config['preferred_skills'])
            matched_preferred = len(matched_skills.get('preferred', []))
            preferred_ratio = matched_preferred / total_preferred if total_preferred > 0 else 0
            
            # Calculate weighted score
            skill_weights = config['scoring_config']['skill_weights']
            required_weight = skill_weights['required']
            preferred_weight = skill_weights['preferred']
            
            raw_score = (
                (required_ratio * required_weight + preferred_ratio * preferred_weight)
                / (required_weight + preferred_weight)
                * max_score
            )
            
            # Apply threshold adjustments
            if required_ratio < required_skills_threshold:
                raw_score *= 0.8
            
            # Round to nearest integer and ensure within bounds
            final_score = min(max_score, max(0, round(raw_score)))
            return final_score
            
        except Exception as e:
            logger.error(f"Technical score calculation failed: {str(e)}")
            return 0
    
    def _mixtral_technical_analysis(self, resume_text: str, role_name: str, matched_skills: Dict, extracted_experience: List[str], technical_score: int) -> Dict[str, Any]:
            """Technical skills analysis using Mixtral"""
            try:
                # Calculate recommendation based on technical score
                recommendation = "NO_MATCH"
                if technical_score >= 85:
                    recommendation = "STRONG_MATCH"
                elif technical_score >= 70:
                    recommendation = "GOOD_MATCH"
                elif technical_score >= 50:
                    recommendation = "POTENTIAL_MATCH"
                
                system_prompt = (
                    "You are an expert technical recruiter analyzing a resume. "
                    "Provide a detailed technical assessment in JSON format. "
                    f"The technical score is {technical_score}/100, resulting in a recommendation of {recommendation}."
                )
                
                user_prompt = (
                    f"Analyze this candidate's technical qualifications for {role_name} role:\n\n"
                    f"Resume Text: {resume_text}\n\n"
                    "Return a JSON object with these fields:\n"
                    "{\n"
                    f'    "technical_match_score": {technical_score},\n'
                    f'    "recommendation": "{recommendation}",\n'
                    '    "skills_assessment": [...],\n'
                    '    "technical_gaps": [...],\n'
                    '    "interview_questions": [...],\n'
                    '    "key_findings": [...],\n'
                    '    "concerns": [...]\n'
                    "}"
                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.execute_request(messages)
                if not response or not response.choices:
                    return self._get_fallback_response()
                
                content = response.choices[0].message.content.strip()
                return self._parse_response(content)
                
            except Exception as e:
                logger.error(f"Technical analysis failed: {str(e)}")
                return self._get_fallback_response()

    def _gemini_experience_analysis(self, resume_text: str, role_name: str) -> Dict[str, Any]:
        """Analyze and classify work experience using Gemini"""
        try:
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.1,
                "top_k": 16,
                "max_output_tokens": 2048,
            }

            safety_settings = []  # Disabled for retry attempts
            
            prompt = (
                f"Analyze the work experience in this resume for a {role_name} role.\n\n"
                f"Resume Text: {resume_text}\n\n"
                "Return a JSON object with EXACTLY these fields:\n"
                "{\n"
                '    "us_experience_years": float,\n'
                '    "non_us_experience_years": float,\n'
                '    "total_professional_years": float,\n'
                '    "internship_count": int,\n'
                '    "experience_breakdown": [string],\n'
                '    "experience_strength": "STRONG" | "MODERATE" | "LIMITED",\n'
                '    "experience_flags": [string]\n'
                "}"
            )

            response = self.gemini_client.generate_content(
                contents=prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            if response and response.text:
                raw_text = response.text.strip()
                return self._parse_gemini_response(raw_text)

            return self._get_fallback_experience_response()

        except Exception as e:
            logger.error(f"Experience analysis failed: {str(e)}")
            return self._get_fallback_experience_response()

    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the response from Gemini model into a structured format."""
        try:
            logging.info("[LLM] Starting to parse Gemini response")
            # First try to extract JSON using regex
            json_match = re.search(r'\{[\s\S]*\}', response_text)

            if json_match:
                try:
                    result = json.loads(json_match.group())
                    logging.info("[LLM] Successfully extracted JSON from response")
                except json.JSONDecodeError as e:
                    logging.error(f"[LLM] JSON decode error: {e}")
                    return self._get_fallback_json()
            else:
                logging.warning("[LLM] No JSON found in response, using fallback")
                return self._get_fallback_json()

            # Log the structure of the result before processing
            logging.info(f"[LLM] Raw result structure: {type(result)}")
            logging.info(f"[LLM] Skills assessment type: {type(result.get('skills_assessment'))}")
            
            # Convert skills assessment to list format with required fields
            skills_raw = result.get('skills_assessment', [])
            logging.info(f"[LLM] Raw skills data: {skills_raw}")
            
            skills_list = []
            
            if isinstance(skills_raw, dict):
                logging.info("[LLM] Processing dictionary format skills")
                for skill_name, proficiency in skills_raw.items():
                    logging.debug(f"[LLM] Processing skill: {skill_name} with proficiency: {proficiency}")
                    # Map proficiency levels to valid values
                    proficiency_map = {
                        'Expert': 'Advanced',
                        'Strong': 'Advanced',
                        'Advanced': 'Advanced',
                        'Intermediate': 'Intermediate',
                        'Basic': 'Beginner',
                        'Beginner': 'Beginner',
                        'Not specified': 'Beginner'
                    }
                    
                    # Extract years if present in proficiency string
                    years = 0.0
                    if isinstance(proficiency, str):
                        # Look for years in the proficiency string
                        year_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', proficiency.lower())
                        if year_match:
                            try:
                                years = float(year_match.group(1))
                            except (ValueError, AttributeError):
                                years = 0.0
                    elif isinstance(proficiency, dict):
                        # Extract years from dictionary format
                        years = float(proficiency.get('years', 0))
                        proficiency = proficiency.get('proficiency', 'Not specified')
                    
                    # Map the proficiency level
                    mapped_proficiency = proficiency_map.get(
                        proficiency if isinstance(proficiency, str) else 'Not specified',
                        'Intermediate'
                    )
                    
                    skills_list.append({
                        "skill": skill_name,
                        "proficiency": mapped_proficiency,
                        "years": years
                    })
            elif isinstance(skills_raw, list):
                logging.info("[LLM] Processing list format skills")
                for skill_item in skills_raw:
                    logging.debug(f"[LLM] Processing skill item: {type(skill_item)} - {skill_item}")
                    if isinstance(skill_item, dict):
                        # Extract skill name from either 'skill' or 'skill_name' field
                        skill_name = skill_item.get('skill') or skill_item.get('skill_name', '')
                        
                        # Map proficiency to valid values
                        raw_proficiency = skill_item.get('proficiency', 'Intermediate')
                        proficiency_map = {
                            'Expert': 'Advanced',
                            'Strong': 'Advanced',
                            'Advanced': 'Advanced',
                            'Intermediate': 'Intermediate',
                            'Basic': 'Beginner',
                            'Beginner': 'Beginner',
                            'Not specified': 'Beginner'
                        }
                        mapped_proficiency = proficiency_map.get(raw_proficiency, 'Intermediate')
                        
                        # Get years, defaulting to 0
                        years = 0.0
                        raw_years = skill_item.get('years')
                        if raw_years is not None:
                            try:
                                if isinstance(raw_years, (int, float)):
                                    years = float(raw_years)
                                elif isinstance(raw_years, str):
                                    # Try to extract years from string format
                                    year_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', raw_years.lower())
                                    if year_match:
                                        years = float(year_match.group(1))
                                    else:
                                        # Try direct conversion if no year pattern found
                                        years = float(raw_years)
                            except (ValueError, AttributeError):
                                years = 0.0
                        
                        if skill_name:  # Only add if we have a valid skill name
                            skills_list.append({
                                "skill": skill_name,
                                "proficiency": mapped_proficiency,
                                "years": years,
                                "context": skill_item.get('context', ''),
                                "confidence": float(skill_item.get('confidence', 1.0)),
                                "last_used": skill_item.get('last_used', 'Recent')
                            })
                    elif isinstance(skill_item, str):
                        # Handle simple string format
                        skills_list.append({
                            "skill": skill_item,
                            "proficiency": "Intermediate",
                            "years": 0.0,
                            "context": "",
                            "confidence": 1.0,
                            "last_used": "Recent"
                        })

            # Log the final processed skills
            logging.info(f"[LLM] Processed {len(skills_list)} skills")
            result['skills_assessment'] = skills_list
            
            # Convert technical gaps to list format
            gaps = result.get('technical_gaps', [])
            if isinstance(gaps, dict):
                gaps_list = []
                for gap, status in gaps.items():
                    gaps_list.append(f"{gap}: {status}")
                result['technical_gaps'] = gaps_list
            elif not isinstance(gaps, list):
                result['technical_gaps'] = []
            
            # Normalize recommendation value
            recommendation = result.get('recommendation', 'NO_MATCH')
            valid_recommendations = {'STRONG_MATCH', 'GOOD_MATCH', 'POTENTIAL_MATCH', 'NO_MATCH'}
            if recommendation not in valid_recommendations:
                if recommendation == 'LOW_MATCH':
                    recommendation = 'NO_MATCH'
                elif 'STRONG' in recommendation.upper():
                    recommendation = 'STRONG_MATCH'
                elif 'GOOD' in recommendation.upper():
                    recommendation = 'GOOD_MATCH'
                elif 'POTENTIAL' in recommendation.upper():
                    recommendation = 'POTENTIAL_MATCH'
                else:
                    recommendation = 'NO_MATCH'
            result['recommendation'] = recommendation
            
            # Ensure all required fields are present
            required_fields = {
                'technical_match_score': 0.0,
                'recommendation': 'NO_MATCH',
                'skills_assessment': [],
                'technical_gaps': [],
                'interview_questions': [],
                'key_findings': [],
                'concerns': []
            }
            
            for field, default_value in required_fields.items():
                if field not in result or not result[field]:
                    result[field] = default_value
                elif isinstance(result[field], (list, tuple)):
                    result[field] = list(result[field])  # Ensure it's a list
            
            return result

        except Exception as e:
            logging.error(f"[LLM] Error parsing response: {str(e)}", exc_info=True)
            return self._get_fallback_json()