from groq import Groq
import google.generativeai as genai
from typing import Dict, List, Any, Optional, Union
import os
from dotenv import load_dotenv
import logging
import json
import time
import re
import yaml
import traceback
from utils.logging_config import setup_logging
from utils.prompt_handler import PromptHandler
from utils.schemas import AnalysisResult, MatchStrength, ExperienceDetails, InterviewQuestion

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
            # Initialize API keys
            self.groq_api_key = os.getenv('GROQ_API_KEY')
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            
            # Initialize model names
            self.mixtral_model = "mixtral-8x7b-32768"
            self.gemini_model = "gemini-pro"
            
            # Initialize clients
            self.groq_client = None
            self.gemini_client = None
            
            # Load config at initialization
            with open('config/jobs.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Initialize prompt handler
            self.prompt_handler = PromptHandler()
            
            # Validate API keys
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
                
            if not self.gemini_api_key:
                logger.warning("GEMINI_API_KEY not found - Gemini features disabled")
            
            # Initialize clients
            self.initialize_clients()
            
        except Exception as e:
            logger.error(f"LLMAnalyzer initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize LLM services: {str(e)}") from e

    def initialize_clients(self):
        """Initialize LLM clients with API keys."""
        try:
            # Initialize Groq client
            if self.groq_api_key:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("Successfully initialized Groq client")
            
            # Initialize Gemini client
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(self.gemini_model)
                logger.info("Successfully initialized Gemini client")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {str(e)}")
            logger.debug(f"Client initialization error details: {traceback.format_exc()}")
            raise RuntimeError("Failed to initialize one or more LLM clients") from e

    def analyze_resume(self, resume_text: str, role: str, matched_skills: Dict, experience_details: Dict) -> AnalysisResult:
        """Analyze resume with enhanced question generation and strict role validation"""
        try:
            # STRICT ROLE VALIDATION
            if not role or not isinstance(role, str):
                logger.error("Invalid role parameter: role must be a non-empty string")
                return self._create_fallback_response()

            # Force role to standard format
            role = role.strip().lower()
            
            # Get and validate role configuration with aggressive checking
            job_roles = self.config.get('job_roles', {})
            if not job_roles:
                logger.error("No job roles found in configuration")
                return self._create_fallback_response()
                
            # Case-insensitive role lookup
            role_config = None
            for config_role_name, config in job_roles.items():
                if config_role_name.lower() == role.lower():
                    role_config = config
                    break
                    
            if not role_config:
                available_roles = list(job_roles.keys())
                logger.error(f"Role '{role}' not found in configuration. Available roles: {available_roles}")
                return self._create_fallback_response()
                
            # Validate required fields in role config
            required_fields = ['required_skills', 'preferred_skills', 'min_years_experience']
            missing_fields = [field for field in required_fields if field not in role_config]
            if missing_fields:
                logger.error(f"Role configuration missing required fields: {missing_fields}")
                return self._create_fallback_response()
            
            # Validate skills lists
            if not isinstance(role_config['required_skills'], list) or not role_config['required_skills']:
                logger.error("Role configuration must include non-empty required_skills list")
                return self._create_fallback_response()
            
            # Extract tech contexts with validation
            tech_contexts = self._extract_tech_contexts(resume_text)
            if tech_contexts is None:  # Explicit None check
                logger.error("Failed to extract tech contexts")
                return self._create_fallback_response()
            
            # Generate contextual questions with validation
            questions = self._generate_contextual_questions(
                resume_text,
                tech_contexts,
                matched_skills,
                experience_details
            )
            if questions is None:  # Explicit None check
                logger.error("Failed to generate contextual questions")
                return self._create_fallback_response()
            
            # Analyze skills with Mixtral
            skill_analysis = self._analyze_skills_with_mixtral(resume_text, role, role_config, matched_skills)
            if not skill_analysis:
                logger.error("Failed to analyze skills")
                return self._create_fallback_response()
            
            # Calculate technical score
            technical_score = self._calculate_technical_score_from_llm(skill_analysis, role_config, matched_skills)
            
            # Analyze experience with Gemini
            experience_analysis = self._analyze_experience_with_gemini(resume_text, role, skill_analysis)
            
            # Adjust score based on experience
            if experience_analysis:
                technical_score = self._adjust_score_with_experience(technical_score, experience_analysis, role_config)
            
            # Determine final recommendation
            recommendation = self._determine_recommendation(technical_score, skill_analysis, experience_analysis)
            
            # Create final result
            result = AnalysisResult(
                technical_match_score=technical_score,
                recommendation=recommendation,
                skills_assessment=skill_analysis.get('skills_assessment', []),
                technical_gaps=skill_analysis.get('technical_gaps', []),
                interview_questions=questions if questions else skill_analysis.get('interview_questions', []),
                key_findings=skill_analysis.get('key_findings', []),
                concerns=skill_analysis.get('concerns', []),
                experience_details=experience_analysis,
                confidence_score=skill_analysis.get('confidence_score', 0.0)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Resume analysis failed: {str(e)}")
            logger.debug(f"Analysis error details: {traceback.format_exc()}")
            return self._create_fallback_response()

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
        """Execute a request to Groq with enhanced retry logic and improved JSON handling."""
        if not self.groq_client:
            logger.info("Client not initialized, initializing now...")
            self.initialize_clients()
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Making API request (attempt {attempt + 1}/{max_retries})")
                
                # Add explicit JSON formatting instruction to system message
                messages[0]["content"] = messages[0]["content"].strip() + "\nIMPORTANT: Return only valid JSON without any additional text or formatting."
                
                response = self.groq_client.chat.completions.create(
                    messages=messages,
                    model=self.mixtral_model,
                    temperature=0.01,
                    max_tokens=4000,
                    top_p=0.1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    response_format={"type": "json_object"}
                )
                logger.info("API request successful")
                logger.debug(f"Raw response type: {type(response)}")
                logger.debug(f"Raw response: {response}")

                # Extract content from response
                content = None
                
                # Case 1: Standard Groq response object
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content
                    logger.debug("Extracted content from standard Groq response")
                
                # Case 2: Dictionary response with choices
                elif isinstance(response, dict):
                    if 'error' in response and 'failed_generation' in response['error']:
                        content = response['error']['failed_generation']
                        logger.debug("Extracted content from error failed_generation")
                    elif 'choices' in response and response['choices']:
                        content = response['choices'][0]['message']['content']
                        logger.debug("Extracted content from dictionary choices")
                    else:
                        content = json.dumps(response)
                        logger.debug("Using raw dictionary response")
                
                # Case 3: String response
                elif isinstance(response, str):
                    content = response
                    logger.debug("Using raw string response")
                
                logger.debug(f"Extracted content type: {type(content)}")
                logger.debug(f"Extracted content: {content[:200]}...")  # Log first 200 chars

                # Process the content
                if content:
                    # If content is already valid JSON string
                    try:
                        if isinstance(content, str):
                            json.loads(content)
                            logger.debug("Content is valid JSON string")
                            return {"choices": [{"message": {"content": content}}]}
                    except json.JSONDecodeError:
                        pass

                    # Clean and validate JSON response
                    cleaned_json = self._extract_json_from_text(content)
                    if cleaned_json:
                        logger.debug("Successfully cleaned and validated JSON")
                        return {"choices": [{"message": {"content": cleaned_json}}]}
                
                logger.warning("Response validation failed, will retry with modified prompt")
                continue
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Request error: {error_msg}")
                
                if "json_validate_failed" in error_msg.lower():
                    try:
                        # Try to extract JSON from the error message
                        error_dict = getattr(e, 'args', [{}])[0] if hasattr(e, 'args') else {}
                        if isinstance(error_dict, dict) and 'error' in error_dict:
                            failed_json = error_dict['error'].get('failed_generation', '')
                        else:
                            match = re.search(r"failed_generation':\s*'([^']+)'", error_msg)
                            failed_json = match.group(1) if match else ''
                        
                        if failed_json:
                            logger.debug(f"Found failed_generation JSON: {failed_json[:200]}...")
                            cleaned_json = self._extract_json_from_text(failed_json)
                            if cleaned_json:
                                logger.info("Successfully extracted JSON from error message")
                                return {"choices": [{"message": {"content": cleaned_json}}]}
                    except Exception as je:
                        logger.error(f"Failed to extract JSON from error message: {str(je)}")
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("All retry attempts failed")
                    raise

        # If we get here, all retries failed
        logger.error("Failed to get valid response after all retries")
        return {"choices": [{"message": {"content": self._create_fallback_json()}}]}

    def _clean_text(self, text: str) -> str:
        """Clean text to handle Unicode characters and special formatting."""
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

    def _analyze_skills_with_mixtral(self, resume_text: str, role: str, role_config: Dict, matched_skills: Dict) -> Dict:
        """Enhanced skill analysis using Mixtral."""
        try:
            system_prompt = """You are an expert technical recruiter analyzing a resume. 
            You must return a valid JSON object with the following structure, and ONLY this structure:
            {
                "technical_match_score": number between 0-100,
                "recommendation": "STRONG_MATCH" or "GOOD_MATCH" or "POTENTIAL_MATCH" or "NO_MATCH",
                "skills_assessment": [
                    {
                        "skill": "skill name",
                        "proficiency": "Expert/Advanced/Intermediate/Beginner",
                        "years": number,
                        "context": "evidence from resume"
                    }
                ],
                "technical_gaps": ["list of missing or weak skills"],
                "key_findings": ["list of important observations"],
                "interview_questions": ["list of relevant technical questions"],
                "concerns": ["list of potential issues"],
                "confidence_score": number between 0-1
            }
            IMPORTANT: 
            1. Return ONLY the JSON object, no other text
            2. Ensure all string values are properly quoted
            3. Use numbers without quotes for numeric values
            4. Do not add any additional fields
            5. Do not include any explanatory text or markdown"""

            user_prompt = f"""Analyze this candidate for {role} role:

Resume Text:
{resume_text}

Required Skills: {', '.join(role_config['required_skills'])}
Preferred Skills: {', '.join(role_config.get('preferred_skills', []))}

Already Matched Skills:
Required: {', '.join(matched_skills.get('required', []))}
Preferred: {', '.join(matched_skills.get('preferred', []))}

Focus on:
1. Skill proficiency levels based on context and achievements
2. Years of experience with each skill
3. Technical depth and complexity of projects
4. Impact and innovations
5. Any gaps or areas for development

Return ONLY a JSON object with the exact structure specified above."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.execute_request(messages)
            logger.debug(f"Mixtral response type: {type(response)}")
            logger.debug(f"Mixtral raw response: {response}")
            
            # Extract content from response
            if isinstance(response, dict):
                if 'choices' in response and response['choices']:
                    content = response['choices'][0]['message']['content']
                    logger.debug(f"Extracted content from choices: {content[:200]}...")
                elif 'error' in response and 'failed_generation' in response['error']:
                    content = response['error']['failed_generation']
                    logger.debug(f"Extracted content from error: {content[:200]}...")
                else:
                    content = json.dumps(response)
                    logger.debug(f"Using raw response: {content[:200]}...")
            else:
                content = str(response)
                logger.debug(f"Converted response to string: {content[:200]}...")

            # Parse and validate the response
            try:
                if isinstance(content, str):
                    # Try to parse as JSON first
                    try:
                        parsed = json.loads(content)
                        logger.debug("Successfully parsed content as JSON")
                    except json.JSONDecodeError:
                        # If direct parsing fails, try to extract JSON
                        logger.debug("Direct JSON parsing failed, attempting extraction")
                        cleaned_json = self._extract_json_from_text(content)
                        if not cleaned_json:
                            logger.error("Failed to extract valid JSON from response")
                            return self._create_fallback_json()
                        parsed = json.loads(cleaned_json)
                    
                # Validate required fields
                required_fields = [
                    'technical_match_score',
                    'recommendation',
                    'skills_assessment',
                    'technical_gaps',
                    'interview_questions',
                    'key_findings',
                    'concerns',
                    'confidence_score'
                ]

                # Convert parsed to dict if it's a string
                if isinstance(parsed, str):
                    try:
                        parsed = json.loads(parsed)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse string result to JSON")
                        return self._create_fallback_json()

                # Ensure all required fields exist
                for field in required_fields:
                    if field not in parsed:
                        logger.warning(f"Missing required field: {field}")
                        if field == 'technical_match_score':
                            parsed[field] = 0
                        elif field == 'recommendation':
                            parsed[field] = "NO_MATCH"
                        elif field == 'confidence_score':
                            parsed[field] = 0.0
                        else:
                            parsed[field] = []

                logger.info("Successfully parsed and validated skill analysis response")
                return parsed

            except Exception as e:
                logger.error(f"Failed to parse skill analysis response: {str(e)}")
                logger.debug(f"Problematic content: {content[:200]}...")
                return self._create_fallback_json()

        except Exception as e:
            logger.error(f"Skill analysis failed: {str(e)}")
            return self._create_fallback_json()

    def _calculate_technical_score_from_llm(self, skill_analysis: Dict, role_config: Dict, matched_skills: Dict) -> int:
        """Calculate technical score with enhanced LLM insights and balanced scoring."""
        try:
            base_score = 0
            total_weight = 0
            
            # Weight required skills more heavily
            required_skills = set(role_config['required_skills'])
            preferred_skills = set(role_config.get('preferred_skills', []))
            
            logger.info("Starting technical score calculation:")
            logger.info(f"Required skills: {required_skills}")
            logger.info(f"Preferred skills: {preferred_skills}")
            
            # Track missing required skills for penalty
            missing_required = required_skills.copy()
            
            # Track skill levels for scaling
            expert_count = 0
            advanced_count = 0
            
            for skill_assessment in skill_analysis.get('skills_assessment', []):
                skill_name = skill_assessment['skill']
                proficiency = skill_assessment['proficiency']
                
                # Remove from missing required if found
                if skill_name in missing_required:
                    missing_required.remove(skill_name)
                
                # Track proficiency levels
                if proficiency == 'Expert':
                    expert_count += 1
                elif proficiency == 'Advanced':
                    advanced_count += 1
                
                # Determine skill weight with scaled weights
                weight = 0.0
                if skill_name in required_skills:
                    weight = 3.0
                    logger.info(f"Required skill found: {skill_name} (weight: 3.0)")
                elif skill_name in preferred_skills:
                    weight = 1.0  # Increased back to 1.0 from 0.8
                    logger.info(f"Preferred skill found: {skill_name} (weight: 1.0)")
                else:
                    logger.debug(f"Skill not in required or preferred list: {skill_name}")
                    continue
                
                # Calculate proficiency score with balanced values
                proficiency_scores = {
                    'Expert': 1.0,     # Increased from 0.9
                    'Advanced': 0.8,   # Increased from 0.7
                    'Intermediate': 0.5,  # Increased from 0.4
                    'Beginner': 0.2    # Increased from 0.15
                }
                proficiency_score = proficiency_scores.get(proficiency, 0.2)
                logger.info(f"  - Proficiency: {proficiency} (score: {proficiency_score})")
                
                # Add to weighted score
                skill_score = weight * proficiency_score
                base_score += skill_score
                total_weight += weight
                logger.info(f"  - Skill score contribution: {skill_score:.2f}")
            
            # Calculate missing skills penalty
            missing_count = len(missing_required)
            if missing_count > 0:
                # Progressive penalty: 25% for first, +25% for each additional
                missing_penalty = 0.25 + (missing_count - 1) * 0.25  # Reduced from 0.35
            else:
                missing_penalty = 0.0
            
            logger.info(f"Base score before normalization: {base_score:.2f}")
            logger.info(f"Total weight: {total_weight:.2f}")
            logger.info(f"Missing required skills penalty: -{missing_penalty*100:.1f}%")
            
            # Enhanced expertise bonus values
            expertise_bonus = 0.0
            if expert_count >= 3 and advanced_count >= 1:  # Strong expertise
                expertise_bonus = 8.0  # Increased from 3.0
            elif expert_count >= 2 or (expert_count >= 1 and advanced_count >= 2):
                expertise_bonus = 5.0  # Increased from 2.0
            elif expert_count >= 1 or advanced_count >= 2:
                expertise_bonus = 2.0  # Increased from 0.5
            
            logger.info(f"Expertise bonus: +{expertise_bonus:.1f}")
            
            # Normalize score to 0-100 range and apply penalties/bonuses
            normalized_score = (base_score / total_weight * 100) if total_weight > 0 else 0
            normalized_score = max(0, normalized_score * (1 - missing_penalty) + expertise_bonus)
            logger.info(f"Normalized score (0-100): {normalized_score:.2f}")
            
            # Apply skill group bonuses with balanced impact
            skill_groups = role_config.get('skill_groups', {})
            logger.info("Calculating skill group bonuses...")
            group_bonus = self._calculate_skill_group_bonus(skill_analysis, skill_groups) * 0.4  # Increased from 0.3
            logger.info(f"Skill group bonus multiplier: {1 + group_bonus:.2f}x")
            
            final_score = min(100, normalized_score * (1 + group_bonus))
            logger.info(f"Final technical score: {final_score:.2f}")
            
            return round(final_score)
            
        except Exception as e:
            logger.error(f"Technical score calculation failed: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            return 0

    def _analyze_experience_with_gemini(self, resume_text: str, role: str, skill_analysis: Dict) -> Optional[ExperienceDetails]:
        """Enhanced experience analysis using Gemini."""
        try:
            prompt = f"""Analyze the candidate's experience for {role} position:

Resume Text:
{resume_text}

Focus on:
1. Years of relevant experience in each area
2. Project complexity and scale
3. Leadership and ownership
4. Industry impact
5. Technical depth

Return a JSON object with the following structure:
{{
    "us_experience_years": float,
    "non_us_experience_years": float,
    "total_professional_years": float,
    "internship_count": int,
    "experience_breakdown": ["list of experience details"],
    "experience_strength": "STRONG" or "MODERATE" or "WEAK",
    "experience_flags": ["list of potential concerns"],
    "employment_gaps": [
        {{
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD"
        }}
    ]
}}

Provide specific examples and metrics where possible."""

            response = self.gemini_client.generate_content(prompt)
            
            if response and response.text:
                experience_data = self._parse_experience_response(response.text)
                return ExperienceDetails(**experience_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Experience analysis failed: {str(e)}")
            return None

    def _parse_experience_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate Gemini's experience analysis response."""
        try:
            # Clean and extract JSON from response
            cleaned_text = self._clean_text(response_text)
            json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
            
            if not json_match:
                logger.error("No JSON object found in response")
                return self._create_fallback_experience()
                
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse experience JSON: {str(e)}")
                return self._create_fallback_experience()
                
            # Convert numeric fields to float
            numeric_fields = ['us_experience_years', 'non_us_experience_years', 'total_professional_years']
            for field in numeric_fields:
                if field in data:
                    try:
                        data[field] = float(data[field])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {field}, defaulting to 0.0")
                        data[field] = 0.0
                else:
                    data[field] = 0.0
            
            # Ensure internship_count is integer
            try:
                data['internship_count'] = int(data.get('internship_count', 0))
            except (ValueError, TypeError):
                data['internship_count'] = 0
            
            # Convert experience breakdown items to strings
            if 'experience_breakdown' in data:
                processed_breakdown = []
                for item in data['experience_breakdown']:
                    if isinstance(item, dict):
                        # Convert dict to formatted string
                        title = item.get('title', '')
                        description = item.get('description', '')
                        processed_breakdown.append(f"{title}: {description}")
                    elif isinstance(item, str):
                        processed_breakdown.append(item)
                    else:
                        continue
                data['experience_breakdown'] = processed_breakdown
            else:
                data['experience_breakdown'] = []
            
            # Validate experience_strength
            valid_strengths = {'STRONG', 'MODERATE', 'WEAK'}
            if not isinstance(data.get('experience_strength'), str) or data.get('experience_strength') not in valid_strengths:
                data['experience_strength'] = 'MODERATE'
            
            # Ensure experience_flags is list of strings
            if not isinstance(data.get('experience_flags'), list):
                data['experience_flags'] = []
            data['experience_flags'] = [str(flag) for flag in data['experience_flags']]
            
            # Process employment gaps
            if 'employment_gaps' in data:
                processed_gaps = []
                for gap in data['employment_gaps']:
                    try:
                        processed_gaps.append({
                            'start_date': datetime.strptime(gap['start_date'], '%Y-%m-%d'),
                            'end_date': datetime.strptime(gap['end_date'], '%Y-%m-%d')
                        })
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Invalid employment gap format: {str(e)}")
                data['employment_gaps'] = processed_gaps
            else:
                data['employment_gaps'] = []
            
            return data
            
        except Exception as e:
            logger.error(f"Experience response parsing failed: {str(e)}")
            return self._create_fallback_experience()

    def _create_fallback_experience(self) -> Dict[str, Any]:
        """Create a fallback experience response when parsing fails."""
        return {
            'us_experience_years': 0.0,
            'non_us_experience_years': 0.0,
            'total_professional_years': 0.0,
            'internship_count': 0,
            'experience_breakdown': [],
            'experience_strength': 'WEAK',
            'experience_flags': ['Unable to analyze experience'],
            'employment_gaps': []
        }

    def _calculate_skill_group_bonus(self, skill_analysis: Dict, skill_groups: Dict) -> float:
        """Calculate bonus score based on skill group coverage.
        
        Args:
            skill_analysis: Dictionary containing skills assessment data
            skill_groups: Dictionary mapping group names to lists of skills
            
        Returns:
            float: Bonus multiplier between 0.0 and 0.3
        """
        try:
            total_bonus = 0.0
            assessed_skills = {
                assessment['skill'].lower() 
                for assessment in skill_analysis.get('skills_assessment', [])
            }
            
            logger.info(f"Calculating skill group bonus. Found {len(assessed_skills)} assessed skills")
            logger.debug(f"Assessed skills: {assessed_skills}")
            
            for group_name, group_skills in skill_groups.items():
                group_skills_lower = {skill.lower() for skill in group_skills}
                if not group_skills_lower:  # Skip empty groups
                    continue
                    
                matched_skills = group_skills_lower.intersection(assessed_skills)
                coverage = len(matched_skills) / len(group_skills_lower)
                
                logger.info(f"Skill group '{group_name}' analysis:")
                logger.info(f"  - Total skills in group: {len(group_skills_lower)}")
                logger.info(f"  - Matched skills: {len(matched_skills)}")
                logger.info(f"  - Coverage ratio: {coverage:.2f}")
                
                # Apply scaled bonus based on coverage
                group_bonus = 0.0
                if coverage >= 0.85:  # Strong coverage (85%+)
                    group_bonus = 0.20  # Higher bonus for exceptional coverage
                    logger.info(f"  - Strong coverage bonus applied: +0.20")
                elif coverage >= 0.70:  # Good coverage (70-84%)
                    group_bonus = 0.12
                    logger.info(f"  - Good coverage bonus applied: +0.12")
                elif coverage >= 0.50:  # Moderate coverage (50-69%)
                    group_bonus = 0.06
                    logger.info(f"  - Moderate coverage bonus applied: +0.06")
                else:
                    logger.info("  - No coverage bonus applied")
                
                total_bonus += group_bonus
            
            # Cap total bonus at 30% for exceptional candidates
            final_bonus = min(0.30, total_bonus)
            logger.info(f"Final skill group bonus (capped at 30%): {final_bonus:.2f}")
            
            return final_bonus
            
        except Exception as e:
            logger.error(f"Skill group bonus calculation failed: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            return 0.0
                    
    def _adjust_score_with_experience(self, base_score: int, experience_details: Optional[ExperienceDetails], role_config: Dict) -> int:
        """Adjust technical score based on experience analysis."""
        try:
            if not experience_details:
                return base_score

            # Get minimum required experience
            min_years = role_config.get('min_years_experience', 0)
            
            # Calculate experience multiplier with stronger penalties
            total_years = experience_details.total_professional_years
            if total_years >= min_years * 2:  # Significantly exceeds requirements
                exp_multiplier = 1.1  # Reduced from 1.15
            elif total_years >= min_years:  # Meets requirements
                exp_multiplier = 1.0  # Reduced from 1.1
            else:  # Below requirements
                # Progressive penalty based on shortfall
                ratio = total_years / min_years
                exp_multiplier = max(0.5, ratio)  # Minimum 50% of original score
            
            # Apply domain expertise multiplier with stronger penalties
            if experience_details.experience_strength == "STRONG":
                domain_multiplier = 1.0  # Reduced from 1.1
            elif experience_details.experience_strength == "MODERATE":
                domain_multiplier = 0.9  # Reduced from 1.05
            else:
                domain_multiplier = 0.7  # Reduced from 1.0
            
            # Additional penalties for academic/research focus
            context_multiplier = 1.0
            if hasattr(experience_details, 'context'):
                if experience_details.context == 'academic':
                    context_multiplier = 0.6  # 40% penalty for academic context
                elif experience_details.context == 'research':
                    context_multiplier = 0.7  # 30% penalty for research context
            
            # Calculate final score with all multipliers
            adjusted_score = base_score * exp_multiplier * domain_multiplier * context_multiplier
            
            # Additional penalty for early career (less than 2 years)
            if total_years < 2:
                adjusted_score *= 0.7  # 30% penalty for very early career
            
            return min(100, round(adjusted_score))
            
        except Exception as e:
            logger.error(f"Score adjustment failed: {str(e)}")
            return base_score

    def _determine_recommendation(self, technical_score: int, skill_analysis: Dict, experience_details: Optional[ExperienceDetails]) -> MatchStrength:
        """Determine final recommendation based on all factors."""
        try:
            # Base recommendation on technical score
            if technical_score >= 85:
                base_rec = MatchStrength.STRONG
            elif technical_score >= 70:
                base_rec = MatchStrength.GOOD
            elif technical_score >= 50:
                base_rec = MatchStrength.POTENTIAL
            else:
                base_rec = MatchStrength.NO_MATCH
            
            # Adjust based on critical factors
            if base_rec in [MatchStrength.STRONG, MatchStrength.GOOD]:
                # Check for critical gaps
                if skill_analysis.get('technical_gaps', []):
                    critical_gaps = [gap for gap in skill_analysis['technical_gaps'] 
                                  if 'critical' in gap.lower() or 'essential' in gap.lower()]
                    if critical_gaps:
                        base_rec = MatchStrength.POTENTIAL
                
                # Check experience strength
                if experience_details and experience_details.experience_strength == "WEAK":
                    base_rec = MatchStrength(min(base_rec.value, MatchStrength.POTENTIAL.value))
            
            return base_rec
            
        except Exception as e:
            logger.error(f"Recommendation determination failed: {str(e)}")
            return MatchStrength.NO_MATCH

    def _generate_interview_questions(self, skill_analysis: Dict, experience_details: Optional[ExperienceDetails], role_config: Dict) -> List[str]:
        """Generate targeted interview questions based on analysis."""
        try:
            # Base questions on role requirements
            questions = []
            
            # Add skill-specific questions
            for skill_assessment in skill_analysis.get('skills_assessment', []):
                if skill_assessment['skill'] in role_config['required_skills']:
                    questions.append(
                        f"Describe a complex problem you solved using {skill_assessment['skill']}. "
                        "What was the challenge, your approach, and the outcome?"
                    )
            
            # Add experience-based questions
            if experience_details:
                if experience_details.experience_flags:
                    for flag in experience_details.experience_flags:
                        questions.append(
                            f"Regarding {flag}, can you elaborate on your experience "
                            "and how you've addressed similar challenges?"
                        )
            
            # Add technical gap questions
            for gap in skill_analysis.get('technical_gaps', [])[:2]:
                questions.append(
                    f"How do you plan to address the gap in {gap}? "
                    "What steps have you taken to develop in this area?"
                )
            
            return questions[:5]  # Limit to top 5 questions
            
        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            return [
                "Please describe your technical background and experience",
                "What are your core technical skills?",
                "How do you approach learning new technologies?"
            ]

    def _get_fallback_experience_response(self) -> AnalysisResult:
        """Provide a default response when analysis fails."""
        logger.info("Using fallback experience response")
        return AnalysisResult(
            technical_match_score=0,
            recommendation=MatchStrength.NO_MATCH,
            skills_assessment=[],
            technical_gaps=["Unable to analyze technical skills"],
            interview_questions=[
                "Please describe your technical background and experience",
                "What are your core technical skills?",
                "What is your experience with the required technologies?"
            ],
            key_findings=["Technical analysis failed - manual review required"],
            concerns=["Unable to automatically assess technical qualifications"]
        )

    def _extract_json_from_text(self, text: str) -> str:
        """Extract and clean JSON from text with improved robustness."""
        try:
            # Pre-process the text to handle common LLM formatting
            text = text.strip()
            
            # Log the raw text for debugging
            logger.debug(f"Processing text of length: {len(text)}")
            
            # Remove any markdown code block indicators and find JSON content
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
            
            # If text starts with a newline and curly brace, clean it up
            text = re.sub(r'^\n*{', '{', text)
            
            # Handle case where text is already valid JSON
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass
            
            # Find the outermost JSON object
            json_str = None
            brace_count = 0
            start = -1
            
            for i, char in enumerate(text):
                if char == '{':
                    if brace_count == 0:
                        start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        json_str = text[start:i+1]
                        break
            
            if not json_str and text.strip().startswith('{') and text.strip().endswith('}'):
                json_str = text.strip()
            
            if not json_str:
                logger.error("No complete JSON object found in response")
                return self._create_fallback_json()
            
            # Clean up the JSON string
            json_str = re.sub(r'[\n\r\t]', ' ', json_str)  # Remove newlines and tabs
            json_str = re.sub(r'\s+', ' ', json_str)       # Normalize whitespace
            
            # Fix common JSON formatting issues
            fixes = [
                (r',\s*}', '}'),                    # Remove trailing commas in objects
                (r',\s*]', ']'),                    # Remove trailing commas in arrays
                (r'([{,])\s*([^"\s{}\[\],]+)\s*:', r'\1"\2":'),  # Quote unquoted keys
                (r':\s*"([^"]*)"([,}])', r':"\1"\2'),  # Fix string value formatting
                (r':\s*(true|false|null)([,}])', r':\1\2'),  # Fix boolean/null values
                (r':\s*(\d+)([,}])', r':\1\2'),    # Fix number formatting
                (r'"\s*\+\s*"', ''),               # Remove string concatenation
            ]
            
            for pattern, replacement in fixes:
                try:
                    json_str = re.sub(pattern, replacement, json_str)
                except Exception as e:
                    logger.warning(f"Fix pattern failed: {str(e)}")
                    continue
            
            # Try to parse the cleaned JSON
            try:
                parsed = json.loads(json_str)
                # Ensure required fields exist
                if 'technical_match_score' not in parsed:
                    parsed['technical_match_score'] = 0
                if 'recommendation' not in parsed:
                    parsed['recommendation'] = "NO_MATCH"
                if 'skills_assessment' not in parsed:
                    parsed['skills_assessment'] = []
                if 'technical_gaps' not in parsed:
                    parsed['technical_gaps'] = []
                if 'interview_questions' not in parsed:
                    parsed['interview_questions'] = []
                if 'key_findings' not in parsed:
                    parsed['key_findings'] = []
                if 'concerns' not in parsed:
                    parsed['concerns'] = []
                if 'confidence_score' not in parsed:
                    parsed['confidence_score'] = 0.0
                    
                return json.dumps(parsed)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.debug(f"Failed JSON string: {json_str[:200]}...")
                return self._create_fallback_json()
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            return self._create_fallback_json()

    def _create_fallback_json(self) -> str:
        """Create a fallback JSON response when parsing fails."""
        fallback = {
            "technical_match_score": 0,
            "recommendation": "NO_MATCH",
            "skills_assessment": [],
            "technical_gaps": ["Unable to analyze technical skills"],
            "interview_questions": [
                "Please describe your technical background and experience",
                "What are your core technical skills?",
                "How do you approach learning new technologies?"
            ],
            "key_findings": ["Analysis failed - manual review required"],
            "concerns": ["Unable to automatically assess qualifications"],
            "confidence_score": 0.0
        }
        return json.dumps(fallback)

    def _create_fallback_response(self) -> AnalysisResult:
        """Create fallback response when analysis fails."""
        return AnalysisResult(
            technical_match_score=0,
            recommendation=MatchStrength.NO_MATCH,
            skills_assessment=[],
            technical_gaps=['Unable to analyze technical skills'],
            interview_questions=[InterviewQuestion(
                category='General Technical',
                question='Please describe your technical background and experience',
                context='Fallback question due to analysis failure'
            )],
            key_findings=['Analysis failed - manual review required'],
            concerns=['Unable to automatically assess qualifications'],
            confidence_score=0.0
        )

    def _extract_tech_contexts(self, resume_text: str) -> List[Dict[str, Any]]:
        """Extract technology usage contexts from resume text."""
        tech_contexts = []
        
        try:
            # Get tech context patterns from config
            patterns = self.config.get('tech_context_patterns', {})
            
            # Get all required and preferred skills
            all_skills = set()
            for role_config in self.config.get('job_roles', {}).values():
                required = role_config.get('required_skills', [])
                preferred = role_config.get('preferred_skills', [])
                
                # Handle both string and dict skills
                for skill in required + preferred:
                    if isinstance(skill, dict):
                        all_skills.add(skill['name'])
                    else:
                        all_skills.add(skill)
            
            # Find contexts for each tech
            for tech in all_skills:
                for context_type, context_patterns in patterns.items():
                    for pattern in context_patterns:
                        try:
                            # Format pattern with tech name
                            tech_pattern = pattern.format(tech=re.escape(tech))
                            matches = re.finditer(tech_pattern, resume_text, re.I)
                            
                            for match in matches:
                                tech_contexts.append({
                                    'tech': tech,
                                    'type': context_type,
                                    'context': self._extract_context(resume_text, match.start(), 150)
                                })
                        except Exception as e:
                            logger.warning(f"Pattern matching failed for {tech}: {str(e)}")
                            continue
            
            return tech_contexts
            
        except Exception as e:
            logger.error(f"Tech context extraction failed: {str(e)}")
            return []

    def _extract_context(self, text: str, position: int, window_size: int = 150) -> str:
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

    def _generate_contextual_questions(
        self, 
        resume_text: str,
        tech_contexts: List[Dict],
        matched_skills: Dict,
        experience_details: Dict
    ) -> List[InterviewQuestion]:
        """Generate contextual interview questions based on resume analysis"""
        questions = []
        
        try:
            # Get tech usage patterns
            tech_patterns = {}
            for ctx in tech_contexts:
                tech = ctx['tech']
                if tech not in tech_patterns:
                    tech_patterns[tech] = []
                tech_patterns[tech].append({
                    'type': ctx['type'],
                    'context': ctx['context']
                })

            # Generate tech-specific questions
            for tech, patterns in tech_patterns.items():
                # Get most significant usage
                main_usage = max(patterns, key=lambda x: len(x['context']))
                
                # Add implementation question
                questions.append(InterviewQuestion(
                    category='Technical Implementation',
                    question=f"In {main_usage['context']}, you used {tech}. Could you discuss:\n"
                            f"- The specific technical challenges\n"
                            f"- Your implementation approach\n"
                            f"- Key decisions and trade-offs?",
                    context=main_usage['context']
                ))

            # Generate experience-based questions
            if experience_details:
                if experience_details.get('context') == 'academic':
                    questions.append(InterviewQuestion(
                        category='Experience Translation',
                        question="Your experience appears primarily academic. How would you:\n"
                                "- Adapt your skills to production environments\n"
                                "- Handle enterprise-scale challenges\n"
                                "- Apply research experience to business problems?",
                        context='Academic background'
                    ))

            # Generate gap-based questions
            missing_skills = set(matched_skills.get('required_skills', [])) - set(matched_skills.get('matched_skills', []))
            if missing_skills:
                questions.append(InterviewQuestion(
                    category='Skill Gaps',
                    question=f"I notice less experience with {', '.join(missing_skills)}. Could you:\n"
                            f"- Describe any exposure to these technologies\n"
                            f"- Explain how you'd apply related experience\n"
                            f"- Share your approach to learning new skills?",
                    context=f"Missing skills: {', '.join(missing_skills)}"
                ))

            return questions

        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            return [InterviewQuestion(
                category='General Technical',
                question="Could you walk me through your most significant technical project?",
                context='Fallback question'
            )]
