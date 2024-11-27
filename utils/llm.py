from groq import Groq
import google.generativeai as genai
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
import logging
import json
import time
import re
import traceback
import yaml
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
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        self.mixtral_model = "mixtral-8x7b-32768"
        self.gemini_model = "gemini-pro"
        
        self.groq_client = None
        self.gemini_client = None
        
        if not self.groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found - Gemini features disabled")
        
        try:
            self.initialize_clients()
        except Exception as e:
            logger.error(f"Failed to initialize during __init__: {str(e)}")
            raise

    def initialize_clients(self) -> None:
        """Initialize the LLM clients."""
        try:
            # Initialize Groq
            self.groq_client = Groq(api_key=self.groq_api_key)
            logger.info("Groq client initialized successfully")
            
            # Initialize Gemini if key available
            if self.gemini_api_key:
                try:
                    genai.configure(api_key=self.gemini_api_key)
                    model_config = {
                        "temperature": 0.7,
                        "top_p": 1,
                        "top_k": 1,
                        "max_output_tokens": 2048,
                    }
                    self.gemini_client = genai.GenerativeModel(
                        model_name=self.gemini_model,
                        generation_config=model_config
                    )
                    logger.info("Gemini client initialized successfully")
                except ImportError as e:
                    logger.error(f"Failed to import Gemini dependencies: {str(e)}")
                    self.gemini_client = None
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini client: {str(e)}")
                    self.gemini_client = None
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {str(e)}")
            raise

    def execute_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute a request to Groq with retry logic."""
        if not self.groq_client:
            logger.info("Client not initialized, initializing now...")
            self.initialize_clients()
            
        retries = 3
        for attempt in range(retries):
            try:
                logger.info(f"Making API request (attempt {attempt + 1}/{retries})")
                
                response = self.groq_client.chat.completions.create(
                    messages=messages,
                    model=self.mixtral_model,
                    temperature=0.7,
                    max_tokens=1000
                )
                logger.info("API request successful")
                return response
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if hasattr(e, 'response'):
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text}")
                
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed")
                    raise

    def _calculate_technical_score(self, matched_skills: Dict, role_name: str) -> int:
        """Calculate technical match score based on matched skills and role-specific constraints."""
        try:
            with open('config/jobs.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            role_config = config['job_roles'].get(role_name)
            if not role_config:
                logger.error(f"Role configuration not found for: {role_name}")
                return 0
            
            # Get scoring constraints with defaults
            scoring_constraints = role_config.get('scoring_constraints', {})
            max_score = scoring_constraints.get('max_score', 100)
            required_skills_threshold = scoring_constraints.get('required_skills_threshold', 0.5)
            
            # Calculate required skills match
            total_required = len(role_config['required_skills'])
            matched_required = len(matched_skills.get('required', []))
            required_ratio = matched_required / total_required if total_required > 0 else 0
            
            # Calculate preferred skills match
            total_preferred = len(role_config['preferred_skills'])
            matched_preferred = len(matched_skills.get('preferred', []))
            preferred_ratio = matched_preferred / total_preferred if total_preferred > 0 else 0
            
            logger.info(f"Required skills match: {required_ratio:.2f} ({matched_required}/{total_required})")
            logger.info(f"Preferred skills match: {preferred_ratio:.2f} ({matched_preferred}/{total_preferred})")
            
            # Get skill weights from config
            skill_weights = config['scoring_config']['skill_weights']
            required_weight = skill_weights['required']
            preferred_weight = skill_weights['preferred']
            
            # Calculate base score using weighted average
            base_score = (
                required_ratio * required_weight +
                preferred_ratio * preferred_weight
            ) * max_score
            
            # Apply gradual scaling based on required skills ratio
            if required_ratio >= required_skills_threshold:
                # Full score if above threshold
                final_score = base_score
            else:
                # Gradual reduction based on how close to threshold
                reduction_factor = (required_ratio / required_skills_threshold) ** 0.5  # Square root for less aggressive reduction
                final_score = base_score * reduction_factor
            
            # Check platform-specific requirements if they exist
            if 'platform_specific_requirements' in scoring_constraints:
                platform_skills = scoring_constraints['platform_specific_requirements']
                min_platform_skills = scoring_constraints.get('minimum_platform_skills', 0)
                
                # Count matched platform-specific skills
                matched_platform_skills = sum(
                    1 for skill in platform_skills 
                    if any(skill.lower() in matched.lower() for matched in matched_skills.get('required', []))
                )
                
                logger.info(f"Platform-specific skills match: {matched_platform_skills}/{len(platform_skills)}")
                
                if matched_platform_skills < min_platform_skills:
                    # Reduce score but don't zero it out
                    platform_reduction = 0.7  # 30% reduction for missing platform skills
                    final_score *= platform_reduction
                    logger.info(f"Applied platform skills reduction: {platform_reduction}")
            
            # Ensure score doesn't exceed max_score
            final_score = min(int(final_score), max_score)
            logger.info(f"Final technical score: {final_score}/{max_score}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return 0

    def analyze_resume(self, 
                      resume_text: str, 
                      role_name: str,
                      matched_skills: Dict,
                      extracted_experience: List[str]) -> Dict:
        """
        Dual analysis:
        - Mixtral: Technical skills evaluation
        - Gemini: Experience validation and classification
        """
        try:
            # Clean resume text to handle Unicode characters
            cleaned_text = self._clean_text(resume_text)
            
            # Calculate technical match score
            technical_score = self._calculate_technical_score(matched_skills, role_name)
            
            technical_analysis = self._mixtral_technical_analysis(
                cleaned_text, role_name, matched_skills, extracted_experience, technical_score
            )
            
            if self.gemini_client:
                experience_analysis = self._gemini_experience_analysis(
                    cleaned_text, role_name
                )
                technical_analysis.update({
                    "experience_details": experience_analysis
                })
            
            # Add confidence_score for backward compatibility
            technical_analysis["confidence_score"] = technical_analysis["technical_match_score"]
            
            return technical_analysis
            
        except Exception as e:
            logger.error(f"Resume analysis failed: {str(e)}")
            raise

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

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from text."""
        try:
            # Find the first { and last } in the text
            start = text.find('{')
            end = text.rfind('}')
            
            if start == -1 or end == -1:
                raise ValueError("No JSON object markers found in text")
            
            # Extract the potential JSON string
            json_str = text[start:end + 1]
            
            # Clean the extracted JSON string
            json_str = re.sub(r'[\n\r\t]', '', json_str)
            json_str = re.sub(r'\s+', ' ', json_str)
            
            # Remove any invalid escape sequences
            json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', json_str)
            
            # Validate JSON structure
            json.loads(json_str)
            
            return json_str
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            raise

    def _mixtral_technical_analysis(self, resume_text, role_name, 
                                  matched_skills, extracted_experience,
                                  technical_score):
        """Technical skills analysis using Mixtral"""
        try:
            # Load job roles configuration
            with open('config/jobs.yaml', 'r') as f:
                config = yaml.safe_load(f)
                
            role_config = config['job_roles'].get(role_name)
            if not role_config:
                raise ValueError(f"Role {role_name} not found in configuration")
            
            # Safely handle matched_skills dictionary with default empty lists
            matched_skills = matched_skills if isinstance(matched_skills, dict) else {}
            matched_required = list(matched_skills.get('required', []))
            matched_preferred = list(matched_skills.get('preferred', []))
            
            # Format experience matches
            experience_summary = "\n".join(extracted_experience) if extracted_experience else "No experience matches found"
            
            # Determine recommendation based on technical score
            max_score = role_config.get('scoring_constraints', {}).get('max_score', 100)
            if technical_score >= int(max_score * 0.85):
                recommendation = "STRONG_MATCH"
            elif technical_score >= int(max_score * 0.75):
                recommendation = "GOOD_MATCH"
            elif technical_score >= int(max_score * 0.50):
                recommendation = "POTENTIAL_MATCH"
            else:
                recommendation = "NO_MATCH"
            
            system_prompt = (
                "You are an expert technical recruiter. Analyze the resume focusing "
                "on technical skills, qualifications, and provide comprehensive feedback. "
                "Return a valid JSON response with all required fields."
            )
            
            user_prompt = f"""Analyze technical qualifications for {role_name}:
            
Resume Text:
{resume_text}
            
Required Technical Skills: {', '.join(role_config['required_skills'])}
Preferred Technical Skills: {', '.join(role_config['preferred_skills'])}
Minimum Years Experience: {role_config['min_years_experience']}
            
Matched Required Skills: {', '.join(matched_required)}
Matched Preferred Skills: {', '.join(matched_preferred)}
            
Experience Matches:
{experience_summary}

Technical Match Score: {technical_score}
Recommendation: {recommendation}
            
Return a valid JSON with these exact fields:
{{
    "technical_match_score": {technical_score},
    "skills_assessment": [
        {{
            "skill": <string>,
            "proficiency": <string>,
            "years": <number>
        }}
    ],
    "technical_gaps": [<strings of missing required skills>],
    "interview_questions": [<technical questions based on role>],
    "recommendation": "{recommendation}",
    "key_findings": [<strings of main positive points>],
    "concerns": [<strings of potential issues or red flags>]
}}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.execute_request(messages)
            content = response.choices[0].message.content
            
            # Extract and clean JSON
            json_str = self._extract_json_from_text(content)
            result = json.loads(json_str)
            
            # Force the technical score and recommendation
            result['technical_match_score'] = technical_score
            result['recommendation'] = recommendation
            
            # Validate required fields
            required_fields = [
                "technical_match_score",
                "skills_assessment",
                "technical_gaps",
                "interview_questions",
                "recommendation",
                "key_findings",
                "concerns"
            ]
            
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            return result
            
        except Exception as e:
            logger.error(f"Mixtral analysis failed: {str(e)}")
            return self._get_fallback_technical_response(technical_score, recommendation)

    def _get_fallback_technical_response(self, technical_score=0, recommendation="NO_MATCH"):
        """Return a fallback technical analysis response"""
        return {
            "technical_match_score": technical_score,
            "confidence_score": technical_score,
            "skills_assessment": [],
            "technical_gaps": ["Error analyzing skills"],
            "interview_questions": [],
            "recommendation": recommendation,
            "key_findings": ["Error during analysis"],
            "concerns": ["Unable to complete analysis"]
        }

    def _gemini_experience_analysis(self, resume_text, role_name):
        """Analyze and classify work experience using Gemini"""
        try:
            prompt = (
                "You are an expert resume analyzer. "
                "Create a structured analysis of the candidate's work experience "
                "focusing on location and type of experience.\n\n"
                f"Role: {role_name}\n"
                f"Resume:\n{resume_text}\n\n"
                "Instructions:\n"
                "1. Calculate years of experience by location and type\n"
                "2. Break down each role into components\n"
                "3. Evaluate overall experience strength\n"
                "4. Identify any potential flags or gaps\n\n"
                "Format your response EXACTLY like this example:\n"
                '{\n'
                '  "us_experience_years": 2.5,\n'
                '  "non_us_experience_years": 0.0,\n'
                '  "total_professional_years": 2.5,\n'
                '  "internship_count": 1,\n'
                '  "experience_breakdown": [\n'
                '    {\n'
                '      "type": "US_PROFESSIONAL",\n'
                '      "duration_years": 1.5,\n'
                '      "role": "Software Engineer",\n'
                '      "organization": "Tech Corp"\n'
                '    }\n'
                '  ],\n'
                '  "experience_strength": "MODERATE",\n'
                '  "experience_flags": []\n'
                '}'
            )

            # Set up generation parameters for consistent output
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.1,
                "top_k": 1,
                "max_output_tokens": 1024,
            }

            # Make the API call
            response = self.gemini_client.generate_content(
                contents=prompt,
                generation_config=generation_config
            )

            if not response or not response.text:
                logger.error("Empty response from Gemini")
                return self._get_fallback_experience_response()

            # Get and clean the response
            raw_text = response.text.strip()
            logger.info("Received Gemini response")

            try:
                # Extract and parse JSON content
                json_str = self._extract_json_from_text(raw_text)
                result = json.loads(json_str)
                
                # Validate required fields
                required_fields = [
                    'us_experience_years', 'non_us_experience_years',
                    'total_professional_years', 'internship_count',
                    'experience_breakdown', 'experience_strength',
                    'experience_flags'
                ]
                
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")

                # Convert numeric fields with dict comprehension
                numeric_fields = {
                    'us_experience_years': float,
                    'non_us_experience_years': float,
                    'total_professional_years': float,
                    'internship_count': int
                }
                
                for field, converter in numeric_fields.items():
                    try:
                        result[field] = converter(result[field])
                    except (TypeError, ValueError) as e:
                        logger.error(f"Error converting {field}: {str(e)}")
                        result[field] = converter(0)

                return result

            except Exception as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return self._get_fallback_experience_response()

        except Exception as e:
            logger.error(f"Experience analysis failed: {str(e)}")
            return self._get_fallback_experience_response()

    def _get_fallback_experience_response(self):
        """Return a fallback experience analysis response"""
        return {
            "us_experience_years": 0,
            "non_us_experience_years": 0,
            "total_professional_years": 0,
            "internship_count": 0,
            "experience_breakdown": [],
            "experience_strength": "LIMITED",
            "experience_flags": ["Error analyzing experience"]
        }
