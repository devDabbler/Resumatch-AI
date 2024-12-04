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

    def analyze_resume(self, resume_text: str, role_name: str, matched_skills: Dict, extracted_experience: List[str]) -> Dict:
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
                
                # Adjust technical score based on experience analysis
                if experience_analysis:
                    # Penalize for short stints if there are more than 2
                    if len(experience_analysis.get('short_stints', [])) > 2:
                        technical_score = int(technical_score * 0.9)  # 10% penalty
                        
                    # Penalize for employment gaps
                    if len(experience_analysis.get('experience_flags', [])) > 0:
                        technical_score = int(technical_score * 0.95)  # 5% penalty
                        
                    # Adjust based on US vs non-US experience ratio
                    us_exp_years = experience_analysis.get('us_experience_years', 0)
                    total_exp_years = experience_analysis.get('total_professional_years', 0)
                    if total_exp_years > 0:
                        us_ratio = us_exp_years / total_exp_years
                        if us_ratio < 0.3:  # Less than 30% US experience
                            technical_score = int(technical_score * 0.9)  # 10% penalty
            
            # Add confidence_score for backward compatibility
            technical_analysis["confidence_score"] = technical_score
            technical_analysis["technical_match_score"] = technical_score
            
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
        """Extract and repair truncated JSON."""
        try:
            text = text.strip()
            logger.debug(f"Input text: {text[:200]}...")

            # Find JSON content
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found")
                
            json_str = text[start:end]

            # Handle truncation by completing the JSON structure
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            # Add missing closing braces/brackets if truncated
            json_str += ']' * (open_brackets - close_brackets)
            json_str += '}' * (open_braces - close_braces)

            # Clean and parse
            try:
                return json.dumps(json.loads(json_str), ensure_ascii=False)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial parse failed: {str(e)}, attempting additional fixes")
                json_str = re.sub(r'([^",}\]]\s*)([\]}])', r'"\1"\2', json_str)
                return json.dumps(json.loads(json_str), ensure_ascii=False)

        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            logger.error(f"Problematic text: {text[:500]}")
            raise ValueError(f"Failed to extract valid JSON: {str(e)}")

    def _mixtral_technical_analysis(self, resume_text, role_name, matched_skills, extracted_experience, technical_score):
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
                "You are an expert technical recruiter analyzing resumes. "
                "Your task is to provide a detailed technical analysis in JSON format. "
                "Rules:\n"
                "1. ONLY return a valid JSON object\n"
                "2. Include ALL required fields even if empty\n"
                "3. Use proper JSON formatting with double quotes\n"
                "4. Provide specific, detailed responses for each field\n"
                "5. Base analysis on the provided resume text and role requirements\n"
                "6. Include at least 3 interview questions\n"
                "7. Provide detailed key findings\n"
                "8. List specific technical gaps\n"
                "9. Include detailed skills assessment\n"
                "10. Do not include any text outside the JSON object"
            )
            
            user_prompt = f"""Analyze technical qualifications for {role_name} and return a detailed JSON object with this exact structure:

{{
    "technical_match_score": {technical_score},
    "skills_assessment": [
        {{
            "skill": "Python",
            "proficiency": "Expert",
            "years": 5,
            "context": "Used Python for data analysis and machine learning"
        }},
        {{
            "skill": "Machine Learning",
            "proficiency": "Advanced",
            "years": 3,
            "context": "Implemented various ML models"
        }}
    ],
    "technical_gaps": [
        "Limited experience with specific technology X",
        "Missing certification Y"
    ],
    "interview_questions": [
        "Describe your experience with technology X",
        "How have you implemented Y in previous projects?",
        "What challenges did you face with Z?"
    ],
    "recommendation": "{recommendation}",
    "key_findings": [
        "Strong background in A",
        "Demonstrated expertise in B",
        "Notable achievements in C"
    ],
    "concerns": [
        "Gap in critical skill X",
        "Limited experience with Y"
    ]
}}

Resume Text: {resume_text}
Required Skills: {', '.join(role_config['required_skills'])}
Preferred Skills: {', '.join(role_config['preferred_skills'])}
Min Experience: {role_config['min_years_experience']}
Matched Required: {', '.join(matched_required)}
Matched Preferred: {', '.join(matched_preferred)}
Experience: {experience_summary}
Score: {technical_score}
Recommendation: {recommendation}

Important:
1. Provide DETAILED responses for each field
2. Include SPECIFIC examples from the resume
3. Make interview questions RELEVANT to the role
4. Base analysis on ACTUAL resume content
5. Return ONLY the JSON object with NO additional text"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Log the prompts for debugging
            logger.debug(f"System prompt: {system_prompt}")
            logger.debug(f"User prompt: {user_prompt}")
            
            response = self.execute_request(messages)
            content = response.choices[0].message.content
            
            # Enhanced logging for response content
            logger.info("Received Mixtral response")
            logger.debug(f"Raw LLM response content: {content[:500]}...")  # Log first 500 chars
            
            # Extract and clean JSON
            try:
                json_str = self._extract_json_from_text(content)
                logger.debug(f"Extracted JSON string: {json_str[:500]}...")  # Log extracted JSON
                
                result = json.loads(json_str)
                logger.debug(f"Successfully parsed JSON result: {json.dumps(result, indent=2)}")
                
                # Force the technical score and recommendation
                result['technical_match_score'] = technical_score
                result['recommendation'] = recommendation
                
                # Enhanced validation with logging
                required_fields = {
                    "technical_match_score": int,
                    "skills_assessment": list,
                    "technical_gaps": list,
                    "interview_questions": list,
                    "recommendation": str,
                    "key_findings": list,
                    "concerns": list
                }
                
                missing_fields = []
                invalid_types = []
                
                for field, field_type in required_fields.items():
                    if field not in result:
                        missing_fields.append(field)
                        result[field] = [] if field_type == list else (0 if field_type == int else "")
                    elif not isinstance(result[field], field_type):
                        invalid_types.append(f"{field} (expected {field_type.__name__})")
                        # Convert to correct type
                        if field_type == list:
                            result[field] = [result[field]] if result[field] else []
                        elif field_type == int:
                            try:
                                result[field] = int(float(result[field]))
                            except (ValueError, TypeError):
                                result[field] = 0
                        else:
                            result[field] = str(result[field])

                if missing_fields:
                    logger.warning(f"Missing fields in LLM response: {missing_fields}")
                if invalid_types:
                    logger.warning(f"Invalid field types in LLM response: {invalid_types}")
                
                # Log final processed result
                logger.debug(f"Final processed result: {json.dumps(result, indent=2)}")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Problematic content: {content[:500]}...")  # Log problematic content
                return self._get_fallback_technical_response(technical_score, recommendation)
                
        except Exception as e:
            logger.error(f"Mixtral analysis failed: {str(e)}")
            logger.error(f"Full error details: {traceback.format_exc()}")
            return self._get_fallback_technical_response(technical_score, recommendation)

    def _get_fallback_technical_response(self, technical_score=0, recommendation="NO_MATCH"):
        """Return a fallback technical analysis response"""
        return {
            "technical_match_score": technical_score,
            "confidence_score": technical_score,
            "skills_assessment": [
                {
                    "skill": "Unable to assess",
                    "proficiency": "Unknown",
                    "years": 0
                }
            ],
            "technical_gaps": ["Error analyzing skills"],
            "interview_questions": [
                "What are your core technical skills?",
                "Can you describe your most relevant experience?",
                "What is your experience with the required technologies?"
            ],
            "recommendation": recommendation,
            "key_findings": [
                "Technical skills assessment incomplete",
                "Manual review recommended"
            ],
            "concerns": ["Unable to complete detailed analysis"]
        }

    def _gemini_experience_analysis(self, resume_text, role_name):
        """Analyze and classify work experience using Gemini"""
        try:
            # Clean and normalize the resume text
            resume_text = self._clean_text(resume_text)
            
            prompt = (
                "You are an expert resume analyzer. "
                "Create a structured analysis of the candidate's work experience "
                "focusing on location, duration, and potential red flags.\n\n"
                f"Role: {role_name}\n"
                f"Resume:\n{resume_text}\n\n"
                "Instructions:\n"
                "1. Calculate years of experience by location and type\n"
                "2. Identify short-term positions (less than 1 year)\n"
                "3. Identify employment gaps (more than 6 months)\n"
                "4. Evaluate overall experience strength\n"
                "5. Flag any concerning patterns\n\n"
                "Format your response EXACTLY like this example with quoted property names:\n"
                '{\n'
                '  "us_experience_years": 5.5,\n'
                '  "non_us_experience_years": 2.0,\n'
                '  "total_professional_years": 7.5,\n'
                '  "internship_count": 1,\n'
                '  "short_stints": [\n'
                '    "Company A: 4 months",\n'
                '    "Company B: 8 months"\n'
                '  ],\n'
                '  "experience_gaps": [\n'
                '    "1.5 year gap between Company B and C"\n'
                '  ],\n'
                '  "experience_breakdown": [\n'
                '    "5 years software development",\n'
                '    "2 years project management"\n'
                '  ],\n'
                '  "experience_strength": "STRONG",\n'
                '  "experience_flags": [\n'
                '    "Multiple short-term positions",\n'
                '    "Significant employment gap",\n'
                '    "Limited US experience"\n'
                '  ]\n'
                '}\n\n'
                'Important:\n'
                '1. Use double quotes for ALL property names and string values\n'
                '2. Always include commas between array elements and object properties\n'
                '3. Format numbers as decimals for years (e.g., 2.5) and integers for counts\n'
                '4. Use [] for empty arrays, never null\n'
                '5. Include all required fields, even if empty\n'
                '6. Do not include any text before or after the JSON object\n'
                '7. Do not use markdown formatting or code blocks\n'
                '8. Ensure all arrays and objects are properly closed\n'
                '9. Use proper JSON boolean values (true/false) and null\n'
                '10. Keep analysis professional and objective'
            )

            # Set up generation parameters to match Mixtral settings
            generation_config = {
                "temperature": 0.01,  # Reduced temperature for more deterministic output
                "top_p": 0.1,         # Added top_p for more focused sampling
                "top_k": 1,           # Keep only the most likely token
                "max_output_tokens": 2000,  # Increased max tokens to avoid truncation
                "candidate_count": 1   # Generate only one response
            }

            # Updated safety settings to be more permissive
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]

            response = self.gemini_client.generate_content(
                contents=prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if hasattr(response, 'prompt_feedback'):
                safety_ratings = response.prompt_feedback.safety_ratings
                for rating in safety_ratings:
                    if rating.probability >= rating.threshold:
                        logger.warning(f"Content blocked: {rating.category}")
                        return self._get_fallback_experience_response()

            # Get and clean the response
            raw_text = response.text.strip()
            logger.info("Received Gemini response")

            # Extract and parse JSON content
            json_str = self._extract_json_from_text(raw_text)
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'us_experience_years', 'non_us_experience_years',
                'total_professional_years', 'internship_count',
                'short_stints', 'experience_gaps',
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

            # Ensure arrays are never null
            array_fields = ['short_stints', 'experience_gaps', 'experience_breakdown', 'experience_flags']
            for field in array_fields:
                if field not in result or result[field] is None:
                    result[field] = []

            return result

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
            "short_stints": [],
            "experience_gaps": [],
            "experience_breakdown": [],
            "experience_strength": "LIMITED",
            "experience_flags": ["Error analyzing experience"]
        }

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
            required_skills_threshold = scoring_constraints.get('required_skills_threshold', 0.85)
            minimum_skills_match = scoring_constraints.get('minimum_skills_match', 0.75)
        
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
        
            # Calculate final score with adjusted weights
            skill_weights = config['scoring_config']['skill_weights']
            required_weight = skill_weights['required']
            preferred_weight = skill_weights['preferred']
        
            # Apply threshold adjustments
            if required_ratio >= required_skills_threshold:
                required_ratio *= 1.2  # 20% bonus for meeting high threshold
            elif required_ratio >= minimum_skills_match:
                required_ratio *= 1.1  # 10% bonus for meeting minimum threshold
            else:
                required_ratio *= 0.8  # 20% penalty for not meeting minimum
            
            # Cap ratios at 1.0
            required_ratio = min(1.0, required_ratio)
            preferred_ratio = min(1.0, preferred_ratio)
            
            # Calculate weighted score with stricter scaling
            raw_score = (
                (required_ratio * required_weight + preferred_ratio * preferred_weight)
                / (required_weight + preferred_weight)  # Normalize to 0-1 range
                * max_score
            )
            
            # Apply scaling based on required skills ratio
            if required_ratio < minimum_skills_match:
                raw_score *= 0.6  # Significant penalty
            elif required_ratio < required_skills_threshold:
                raw_score *= 0.8  # Moderate penalty
            
            # Round to nearest integer and ensure within bounds
            final_score = min(max_score, max(0, round(raw_score)))
            logger.info(f"Final technical score: {final_score}/{max_score}")
            return final_score
        
        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return 0
