from groq import Groq
import google.generativeai as genai
from typing import Dict, List, Any
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
        """Extract JSON object from text with improved robustness."""
        try:
            # Log the raw text for debugging
            logger.debug(f"Raw text to parse: {text}")
            
            # Pre-process the text to handle common LLM formatting
            text = text.strip()
            
            # If the text starts with markdown-style language indicator, remove it
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            # Find the JSON content
            json_str = None
            
            # Try to find JSON between triple backticks first
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Find the outermost JSON object
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
                
                if not json_str:
                    # Try to find any JSON-like structure
                    potential_json = re.search(r'\{[^}]*\}', text)
                    if potential_json:
                        json_str = potential_json.group(0)
                    else:
                        raise ValueError("No valid JSON object found in text")
            
            # Clean the extracted JSON string
            json_str = re.sub(r'[\n\r\t]', ' ', json_str)
            json_str = re.sub(r'\s+', ' ', json_str)
            
            # Fix common JSON formatting issues
            def fix_json_formatting(s):
                # Remove any trailing commas before closing brackets/braces
                s = re.sub(r',(\s*[}\]])', r'\1', s)
                
                # Add missing commas between elements
                s = re.sub(r'([\]"}0-9])\s*([\[{"])', r'\1,\2', s)
                
                # Fix boolean and null values
                s = re.sub(r':\s*true\b', ': true', s, flags=re.IGNORECASE)
                s = re.sub(r':\s*false\b', ': false', s, flags=re.IGNORECASE)
                s = re.sub(r':\s*null\b', ': null', s, flags=re.IGNORECASE)
                
                # Quote unquoted property names
                s = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s)
                
                # Handle empty arrays and objects
                s = re.sub(r'\[\s*\]', '[]', s)
                s = re.sub(r'\{\s*\}', '{}', s)
                
                return s
            
            # Apply fixes multiple times to catch nested issues
            for _ in range(3):
                json_str = fix_json_formatting(json_str)
            
            try:
                # Try parsing the JSON
                parsed = json.loads(json_str)
                
                # Ensure required fields exist
                required_fields = {
                    'technical_match_score': 0,
                    'skills_assessment': [],
                    'technical_gaps': [],
                    'interview_questions': [],
                    'recommendation': 'MANUAL_REVIEW',
                    'key_findings': [],
                    'concerns': []
                }
                
                # Add any missing fields with default values
                for field, default_value in required_fields.items():
                    if field not in parsed:
                        parsed[field] = default_value
                
                return json.dumps(parsed, ensure_ascii=False)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {str(e)}")
                # Try one more time with additional cleaning
                json_str = re.sub(r'([{,]\s*)([^"\s{},\[\]]+)\s*:', r'\1"\2":', json_str)
                json_str = re.sub(r'\\([^"\\\/bfnrt])', r'\1', json_str)
                json_str = re.sub(r'"\s+"', '","', json_str)
                json_str = re.sub(r'}\s*{', '},{', json_str)
                json_str = re.sub(r']\s*\[', '],[', json_str)
                json_str = re.sub(r':\s*(?=[,}])', ': null', json_str)
                
                # Parse again with default values
                try:
                    parsed = json.loads(json_str)
                    for field, default_value in required_fields.items():
                        if field not in parsed:
                            parsed[field] = default_value
                    return json.dumps(parsed, ensure_ascii=False)
                except:
                    return json.dumps(required_fields)
                
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            logger.debug(f"Problematic text: {text[:500]}...")
            
            # Return a valid JSON with default values
            default_response = {
                'technical_match_score': 0,
                'skills_assessment': [],
                'technical_gaps': ['Unable to analyze technical skills'],
                'interview_questions': [
                    'Please describe your technical background and experience',
                    'What are your core technical skills?',
                    'What is your experience with the required technologies?'
                ],
                'recommendation': 'MANUAL_REVIEW',
                'key_findings': ['Technical analysis failed - manual review required'],
                'concerns': ['Unable to automatically assess technical qualifications']
            }
            return json.dumps(default_response)

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
                "You are an expert technical recruiter analyzing a resume. "
                "Provide a detailed technical assessment in JSON format. "
                "Focus on specific technical qualifications and experience."
            )
            
            user_prompt = (
                f"Analyze this candidate's technical qualifications for {role_name} role:\n\n"
                f"Resume Text: {resume_text}\n\n"
                f"Required Skills: {', '.join(role_config['required_skills'])}\n"
                f"Preferred Skills: {', '.join(role_config['preferred_skills'])}\n"
                f"Min Experience: {role_config['min_years_experience']} years\n"
                f"Matched Required Skills: {', '.join(matched_required)}\n"
                f"Matched Preferred Skills: {', '.join(matched_preferred)}\n"
                f"Experience Summary: {experience_summary}\n"
                f"Technical Score: {technical_score}/100\n"
                f"Recommendation: {recommendation}\n\n"
                "Return a JSON object with these fields:\n"
                "{\n"
                f'    "technical_match_score": {technical_score},\n'
                '    "skills_assessment": [\n'
                '        {\n'
                '            "skill": "Python",\n'
                '            "proficiency": "Expert",\n'
                '            "years": 5\n'
                '        }\n'
                '    ],\n'
                '    "technical_gaps": [\n'
                '        "Missing required skill: Kubernetes"\n'
                '    ],\n'
                '    "interview_questions": [\n'
                '        "Describe your experience with Python async/await"\n'
                '    ],\n'
                f'    "recommendation": "{recommendation}",\n'
                '    "key_findings": [\n'
                '        "Strong backend development experience"\n'
                '    ],\n'
                '    "concerns": [\n'
                '        "Limited cloud experience"\n'
                '    ]\n'
                "}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.execute_request(messages)
            content = response.choices[0].message.content
            
            # Extract and parse JSON response
            json_str = self._extract_json_from_text(content)
            result = json.loads(json_str)
            
            # Ensure required fields exist with proper types
            result['technical_match_score'] = technical_score
            result['recommendation'] = recommendation
            
            if 'skills_assessment' not in result or not result['skills_assessment']:
                result['skills_assessment'] = []
                
            if 'technical_gaps' not in result or not result['technical_gaps']:
                result['technical_gaps'] = []
                
            if 'interview_questions' not in result or not result['interview_questions']:
                result['interview_questions'] = []
                
            if 'key_findings' not in result or not result['key_findings']:
                result['key_findings'] = []
                
            if 'concerns' not in result or not result['concerns']:
                result['concerns'] = []
            
            # Validate skills assessment format
            for skill in result['skills_assessment']:
                if not isinstance(skill, dict):
                    continue
                if 'skill' not in skill:
                    skill['skill'] = ''
                if 'proficiency' not in skill:
                    skill['proficiency'] = ''
                if 'years' not in skill:
                    skill['years'] = 0
                try:
                    skill['years'] = int(skill['years'])
                except (ValueError, TypeError):
                    skill['years'] = 0
            
            # Remove any empty strings from arrays
            result['technical_gaps'] = [gap for gap in result['technical_gaps'] if gap and isinstance(gap, str)]
            result['interview_questions'] = [q for q in result['interview_questions'] if q and isinstance(q, str)]
            result['key_findings'] = [f for f in result['key_findings'] if f and isinstance(f, str)]
            result['concerns'] = [c for c in result['concerns'] if c and isinstance(c, str)]
            
            return result
            
        except Exception as e:
            logger.error(f"Experience analysis failed: {str(e)}")
            return self._get_fallback_experience_response()

    def _get_fallback_experience_response(self) -> Dict[str, Any]:
        """Provide a default response when experience analysis fails."""
        logger.info("Using fallback experience response")
        return {
            "technical_match_score": 0,
            "skills_assessment": [],
            "technical_gaps": ["Unable to analyze technical skills"],
            "interview_questions": [
                "Please describe your technical background and experience",
                "What are your core technical skills?",
                "What is your experience with the required technologies?"
            ],
            "recommendation": "MANUAL_REVIEW",
            "key_findings": ["Technical analysis failed - manual review required"],
            "concerns": ["Unable to automatically assess technical qualifications"]
        }
    
    def _gemini_experience_analysis(self, resume_text, role_name):
        """Analyze and classify work experience using Gemini"""
        try:
            # Configure generation settings
            generation_config = {
                "temperature": 0.1,
                "candidate_count": 1,
                "max_output_tokens": 1024,
                "stop_sequences": []
            }

            # Create a factual, data-focused prompt
            prompt = (
                "Analyze the employment history in this resume. Extract only factual data:\n\n"
                f"Resume text:\n{resume_text}\n\n"
                "Instructions:\n"
                "1. Calculate total years of experience in each location\n"
                "2. Count internships and professional roles separately\n"
                "3. List experience chronologically\n"
                "4. Note any employment gaps\n\n"
                "Format response as JSON:\n"
                "{\n"
                '  "us_experience_years": 5.5,\n'
                '  "non_us_experience_years": 2.0,\n'
                '  "total_professional_years": 7.5,\n'
                '  "internship_count": 1,\n'
                '  "experience_breakdown": [\n'
                '    "5 years software development",\n'
                '    "2 years project management"\n'
                '  ],\n'
                '  "experience_strength": "STRONG",\n'
                '  "experience_flags": [\n'
                '    "Multiple role changes",\n'
                '    "Gap in employment"\n'
                '  ]\n'
                "}\n\n"
                "Focus on:\n"
                "- Employment dates\n"
                "- Job locations\n"
                "- Role durations\n"
                "- Professional vs internship roles"
            )

            try:
                # Configure safety settings to disable all content filtering
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
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]

                # Make API request with disabled safety settings
                response = self.gemini_client.generate_content(
                    contents=prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                if response and response.text:
                    raw_text = response.text.strip()
                    logger.info("Received Gemini response")
                    return self._parse_gemini_response(raw_text)

            except Exception as e:
                logger.error(f"Gemini request failed: {str(e)}")
                # Try again with more restrictive prompt
                try:
                    # Simplify prompt further
                    simple_prompt = (
                        "Extract employment data from this resume:\n\n"
                        f"Text: {resume_text}\n\n"
                        "Return JSON with:\n"
                        "- Years of experience by location\n"
                        "- Total professional years\n"
                        "- Number of internships\n"
                        "- Experience timeline\n"
                        "- Any career gaps"
                    )

                    response = self.gemini_client.generate_content(
                        contents=simple_prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings  # Keep safety settings disabled for retry
                    )

                    if response and response.text:
                        raw_text = response.text.strip()
                        logger.info("Received Gemini response on second attempt")
                        return self._parse_gemini_response(raw_text)

                except Exception as second_error:
                    logger.error(f"Both Gemini attempts failed: {str(second_error)}")
                    raise

            logger.error("Empty response from Gemini")
            return self._get_fallback_experience_response()

        except Exception as e:
            logger.error(f"Experience analysis failed: {str(e)}")
            return self._get_fallback_experience_response()

    def _parse_gemini_response(self, raw_text: str) -> Dict:
        """Parse and validate Gemini response"""
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

            # Convert numeric fields
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
            array_fields = ['experience_breakdown', 'experience_flags']
            for field in array_fields:
                if field not in result or result[field] is None:
                    result[field] = []

            return result

        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {str(e)}")
            return self._get_fallback_experience_response()

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
            us_experience_bonus = scoring_constraints.get('us_experience_bonus', 5)  # Get US experience bonus
        
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
            
            # Apply US experience bonus if applicable
            if matched_skills.get('experience_details', {}).get('us_experience_years', 0) > 0:
                raw_score = min(max_score, raw_score + us_experience_bonus)
            
            # Round to nearest integer and ensure within bounds
            final_score = min(max_score, max(0, round(raw_score)))
            logger.info(f"Final technical score: {final_score}/{max_score}")
            return final_score
        
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            logger.debug(f"Technical analysis error details: {traceback.format_exc()}")
            return 0
