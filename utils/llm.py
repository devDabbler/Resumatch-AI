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
            
            # Handle truncated or malformed values
            json_str = re.sub(r':\s*(?=[,}])', ': null', json_str)  # Add null for missing values
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            
            # Fix missing commas between array elements
            json_str = re.sub(r'(\]|\}|\"|\'|\d)\s+(\{|\[|\"|\d+)', r'\1,\2', json_str)
            
            # Remove trailing commas before closing brackets/braces
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix escape sequences
            def fix_escapes(match):
                s = match.group(0)
                if s in ['\\n', '\\r', '\\t', '\\"', '\\\\', '\\/', '\\b', '\\f']:
                    return s
                return s[1:]
            
            # Fix invalid escape sequences
            json_str = re.sub(r'\\[^"\\\/bfnrt]', fix_escapes, json_str)
            
            # Quote ALL property names (even if already quoted)
            def quote_props(match):
                prop = match.group(2).strip('"\'')  # Remove any existing quotes
                return f'{match.group(1)}"{prop}":'
            
            # Quote property names more aggressively
            json_str = re.sub(r'([{,]\s*)([^"\s{},\[\]]+)\s*:', quote_props, json_str)
            
            # Handle single quotes
            json_str = json_str.replace("'", '"')
            
            # Handle boolean values and null
            json_str = re.sub(r':\s*true\b', ': true', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r':\s*false\b', ': false', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r':\s*null\b', ': null', json_str, flags=re.IGNORECASE)
            
            # Fix missing commas in arrays and between objects
            json_str = re.sub(r'\]\s*\[', '],[', json_str)
            json_str = re.sub(r'\}\s*\{', '},{', json_str)
            
            # Fix missing commas after strings in arrays
            json_str = re.sub(r'"\s+(?=(?:[^"]*"[^"]*")*[^"]*$)', '",', json_str)
            
            # Ensure arrays and objects are properly terminated
            def balance_braces(s):
                stack = []
                for i, char in enumerate(s):
                    if char in '{[':
                        stack.append(char)
                    elif char in '}]':
                        if not stack:
                            continue
                        if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                            stack.pop()
                
                # Close any remaining open braces/brackets
                while stack:
                    char = stack.pop()
                    s += '}' if char == '{' else ']'
                return s
            
            json_str = balance_braces(json_str)
            
            # Validate JSON structure before parsing
            def validate_json_structure(s):
                # Ensure all arrays have at least one element or are empty
                s = re.sub(r'\[\s*\]', '[]', s)
                s = re.sub(r'\[\s*,', '[', s)
                s = re.sub(r',\s*\]', ']', s)
                
                # Ensure all objects have valid key-value pairs
                s = re.sub(r'\{\s*\}', '{}', s)
                s = re.sub(r'\{\s*,', '{', s)
                s = re.sub(r',\s*\}', '}', s)
                
                return s
            
            json_str = validate_json_structure(json_str)
            
            # Log the processed JSON string
            logger.debug(f"Processed JSON string: {json_str}")
            
            try:
                # Try to parse the JSON
                parsed = json.loads(json_str)
                return json.dumps(parsed, ensure_ascii=False)  # Added ensure_ascii=False
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {str(e)}")
                
                # Try one more time with additional cleaning
                json_str = re.sub(r'([{,]\s*)([^"\s{},\[\]]+)\s*:', r'\1"\2":', json_str)
                json_str = re.sub(r'\\([^"\\\/bfnrt])', r'\1', json_str)
                json_str = re.sub(r'"\s+"', '","', json_str)  # Fix missing commas between strings
                json_str = re.sub(r'}\s*{', '},{', json_str)  # Fix missing commas between objects
                json_str = re.sub(r']\s*\[', '],[', json_str)  # Fix missing commas between arrays
                json_str = re.sub(r':\s*(?=[,}])', ': null', json_str)  # Add null for missing values
                
                # Validate structure again
                json_str = validate_json_structure(json_str)
                
                # Try parsing again with ensure_ascii=False
                return json.dumps(json.loads(json_str), ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            logger.debug(f"Problematic text: {text[:500]}...")
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
                "You are an expert technical recruiter. Your task is to analyze the resume "
                "and return ONLY a valid JSON object. Follow these strict formatting rules:\n"
                "1. Use double quotes for ALL property names and string values\n"
                "2. Always include commas between array elements and object properties\n"
                "3. Format numbers as integers (no decimal points)\n"
                "4. Use [] for empty arrays, never null\n"
                "5. Include all required fields, even if empty\n"
                "6. Do not include any text before or after the JSON object\n"
                "7. Do not use markdown formatting or code blocks\n"
                "8. Ensure all arrays and objects are properly closed\n"
                "9. Use proper JSON boolean values (true/false) and null"
            )
            
            user_prompt = f"""Analyze technical qualifications for {role_name} and return a JSON object exactly matching this structure:

{{
    "technical_match_score": {technical_score},
    "skills_assessment": [
        {{
            "skill": "Python",
            "proficiency": "Expert",
            "years": 5
        }}
    ],
    "technical_gaps": [
        "Missing skill 1",
        "Missing skill 2"
    ],
    "interview_questions": [
        "Question 1",
        "Question 2"
    ],
    "recommendation": "{recommendation}",
    "key_findings": [
        "Finding 1",
        "Finding 2"
    ],
    "concerns": [
        "Concern 1",
        "Concern 2"
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

Remember: Return ONLY the JSON object with proper formatting."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Log the prompts for debugging
            logger.debug(f"System prompt: {system_prompt}")
            logger.debug(f"User prompt: {user_prompt}")
            
            response = self.execute_request(messages)
            content = response.choices[0].message.content
            
            # Log the raw response
            logger.debug(f"Raw LLM response: {content}")
            
            # Extract and clean JSON
            json_str = self._extract_json_from_text(content)
            result = json.loads(json_str)
            
            # Force the technical score and recommendation
            result['technical_match_score'] = technical_score
            result['recommendation'] = recommendation
            
            # Validate required fields and ensure proper types
            required_fields = {
                "technical_match_score": int,
                "skills_assessment": list,
                "technical_gaps": list,
                "interview_questions": list,
                "recommendation": str,
                "key_findings": list,
                "concerns": list
            }
            
            for field, field_type in required_fields.items():
                if field not in result:
                    if field_type == list:
                        result[field] = []
                    elif field_type == int:
                        result[field] = 0
                    else:
                        result[field] = ""
                elif not isinstance(result[field], field_type):
                    if field_type == list:
                        result[field] = [result[field]] if result[field] else []
                    elif field_type == int:
                        try:
                            result[field] = int(result[field])
                        except (ValueError, TypeError):
                            result[field] = 0
                    else:
                        result[field] = str(result[field])
            
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
                "Format your response EXACTLY like this example with quoted property names:\n"
                '{\n'
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

            # Updated safety settings to be more permissive while maintaining professionalism
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_DEROGATORY",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_TOXICITY",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_VIOLENCE",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUAL",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_MEDICAL",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]

            # Make the API call
            response = self.gemini_client.generate_content(
                contents=prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
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

                # Ensure arrays are never null
                array_fields = ['experience_breakdown', 'experience_flags']
                for field in array_fields:
                    if field not in result or result[field] is None:
                        result[field] = []

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
