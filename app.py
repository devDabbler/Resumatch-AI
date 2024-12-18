import streamlit as st
import yaml
from pathlib import Path
import os
import json
import traceback
from functools import partial
from typing import Dict, Any, List
import logging
from dotenv import load_dotenv
from utils.linkedin_validator import LinkedInValidator
from utils.chat_handler import ChatHandler

# Set page config (must be the first Streamlit command)
st.set_page_config(
    page_title="Resumatch AI",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom CSS to control max width and padding
st.markdown("""
    <style>
        .block-container {
            max-width: 1200px;  /* Increased from 1000px */
            padding-top: 1rem;  /* Reduced from 2rem */
            padding-bottom: 1rem;  /* Reduced from 2rem */
        }
        .stButton > button {
            width: 100%;
        }
        /* Make columns more compact */
        .stColumn {
            padding: 0 1rem;  /* Reduced from 1.5rem */
        }
        /* Adjust container margins */
        .element-container {
            margin: 0.8rem 0;  /* Reduced from 1.5rem */
        }
        /* Make text inputs more compact */
        .stTextInput > div > div > input {
            padding: 0.8rem;  /* Reduced from 1.5rem */
        }
        /* Adjust file uploader size */
        .stFileUploader > div {
            padding: 0.8rem;  /* Reduced from 1.5rem */
        }
        /* Make markdown text more compact */
        .stMarkdown {
            margin: 0.8rem 0;  /* Reduced from 1.5rem */
        }
        /* Adjust header margins */
        h1, h2, h3 {
            margin: 0.8rem 0;  /* Reduced from 1.5rem */
        }
        /* Make success/info/warning messages more compact */
        .stAlert {
            padding: 0.8rem;  /* Reduced from 1.5rem */
            margin: 0.8rem 0;  /* Reduced from 1.5rem */
        }
        /* Increase width of selectbox */
        .stSelectbox {
            min-width: 100%;
        }
        /* Make selectbox container wider */
        [data-testid="stSelectbox"] {
            width: 100%;
        }
        /* Adjust spacing between sections */
        [data-testid="stVerticalBlock"] > div {
            margin-bottom: 0.8rem;  /* Reduced spacing between sections */
        }
    </style>
""", unsafe_allow_html=True)

# Import processors and analyzers
from utils.analyzer import ResumeAnalyzer
from utils.extractor import ResumeExtractor
from utils.matcher import PatternMatcher
from utils.validator import ExperienceValidator
from utils.parallel_processor import ParallelProcessor
from utils.schemas import AnalysisResult, InterviewQuestion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize processors
resume_analyzer = ResumeAnalyzer('config/jobs.yaml')
resume_extractor = ResumeExtractor()
pattern_matcher = PatternMatcher(resume_analyzer.config)
experience_validator = ExperienceValidator(resume_analyzer.config)
linkedin_validator = LinkedInValidator()
chat_handler = ChatHandler('config/jobs.yaml')

# Load environment variables
load_dotenv()

def load_config() -> Dict:
    """Load job configuration"""
    try:
        with open('config/jobs.yaml', 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

# Initialize configuration
CONFIG_PATH = Path("config/jobs.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

def display_interview_questions(questions):
    """Helper function to display interview questions in a consistent format"""
    from utils.schemas import InterviewQuestion
    
    for i, question in enumerate(questions, 1):
        if isinstance(question, (dict, InterviewQuestion)):
            # Get attributes safely whether it's a dict or InterviewQuestion
            if isinstance(question, dict):
                category = question.get('category', 'Technical')
                q_text = question.get('question', '')
                context = question.get('context', '')
            else:  # InterviewQuestion
                category = question.category
                q_text = question.question
                context = question.context
                
            st.markdown(
                f"""
                <div style='margin-bottom: 1.2rem; padding: 0.8rem;
                     background: #f8f9fa; border-radius: 8px; 
                     font-size: 1.1em;'>
                    <strong>Q{i} ({category}):</strong> {q_text}
                    {f"<br><em>Context: {context}</em>" if context else ""}
                </div>
                """,
                unsafe_allow_html=True
            )
        elif isinstance(question, str):
            st.markdown(
                f"""
                <div style='margin-bottom: 1.2rem; padding: 0.8rem;
                     background: #f8f9fa; border-radius: 8px; 
                     font-size: 1.1em;'>
                    <strong>Q{i}:</strong> {question}
                </div>
                """,
                unsafe_allow_html=True
            )

def display_results(result):
    """Display analysis results in a structured format."""
    try:
        # Display match score and recommendation
        score = result.technical_match_score
        recommendation = result.recommendation
        if recommendation == "STRONG_MATCH":
            st.success("✅ Strong Match - Proceed with Recruiter Screen")
            color = "#28a745"
        elif recommendation == "GOOD_MATCH":
            st.success("👍 Good Match - Consider for Interview")
            color = "#28a745"
        elif recommendation == "POTENTIAL_MATCH":
            st.info("🤔 Potential Match - Review Further")
            color = "#17a2b8"
        else:
            st.warning("⚠️ Not a Match")
            color = "#ffc107"

        # Score display
        st.markdown(
            f"""
            <div style='text-align: center; padding: 1rem; margin-bottom: 1rem;'>
                <h2 style='color: {color}; font-size: 2em;'>Match Score: {score}%</h2>
                <div style='background: #e9ecef; border-radius: 10px; height: 20px; width: 100%;'>
                    <div style='background: {color}; width: {score}%; height: 100%; border-radius: 10px;'></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display sections
        if result.key_findings:
            st.markdown("### 🎯 Key Findings")
            for finding in result.key_findings:
                st.markdown(f"• {finding}")

        if result.technical_gaps:
            st.markdown("### ⚠️ Technical Gaps")
            for gap in result.technical_gaps:
                st.markdown(f"• {gap}")

        if result.interview_questions:
            st.markdown("### 💬 Recommended Interview Questions")
            display_interview_questions(result.interview_questions)

        if result.concerns:
            st.markdown("### ⚠️ Areas of Concern")
            for concern in result.concerns:
                st.markdown(f"• {concern}")

    except Exception as e:
        logger.error(f"[UI] Display error: {str(e)}")
        st.error("Failed to display analysis results")

def process_resume(resume_path: Path, job_role: str = None) -> Dict[str, Any]:
    """Process a single resume with enhanced error handling and logging."""
    try:
        if not job_role:
            raise ValueError("No job role selected")
            
        filename = resume_path.name
        file_extension = resume_path.suffix.lower()
        
        # Extract text based on file type
        try:
            if file_extension == '.pdf':
                text = resume_analyzer.extract_text_from_path(str(resume_path))
            elif file_extension in ['.docx', '.doc']:
                text = resume_extractor.extract_text(str(resume_path))
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            if not text or len(text.strip()) == 0:
                raise ValueError("No text could be extracted from the resume")
                
        except Exception as extract_error:
            logger.error(f"Text extraction failed for {filename}: {str(extract_error)}")
            logger.error(f"Extraction error traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to extract text from resume: {str(extract_error)}")
        
        # Process skills with detailed logging
        try:
            skills = pattern_matcher.match_skills(text, job_role)
        except Exception as skills_error:
            logger.error(f"Skills matching failed for {filename}: {str(skills_error)}")
            logger.error(f"Skills error traceback: {traceback.format_exc()}")
            skills = {'required': [], 'preferred': [], 'context': {}}
        
        # Process experience with detailed logging
        try:
            experience = pattern_matcher.extract_experience(text)
        except Exception as exp_error:
            logger.error(f"Experience extraction failed for {filename}: {str(exp_error)}")
            logger.error(f"Experience error traceback: {traceback.format_exc()}")
            experience = {}
        
        # Get LinkedIn URL if provided
        linkedin_url = st.session_state.get('linkedin_url', '')
        
        # Validate LinkedIn profile if URL is provided
        if linkedin_url:
            is_valid, discrepancies = linkedin_validator.validate_profile(linkedin_url, text)
            if not is_valid:
                st.warning("LinkedIn profile validation failed:")
                for discrepancy in discrepancies:
                    st.write(f"- {discrepancy}")
        
        # LLM Analysis with detailed logging
        try:
            analysis_dict = experience_validator.analyze_resume(text, job_role, skills, experience)
        except Exception as analysis_error:
            logger.error(f"LLM analysis failed for {filename}: {str(analysis_error)}")
            logger.error(f"Analysis error traceback: {traceback.format_exc()}")
            # Create fallback analysis dictionary
            analysis_dict = {
                'technical_match_score': 0,
                'recommendation': 'NO_MATCH',
                'technical_gaps': ['Unable to analyze resume'],
                'interview_questions': [{
                    'category': 'Technical Implementation',
                    'question': 'Please describe your technical background and experience',
                    'context': 'Fallback question due to processing error'
                }],
                'key_findings': ['Analysis failed - manual review required'],
                'concerns': ['Unable to automatically assess technical qualifications'],
                'confidence_score': 0.0
            }
        
        # Create AnalysisResult with detailed validation
        try:
            from utils.schemas import InterviewQuestion
            
            # Convert interview questions to proper objects
            interview_questions = []
            for q in analysis_dict.get('interview_questions', []):
                if isinstance(q, dict):
                    interview_questions.append(InterviewQuestion(
                        category=q.get('category', 'Technical'),
                        question=q.get('question', ''),
                        context=q.get('context', '')
                    ))
                elif isinstance(q, InterviewQuestion):
                    interview_questions.append(q)
            
            analysis_result = AnalysisResult(
                technical_match_score=analysis_dict.get('technical_match_score', 0),
                recommendation=analysis_dict.get('recommendation', 'NO_MATCH'),
                technical_gaps=analysis_dict.get('technical_gaps', []),
                interview_questions=interview_questions,
                key_findings=analysis_dict.get('key_findings', []),
                concerns=analysis_dict.get('concerns', []),
                confidence_score=analysis_dict.get('confidence_score', 0.0),
                experience_details=analysis_dict.get('experience_details')
            )
        except Exception as result_error:
            logger.error(f"Failed to create AnalysisResult object for {filename}")
            logger.error(f"Validation error: {str(result_error)}")
            logger.error(f"Validation error traceback: {traceback.format_exc()}")
            logger.error(f"Raw analysis dict: {json.dumps(analysis_dict, indent=2)}")
            
            # Create fallback InterviewQuestion
            fallback_question = InterviewQuestion(
                category='Technical Implementation',
                question='Please describe your technical background and experience',
                context='Fallback question due to processing error'
            )
            
            analysis_result = AnalysisResult(
                technical_match_score=0,
                recommendation="NO_MATCH",
                technical_gaps=["Error processing resume"],
                interview_questions=[fallback_question],
                key_findings=["Analysis failed - manual review required"],
                concerns=["Unable to automatically assess technical qualifications"],
                confidence_score=0.0,
                experience_details=None
            )
        
        return {
            'filename': filename,
            'status': 'success',
            'skills': skills,
            'analysis': analysis_result,
            'text': text
        }
        
    except Exception as e:
        logger.error(f"Critical error processing {resume_path.name}: {str(e)}")
        logger.error(f"Critical error traceback: {traceback.format_exc()}")
        return {
            'filename': resume_path.name if resume_path else 'Unknown',
            'status': 'failed',
            'error': str(e)
        }

def home_page():
    """Display the enhanced home page"""
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem 0;
             background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
             border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='font-size: 3.5em; color: white; margin-bottom: 1rem;'>
                🤖 Resumatch AI
            </h1>
            <p style='color: #e3f2fd; font-size: 1.4em;'>
                Next-Generation Resume Analysis for Modern Recruitment
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Features section with icons
    st.markdown(
        """
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); 
             gap: 2rem; margin: 2rem 0;'>
            <div style='text-align: center; padding: 2rem; 
                 background: #f8f9fa; border-radius: 10px;
                 box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-size: 3em; margin-bottom: 1rem;'>🎯</div>
                <h3 style='color: #1a237e; margin-bottom: 1rem;'>
                    Smart Matching
                </h3>
                <p style='color: #666;'>
                    AI-powered candidate evaluation
                </p>
            </div>
            <div style='text-align: center; padding: 2rem;
                 background: #f8f9fa; border-radius: 10px;
                 box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-size: 3em; margin-bottom: 1rem;'>⚡</div>
                <h3 style='color: #1a237e; margin-bottom: 1rem;'>
                    Instant Analysis
                </h3>
                <p style='color: #666;'>
                    Get comprehensive results in seconds
                </p>
            </div>
            <div style='text-align: center; padding: 2rem;
                 background: #f8f9fa; border-radius: 10px;
                 box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-size: 3em; margin-bottom: 1rem;'>📊</div>
                <h3 style='color: #1a237e; margin-bottom: 1rem;'>
                    Detailed Reports
                </h3>
                <p style='color: #666;'>
                    Generate professional PDF reports instantly
                </p>
            </div>
        </div>

        <!-- Powered By Section -->
        <div style='text-align: center; margin: 3rem 0; padding: 2rem;
             background: #f8f9fa; border-radius: 10px;
             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <h2 style='color: #1a237e; margin-bottom: 2rem; font-size: 2em;'>
                Powered By Advanced AI
            </h2>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;'>
                <div style='text-align: center;'>
                    <div style='font-size: 3em; margin-bottom: 1rem;'>🧠</div>
                    <h3 style='color: #1a237e; margin-bottom: 0.5rem;'>Mixtral AI</h3>
                    <p style='color: #666;'>Advanced LLM for Technical Analysis</p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 3em; margin-bottom: 1rem;'>🤖</div>
                    <h3 style='color: #1a237e; margin-bottom: 0.5rem;'>Gemini Pro</h3>
                    <p style='color: #666;'>Google's AI for Experience Validation</p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 3em; margin-bottom: 1rem;'>⚡</div>
                    <h3 style='color: #1a237e; margin-bottom: 0.5rem;'>Groq</h3>
                    <p style='color: #666;'>Ultra-Fast AI Inference</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Get Started section with options
    st.markdown(
        """
        <div style='text-align: center; margin: 3rem 0;'>
            <h2 style='color: #1a237e; margin-bottom: 2rem;'>Get Started</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "📝 Evaluate Resumes",
            type="primary",
            use_container_width=True,
            key="evaluate_button",
            help="Upload and analyze resumes against job requirements"
        ):
            st.session_state["navigation"] = "Resume Evaluation"
            st.rerun()

    with col2:
        if st.button(
            "💬 Chat with Resumes",
            type="primary",
            use_container_width=True,
            key="chat_button",
            help="Have an interactive conversation about your resumes"
        ):
            st.session_state["navigation"] = "Chat with Resume"
            st.rerun()

    # Add some space at the bottom
    st.markdown("<div style='margin-bottom: 4rem;'></div>", unsafe_allow_html=True)

def evaluation_page():
    """Display the resume evaluation page"""
    try:
        # Initialize components
        config = load_config()
        parallel_processor = ParallelProcessor()

        # Display header
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem; margin-bottom: 2rem; 
                 background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
                 border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-size: 3em; color: white; margin-bottom: 0.5rem;'>
                    🤖 Resumatch AI
                </div>
                <div style='font-size: 1.2em; color: #e3f2fd;'>
                    Intelligent Resume Analysis Powered by AI
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Create columns
        col1, col2 = st.columns([1, 2])

        with col1:
            # Job role selection
            st.markdown("### 🎯 Select Job Role")
            selected_role = st.selectbox(
                "Select Job Role",
                options=list(config['job_roles'].keys()),
                key="role_selector"
            )
            st.session_state['selected_role'] = selected_role

            # LinkedIn URL input
            linkedin_url = st.text_input(
                "LinkedIn Profile URL (optional)",
                help="Enter the candidate's LinkedIn profile URL for additional validation"
            )
            st.session_state['linkedin_url'] = linkedin_url

            # File upload
            st.markdown("### 📎 Upload Resumes")
            uploaded_files = st.file_uploader(
                "Upload Resumes",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                help="Upload up to 5 resumes in PDF or DOCX format"
            )

            if uploaded_files:
                # Process uploaded files
                if len(uploaded_files) > 5:
                    st.warning("️ Maximum 5 resumes allowed. Only the first 5 will be analyzed.")
                    uploaded_files = uploaded_files[:5]

                # Display file list
                st.markdown("#### 📋 Uploaded Files:")
                for file in uploaded_files:
                    icon = "📄" if file.name.endswith('.pdf') else "📝"
                    st.markdown(f"{icon} {file.name}")

                st.success(f"✅ {len(uploaded_files)} resume(s) uploaded successfully")

                # Analyze button
                if st.button("🔍 Analyze Resumes", type="primary", use_container_width=True):
                    logger.info("[UI] Starting resume analysis")
                    with st.spinner("🤖 AI is analyzing your resumes in parallel..."):
                        try:
                            # Save files to temp directory
                            temp_dir = Path("temp_uploads")
                            temp_dir.mkdir(exist_ok=True)
                            
                            # Process files
                            saved_files = []
                            for uploaded_file in uploaded_files:
                                temp_path = temp_dir / uploaded_file.name
                                with open(temp_path, 'wb') as f:
                                    f.write(uploaded_file.getbuffer())
                                saved_files.append(temp_path)

                            # Process resumes
                            processor = ParallelProcessor(max_workers=min(4, len(saved_files)))
                            process_func = partial(process_resume, job_role=selected_role)
                            results, stats = processor.process_batch(saved_files, process_func)

                            logger.info(f"[UI] Got {len(results)} results")

                            # Process results
                            successful_results = []
                            failed_results = []

                            for result in results:
                                try:
                                    logger.info(f"[UI] Processing result for item: {result.get('filename', 'unknown')}")
                                    logger.info(f"Result status: {result.get('status', 'unknown')}")
                                    logger.info(f"Result keys: {list(result.keys())}")
                                    
                                    if not isinstance(result, dict):
                                        logger.error(f"[UI] Invalid result type: {type(result)}")
                                        failed_results.append({'filename': 'Unknown', 'error': 'Invalid result type'})
                                        continue
                                        
                                    if 'analysis' not in result:
                                        logger.error("[UI] No analysis in result")
                                        failed_results.append(result)
                                        continue
                                        
                                    analysis = result['analysis']
                                    if not isinstance(analysis, AnalysisResult):
                                        logger.error(f"[UI] Invalid analysis type: {type(analysis)}")
                                        failed_results.append(result)
                                        continue
                                        
                                    logger.info("Analysis type: %s", type(analysis))
                                    logger.info("Analysis fields:")
                                    for field in ['technical_match_score', 'recommendation',
                                                'technical_gaps', 'interview_questions', 'key_findings', 
                                                'concerns', 'experience_details', 'analysis_timestamp', 
                                                'confidence_score']:
                                        logger.info(f"- {field}: {type(getattr(analysis, field, None))}")
                                    
                                    successful_results.append(result)
                                    logger.info("[UI] Added to successful results")
                                except Exception as e:
                                    logger.error(f"[UI] Error processing result: {str(e)}")
                                    if isinstance(result, dict):
                                        failed_results.append(result)
                                    else:
                                        failed_results.append({'filename': 'Unknown', 'error': str(e)})

                            # Sort results by match score
                            successful_results.sort(
                                key=lambda x: x['analysis'].technical_match_score if x.get('analysis') else 0,
                                reverse=True
                            )

                            # Display results in second column
                            with col2:
                                st.empty()  # Clear previous results
                                
                                if successful_results:
                                    for idx, result in enumerate(successful_results, 1):
                                        with st.container():
                                            st.markdown(f"### Resume {idx}: {result['filename']}")
                                            try:
                                                display_results(result['analysis'])
                                                st.divider()
                                            except Exception as e:
                                                logger.error(f"[UI] Error displaying result {idx}: {str(e)}")
                                                st.error(f"Error displaying result {idx}")
                                else:
                                    st.warning("No results to display")

                                if failed_results:
                                    st.error("⚠️ Failed to analyze some resumes:")
                                    for result in failed_results:
                                        st.warning(f"• {result.get('filename', 'Unknown file')}")

                        except Exception as e:
                            logger.error(f"[UI] Analysis failed: {str(e)}")
                            st.error("Failed to analyze resumes. Please try again.")

    except Exception as e:
        logger.error(f"[UI] Page error: {str(e)}")
        st.error("An error occurred. Please refresh the page and try again.")

def chat_page():
    """Display the chat interface for resume analysis."""
    st.title("Chat with Resume")
    
    # Session state for storing uploaded resumes
    if 'primary_resume' not in st.session_state:
        st.session_state.primary_resume = None
    if 'comparison_resume' not in st.session_state:
        st.session_state.comparison_resume = None
    if 'job_role' not in st.session_state:
        st.session_state.job_role = None
        
    # File upload section
    st.subheader("Upload Resumes")
    col1, col2 = st.columns(2)
    
    with col1:
        primary_file = st.file_uploader("Upload Primary Resume", type=['pdf', 'docx'], key='primary_upload')
        if primary_file:
            # Save the file temporarily
            temp_path = f"temp_{primary_file.name}"
            with open(temp_path, "wb") as f:
                f.write(primary_file.getvalue())
            st.session_state.primary_resume = temp_path
            
    with col2:
        comparison_file = st.file_uploader("Upload Comparison Resume (Optional)", type=['pdf', 'docx'], key='comparison_upload')
        if comparison_file:
            # Save the file temporarily
            temp_path = f"temp_{comparison_file.name}"
            with open(temp_path, "wb") as f:
                f.write(comparison_file.getvalue())
            st.session_state.comparison_resume = temp_path
            
    # Job role selection
    job_roles = list(config['job_roles'].keys())
    selected_role = st.selectbox("Select Job Role", job_roles, key='job_role_selector')
    st.session_state.job_role = selected_role
    
    # Chat interface
    st.subheader("Chat Interface")
    
    if st.session_state.primary_resume:
        try:
            # Extract text from primary resume
            primary_text = resume_extractor.extract_text(st.session_state.primary_resume)
            if not primary_text:
                st.error("Failed to extract text from primary resume")
                return
            
            # Extract text from comparison resume if available
            comparison_text = None
            comparison_name = None
            if st.session_state.comparison_resume:
                comparison_text = resume_extractor.extract_text(st.session_state.comparison_resume)
                if comparison_text:
                    # Extract name from first line
                    comparison_name = comparison_text.split('\n')[0].strip()
            
            # Chat input
            user_query = st.text_input("Ask a question about the resume(s)")
            
            if user_query:
                with st.spinner("Processing your query..."):
                    response = chat_handler.chat(
                        query=user_query,
                        resume_text=primary_text,
                        job_title=st.session_state.job_role,
                        comparison_resume=comparison_text,
                        comparison_name=comparison_name
                    )
                    st.write("Response:", response)
                    
            # Example queries
            st.subheader("Example Queries")
            examples = [
                "What skills are missing in this resume for the current role?",
                "What are the candidate's strongest qualifications?",
                "How many years of relevant experience does the candidate have?",
            ]
            if comparison_text:
                examples.append(f"Compare this candidate with {comparison_name}.")
                
            for example in examples:
                if st.button(example):
                    with st.spinner("Processing your query..."):
                        response = chat_handler.chat(
                            query=example,
                            resume_text=primary_text,
                            job_title=st.session_state.job_role,
                            comparison_resume=comparison_text,
                            comparison_name=comparison_name
                        )
                        st.write("Response:", response)
                        
            # Clear conversation button
            if st.button("Clear Conversation"):
                chat_handler.clear_conversation()
                st.success("Conversation history cleared!")
                
        except Exception as e:
            logger.error(f"Error in chat interface: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            
    else:
        st.info("Please upload a primary resume to start chatting.")
        
    # Cleanup temporary files on session end
    def cleanup_temp_files():
        if st.session_state.primary_resume:
            try:
                os.remove(st.session_state.primary_resume)
            except:
                pass
        if st.session_state.comparison_resume:
            try:
                os.remove(st.session_state.comparison_resume)
            except:
                pass
                
    # Register cleanup
    import atexit
    atexit.register(cleanup_temp_files)

# Rest of the file remains the same...

def main():
    """Main application entry point"""
    # Initialize session state
    if "navigation" not in st.session_state:
        st.session_state["navigation"] = "Home"
    
    pages = {
        "Home": home_page,
        "Resume Evaluation": evaluation_page,
        "Chat with Resume": chat_page
    }
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()), key="nav_radio", index=list(pages.keys()).index(st.session_state["navigation"]))
    
    # Update navigation state if changed
    if selection != st.session_state["navigation"]:
        st.session_state["navigation"] = selection
        st.rerun()
    
    # Display selected page
    pages[st.session_state["navigation"]]()

if __name__ == "__main__":
    main()
