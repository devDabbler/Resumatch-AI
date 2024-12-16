import streamlit as st
import yaml
import json
import logging
import traceback
from utils.pdf import PDFProcessor, DOCXProcessor
from utils.matcher import JobMatcher
from utils.llm import LLMAnalyzer
from utils.report_generator import ReportGenerator
from utils.schemas import AnalysisResult
from utils.parallel_processor import ParallelProcessor
from typing import Dict, Any
from dotenv import load_dotenv
from utils.logging_config import setup_logging
from pathlib import Path
from functools import partial

# Initialize logging configuration
setup_logging()
logger = logging.getLogger('app')

# Load environment variables
load_dotenv()

# Initialize global components for parallel processing
pdf_processor = PDFProcessor()
docx_processor = DOCXProcessor()
job_matcher = JobMatcher('config/jobs.yaml')
llm_analyzer = LLMAnalyzer()

def load_config() -> Dict:
    """Load job configuration"""
    try:
        with open('config/jobs.yaml', 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def display_interview_questions(questions):
    """Helper function to display interview questions in a consistent format"""
    for i, question in enumerate(questions, 1):
        if isinstance(question, dict):
            category = question.get('category', 'Technical')
            q_text = question.get('question', '')
            context = question.get('context', '')
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
            
def display_results(analysis_results: AnalysisResult):
    """Display analysis results in a structured format."""
    try:
        # Validate input
        if not isinstance(analysis_results, AnalysisResult):
            logger.error(f"[UI] Invalid analysis_results type: {type(analysis_results)}")
            st.error("Invalid analysis results")
            return

        # Display match score and recommendation
        try:
            score = analysis_results.technical_match_score
            recommendation = analysis_results.recommendation.value
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

        except Exception as e:
            logger.error(f"[UI] Error displaying score: {str(e)}")
            st.error("Error displaying match score")

        # Display sections
        try:
            if analysis_results.key_findings:
                st.markdown("### 🎯 Key Findings")
                for finding in analysis_results.key_findings:
                    st.markdown(f"• {finding}")

            if analysis_results.skills_assessment:
                st.markdown("### 💡 Skills Assessment")
                skills_data = []
                for skill in analysis_results.skills_assessment:
                    skills_data.append({
                        "Skill": skill.skill,
                        "Level": skill.proficiency,
                        "Years": skill.years
                    })
                if skills_data:
                    st.dataframe(skills_data, use_container_width=True)

            if analysis_results.technical_gaps:
                st.markdown("### ⚠️ Technical Gaps")
                for gap in analysis_results.technical_gaps:
                    st.markdown(f"• {gap}")

            if analysis_results.interview_questions:
                st.markdown("### 💬 Recommended Interview Questions")
                for i, question in enumerate(analysis_results.interview_questions, 1):
                    if isinstance(question, str):
                        st.markdown(f"{i}. {question}")
                    else:  # InterviewQuestion object
                        category = question.category
                        q_text = question.question
                        context = question.context or ""
                        st.markdown(
                            f"""
                            <div style='margin-bottom: 1.2rem; padding: 0.8rem; 
                                  background-color: #f8f9fa; border-radius: 5px;'>
                                <strong>{i}. {category}</strong><br/>
                                {q_text}<br/>
                                <em style='color: #6c757d; font-size: 0.9em;'>{context}</em>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            if analysis_results.concerns:
                st.markdown("### ⚠️ Areas of Concern")
                for concern in analysis_results.concerns:
                    st.markdown(f"• {concern}")

        except Exception as e:
            logger.error(f"[UI] Error displaying sections: {str(e)}")
            st.error("Error displaying analysis sections")

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
        if file_extension == '.pdf':
            text = pdf_processor.extract_text_from_path(str(resume_path))
        elif file_extension in ['.docx', '.doc']:
            text = docx_processor.extract_text(str(resume_path))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Process skills with detailed logging
        try:
            skills = job_matcher.match_skills(text, job_role)
        except Exception as skills_error:
            logger.error(f"Skills matching failed for {filename}: {str(skills_error)}")
            logger.error(f"Skills error traceback: {traceback.format_exc()}")
            skills = {'required': [], 'preferred': [], 'context': {}}
        
        # Process experience with detailed logging
        try:
            experience = job_matcher.extract_experience(text)
        except Exception as exp_error:
            logger.error(f"Experience extraction failed for {filename}: {str(exp_error)}")
            logger.error(f"Experience error traceback: {traceback.format_exc()}")
            experience = {}
        
        # LLM Analysis with detailed logging
        try:
            analysis_dict = llm_analyzer.analyze_resume(text, job_role, skills, experience)
        except Exception as analysis_error:
            logger.error(f"LLM analysis failed for {filename}: {str(analysis_error)}")
            logger.error(f"Analysis error traceback: {traceback.format_exc()}")
            # Create fallback analysis dictionary
            analysis_dict = {
                'technical_match_score': 0,
                'recommendation': 'NO_MATCH',
                'skills_assessment': [],
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
            analysis_result = AnalysisResult(
                technical_match_score=analysis_dict.get('technical_match_score', 0),
                recommendation=analysis_dict.get('recommendation', 'NO_MATCH'),
                skills_assessment=analysis_dict.get('skills_assessment', []),
                technical_gaps=analysis_dict.get('technical_gaps', []),
                interview_questions=analysis_dict.get('interview_questions', []),
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
            
            analysis_result = AnalysisResult(
                technical_match_score=0,
                recommendation="NO_MATCH",
                skills_assessment=[],
                technical_gaps=["Error processing resume"],
                interview_questions=[{
                    'category': 'Technical Implementation',
                    'question': 'Please describe your technical background and experience',
                    'context': 'Fallback question due to processing error'
                }],
                key_findings=["Analysis failed - manual review required"],
                concerns=["Unable to automatically assess qualifications"],
                confidence_score=0.0
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
                    AI-powered skill matching and candidate evaluation
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

    # Call to action button that actually works
    if st.button(
        "🚀 Get Started",
        type="primary",
        use_container_width=True,
        key="get_started_button"
    ):
        st.session_state.page = "evaluation"
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
                                    for field in ['technical_match_score', 'recommendation', 'skills_assessment', 
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

def main():
    """Main application entry point"""
    logger.info("Starting Resumatch AI application")
    st.set_page_config(
        page_title="Resumatch AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    # Navigation in sidebar
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; font-size: 1.6em;'>📍 Navigation</h3>", unsafe_allow_html=True)
        if st.button("🏠 Home", use_container_width=True):
            logger.info("User navigated to Home page")
            st.session_state.page = "home"
            st.rerun()
        if st.button("📄 Resume Evaluation", use_container_width=True):
            logger.info("User navigated to Resume Evaluation page")
            st.session_state.page = "evaluation"
            st.rerun()

    # Display the appropriate page
    if st.session_state.page == "home":
        home_page()
    else:
        evaluation_page()

if __name__ == "__main__":
    main()