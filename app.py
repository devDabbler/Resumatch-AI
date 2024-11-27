import streamlit as st
import yaml
import json
import logging
from utils.pdf import PDFProcessor
from utils.matcher import JobMatcher
from utils.llm import LLMAnalyzer
from typing import Dict
from dotenv import load_dotenv
from utils.logging_config import setup_logging

# Initialize logging configuration
setup_logging()
logger = logging.getLogger('app')

# Load environment variables
load_dotenv()

def load_config() -> Dict:
    """Load job configuration"""
    try:
        with open('config/jobs.yaml', 'r') as file:
            config = yaml.safe_load(file)
            logger.info("Successfully loaded job configuration")
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def display_results(analysis_results: Dict, rank: int = None):
    """Display analysis results in a structured format"""
    try:
        # Ensure we have a valid dictionary
        if not isinstance(analysis_results, dict):
            logger.error("Invalid analysis results format")
            st.error("Invalid analysis results format")
            st.json(analysis_results)
            return

        # Validate required fields
        required_fields = [
            "recommendation", "technical_match_score", "key_findings",
            "interview_questions", "concerns"
        ]
        missing_fields = [
            field for field in required_fields if field not in analysis_results
        ]
        if missing_fields:
            logger.error(f"Missing required fields in analysis results: {missing_fields}")
            st.error(
                "Missing required fields in analysis results: "
                f"{', '.join(missing_fields)}"
            )
            st.json(analysis_results)
            return

        # Display rank if provided
        if rank is not None:
            st.markdown(
                "<h2 style='text-align: center; font-size: 2.2em;'>"
                f"🏆 Rank #{rank}</h2>",
                unsafe_allow_html=True
            )

        # Initialize recommendation color
        recommendation_color = "#ffc107"  # Default warning yellow

        # Display recommendation with clear visual indicator
        recommendation = analysis_results["recommendation"]
        logger.info(f"Displaying results with recommendation: {recommendation}")
        
        if recommendation == "STRONG_MATCH":
            st.success("✅ Strong Match - Proceed with Recruiter Screen and Fast Track to Hiring Team")
            recommendation_color = "#28a745"  # Bootstrap success green
        elif recommendation == "GOOD_MATCH":
            st.success("👍 Good Match - Proceed with Recruiter Screen")
            recommendation_color = "#28a745"  # Bootstrap success green
        elif recommendation == "POTENTIAL_MATCH":
            st.info("🤔 Potential Match - Additional Screening Required")
            recommendation_color = "#17a2b8"  # Bootstrap info blue
        else:
            st.warning("⚠️ Not a Match - Do Not Proceed")

        # Display match score with visual meter
        score = analysis_results['technical_match_score']
        st.markdown(
            f"""
            <div style='text-align: center; padding: 1.5rem;'>
                <h3 style='color: {recommendation_color}; font-size: 1.8em;'>
                    Match Score: {score}%
                </h3>
                <div style='background: #e9ecef; border-radius: 10px; 
                     height: 24px; width: 100%; max-width: 400px; 
                     margin: 1rem auto;'>
                    <div style='background: {recommendation_color}; 
                         width: {score}%; height: 100%; border-radius: 10px; 
                         transition: width 0.5s ease-in-out;'></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display key findings
        if analysis_results["key_findings"]:
            st.markdown("### 🎯 Key Findings")
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            for finding in analysis_results["key_findings"]:
                st.markdown(
                    "<div style='margin-bottom: 0.8rem; font-size: 1.1em;'>"
                    f"✓ {finding}</div>",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Display interview questions
        if analysis_results["interview_questions"]:
            st.markdown("### 💬 Suggested Interview Questions")
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            for i, question in enumerate(analysis_results["interview_questions"], 1):
                st.markdown(
                    f"""
                    <div style='margin-bottom: 1.2rem; padding: 0.8rem;
                         background: #f8f9fa; border-radius: 8px; 
                         font-size: 1.1em;'>
                        <strong>Q{i}.</strong> {question}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Display concerns
        if analysis_results["concerns"]:
            st.markdown("### ⚠️ Areas of Concern")
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            for concern in analysis_results["concerns"]:
                st.markdown(
                    f"""
                    <div style='color: #dc3545; margin-bottom: 0.8rem; 
                         font-size: 1.1em;'>
                        • {concern}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        st.error("Error displaying results")
        st.error(str(e))
        st.json(analysis_results)  # Show raw results for debugging

def home_page():
    """Display the home page with app introduction and CTA"""
    logger.info("Displaying home page")
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='font-size: 3em; margin-bottom: 0.5rem;'>Resumatch AI</h1>
            <p style='color: #6c757d; font-size: 1.2em; margin-bottom: 1rem;'>
                Intelligent Resume Analysis for Modern Recruitment
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main value proposition
    st.markdown(
        """
        <div style='background-color: #f8f9fa; border-radius: 10px; 
             padding: 1.5rem; margin: 0.5rem 0;'>
            <h2 style='text-align: center; font-size: 1.8em; margin-bottom: 1rem;'>
                Why Resumatch AI?
            </h2>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); 
                 gap: 1.5rem; margin-top: 0.5rem;'>
                <div>
                    <h3 style='color: #0066cc; font-size: 1.3em; margin-bottom: 0.5rem;'>
                        🎯 Smart Calibration
                    </h3>
                    <p style='font-size: 1.1em; color: #444;'>
                        Quickly calibrate resumes against job requirements using advanced AI technology for deep candidate insights
                    </p>
                </div>
                <div>
                    <h3 style='color: #0066cc; font-size: 1.3em; margin-bottom: 0.5rem;'>
                        💡 Instant Screening
                    </h3>
                    <p style='font-size: 1.1em; color: #444;'>
                        Get comprehensive analysis and stack ranking in seconds to conduct more focused and valuable recruiter screens
                    </p>
                </div>
                <div>
                    <h3 style='color: #0066cc; font-size: 1.3em; margin-bottom: 0.5rem;'>
                        ⚡ Smart Insights
                    </h3>
                    <p style='font-size: 1.1em; color: #444;'>
                        Help business teams identify top talent with tailored interview questions and detailed evaluations
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Powered by section
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0; margin-top: 0.5rem;'>
            <h3 style='font-size: 2em; margin-bottom: 1rem;'>Powered By</h3>
            <div style='display: flex; justify-content: center; gap: 3rem; 
                 margin-top: 1rem; font-size: 1.4em;'>
                <div>🧠 Mixtral AI from Mistral</div>
                <div>🤖 Gemini from Google</div>
                <div>⚡ Streamlit</div>
            </div>
            <div style='margin-top: 1rem; font-size: 1.1em; color: #6c757d;'>
                Leveraging state-of-the-art AI models for comprehensive resume analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # CTA button
    if st.button("🚀 Get Started!", type="primary", use_container_width=True):
        logger.info("User clicked 'Get Started' button")
        st.session_state.page = "evaluation"
        st.rerun()

def evaluation_page():
    """Display the resume evaluation page"""
    logger.info("Displaying evaluation page")
    
    # Initialize components
    config = load_config()
    pdf_processor = PDFProcessor()
    job_matcher = JobMatcher('config/jobs.yaml')
    llm_analyzer = LLMAnalyzer()

    # Header
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem 0;'>
            <h2 style='font-size: 2.8em;'>📄 Resume Evaluation</h2>
            <p style='color: #6c757d; font-size: 1.4em;'>
                Upload resumes and select a job role for instant AI-powered analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        # Job role selection
        selected_role = st.selectbox(
            "Select Job Role",
            options=list(config['job_roles'].keys())
        )
        logger.info(f"Selected job role: {selected_role}")

        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Upload Resumes (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload up to 5 PDF resumes to analyze"
        )

        if uploaded_files:
            if len(uploaded_files) > 5:
                logger.warning("User attempted to upload more than 5 resumes")
                st.warning(
                    "⚠️ Maximum 5 resumes allowed. Only the first 5 will be analyzed."
                )
                uploaded_files = uploaded_files[:5]
            
            logger.info(f"Successfully uploaded {len(uploaded_files)} resume(s)")
            st.success(f"✅ {len(uploaded_files)} resume(s) uploaded successfully")
            
            if st.button("🔍 Analyze Resumes", type="primary", use_container_width=True):
                logger.info("Starting resume analysis")
                with st.spinner("🔄 Processing resumes..."):
                    try:
                        all_results = []
                        
                        # Process each resume
                        for file in uploaded_files:
                            logger.info(f"Processing resume: {file.name}")
                            # Process PDF
                            resume_text = pdf_processor.extract_text(file)
                            resume_sections = pdf_processor.extract_sections(
                                resume_text
                            )

                            # Match patterns
                            experience_matches = job_matcher.extract_experience(
                                resume_sections.get('experience', '')
                            )
                            skill_matches = job_matcher.match_skills(
                                resume_text,
                                selected_role
                            )

                            # LLM Analysis
                            analysis_results = llm_analyzer.analyze_resume(
                                resume_text,
                                selected_role,  # Pass role name instead of config
                                skill_matches,
                                experience_matches['matches'] if experience_matches else []
                            )
                            
                            all_results.append({
                                'filename': file.name,
                                'results': analysis_results
                            })

                        # Sort results by confidence score
                        all_results.sort(
                            key=lambda x: x['results']['technical_match_score'],
                            reverse=True
                        )
                        logger.info("Successfully analyzed all resumes")

                        # Display results in the second column
                        with col2:
                            st.markdown(
                                "<h3 style='text-align: center; font-size: 2.4em;'>"
                                "📊 Analysis Results</h3>",
                                unsafe_allow_html=True
                            )
                                
                            if len(all_results) > 1:
                                st.markdown(
                                    "<h4 style='text-align: center; "
                                    "font-size: 1.8em;'>"
                                    "🏆 Stack Ranked Results</h4>",
                                    unsafe_allow_html=True
                                )
                                
                            # Display each result
                            for rank, result in enumerate(all_results, 1):
                                st.markdown(
                                    f"## 📄 {result['filename']}"
                                )
                                display_results(
                                    result['results'],
                                    rank if len(all_results) > 1 else None
                                )
                                st.divider()  # Add visual separator between results

                    except Exception as e:
                        logger.error(f"Analysis failed: {str(e)}")
                        st.error("❌ An error occurred during analysis")
                        if "GROQ_API_KEY" in str(e):
                            st.error(
                                "🔑 Please make sure you have set up your "
                                "GROQ_API_KEY in the .env file."
                            )
                        else:
                            st.error(f"Error details: {str(e)}")
                            st.error(
                                "Please try again. If the error persists, "
                                "check the logs for details."
                            )
        else:
            with col2:
                st.markdown(
                    "<div style='text-align: center; font-size: 1.2em;'>"
                    "👈 Please upload resumes and select a job role to begin analysis"
                    "</div>",
                    unsafe_allow_html=True
                )

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
