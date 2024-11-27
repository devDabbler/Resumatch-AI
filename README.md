# Resumatch AI 🎯

Resumatch AI is an intelligent resume analysis tool that leverages advanced AI models to streamline the recruitment process. It helps recruiters quickly calibrate resumes against job requirements, providing deep candidate insights and customized interview questions.

## Features ✨

- **Smart Resume Analysis**: Utilizes Mixtral AI and Google's Gemini for comprehensive resume evaluation
- **Multi-Resume Processing**: Analyze up to 5 resumes simultaneously
- **Stack Ranking**: Automatically ranks candidates based on job fit
- **Custom Interview Questions**: Generates role-specific interview questions
- **Skill Matching**: Advanced pattern matching for required and preferred skills
- **Experience Analysis**: Smart detection of relevant work experience
- **Sentiment Analysis**: Goes beyond keyword matching to understand context
- **Instant Insights**: Quick analysis with clear visual feedback

## Technology Stack 🛠️

- **Frontend**: Streamlit
- **AI/ML**:
  - Mixtral 8x7B (via Groq)
  - Google Gemini
- **PDF Processing**: PyMuPDF (fitz)
- **Data Processing**:
  - YAML for configuration
  - Regex for pattern matching
- **Environment**: Python with dotenv

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resumatch-ai.git
cd resumatch-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage 💡

1. Start the application:
```bash
streamlit run app.py
```

2. Navigate to the Resume Evaluation page

3. Select a job role from the available options:
   - Data Scientist
   - Software Development Engineer
   - Full Stack Software Engineer
   - UI Designer
   - Engagement Manager
   - ML Ops Engineer
   - Solutions Architect
   - AWS Platform Data Engineer
   - Site Reliability Engineer
   - Generative AI Architect

4. Upload up to 5 resumes in PDF format

5. Click "Analyze Resumes" to get detailed insights:
   - Technical match score
   - Key findings
   - Suggested interview questions
   - Areas of concern
   - Stack ranking (when analyzing multiple resumes)

## Project Structure 📁

```
resumatch-ai/
├── app.py              # Main Streamlit application
├── config/
│   └── jobs.yaml       # Job roles and scoring configuration
├── utils/
│   ├── llm.py         # AI model integration
│   ├── matcher.py     # Pattern matching and scoring
│   └── pdf.py         # PDF processing utilities
├── requirements.txt    # Project dependencies
└── .env               # Environment variables
```

## Key Features Explained 🔍

### Smart Calibration
- Advanced pattern recognition for skill matching
- Context-aware experience analysis
- Comprehensive scoring system based on role requirements

### Instant Screening
- Quick processing of multiple resumes
- Clear visual presentation of results
- Stack ranking for easy comparison

### Smart Insights
- AI-generated interview questions
- Detailed technical gap analysis
- Role-specific recommendations

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

[MIT License](LICENSE)

---

Built with ❤️ using Mixtral AI and Google Gemini
