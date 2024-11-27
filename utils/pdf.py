import fitz
import logging
from typing import Dict


logger = logging.getLogger(__name__)


class PDFProcessor:
    @staticmethod
    def extract_text(pdf_file) -> str:
        """
        Extract text from uploaded PDF with error handling
        
        Args:
            pdf_file: StreamIO object containing PDF data
            
        Returns:
            str: Extracted text from PDF
            
        Raises:
            ValueError: If PDF is corrupted or cannot be processed
        """
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            raise ValueError(
                "Unable to process PDF file. Please ensure it is not corrupted."
            ) from e

    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """
        Extract key resume sections with improved detection
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Dict containing extracted sections
        """
        sections = {
            'experience': '',
            'education': '',
            'skills': ''
        }
        
        # Enhanced section detection patterns
        section_keywords = {
            'experience': [
                'experience', 'work history', 'employment', 'work experience'
            ],
            'education': [
                'education', 'academic background', 'qualifications',
                'academic history'
            ],
            'skills': [
                'skills', 'technical skills', 'core competencies', 'expertise'
            ]
        }
        
        try:
            lines = text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                lower_line = line.lower()
                
                # Check for section headers
                for section, keywords in section_keywords.items():
                    if (any(keyword in lower_line for keyword in keywords) and
                            not line.strip().islower()):
                        current_section = section
                        break
                        
                # Add content to current section
                if current_section and line:
                    # Don't add the section header itself
                    if not any(
                        keyword in lower_line 
                        for keyword in section_keywords[current_section]
                    ):
                        sections[current_section] = (
                            sections[current_section] + line + '\n'
                        )
            
            # Clean up trailing whitespace
            for section in sections:
                sections[section] = sections[section].strip()
            
            logger.debug(f"Extracted sections: {list(sections.keys())}")
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting sections: {str(e)}")
            return sections  # Return empty sections rather than failing
