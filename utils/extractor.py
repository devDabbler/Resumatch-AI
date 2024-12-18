import PyPDF2
import docx
import fitz
import pdfplumber
import re
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeExtractor:
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx'}

    def extract_text(self, file_bytes: bytes, filename: str) -> Optional[str]:
        """Extract text from PDF or DOCX files."""
        file_ext = self._get_file_extension(filename)
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        try:
            if file_ext == '.pdf':
                return self._extract_from_pdf(file_bytes)
            elif file_ext == '.docx':
                return self._extract_from_docx(file_bytes)
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            return None

    def _get_file_extension(self, filename: str) -> str:
        """Get the lowercase file extension including the dot."""
        return filename[filename.rfind('.'):].lower()

    def _extract_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF using multiple libraries for better results."""
        text = ""
        
        # Try PyMuPDF first
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = " ".join([page.get_text() for page in doc])
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")

        # If PyMuPDF fails or returns empty text, try pdfplumber
        if not text.strip():
            try:
                with pdfplumber.open(file_bytes) as pdf:
                    text = " ".join([page.extract_text() or "" for page in pdf.pages])
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {str(e)}")

        # If both fail, try PyPDF2 as last resort
        if not text.strip():
            try:
                reader = PyPDF2.PdfReader(file_bytes)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {str(e)}")

        if not text.strip():
            raise ValueError("Failed to extract text from PDF using all available methods")

        return self._clean_text(text)

    def _extract_from_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX file."""
        doc = docx.Document(file_bytes)
        text = " ".join([paragraph.text for paragraph in doc.paragraphs])
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and special characters."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep newlines for structure
        text = re.sub(r'[^\w\s\n.,()-@]', '', text)
        return text.strip()
