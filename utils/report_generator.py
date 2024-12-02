from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12
        ))

    def _create_score_table(self, score: int) -> Table:
        """Create a table displaying the match score"""
        data = [['Technical Match Score', f'{score}%']]
        table = Table(data, colWidths=[4*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table

    def generate_report(self, analysis_results: Dict) -> bytes:
        """Generate a PDF report from the analysis results"""
        buffer = io.BytesIO()
        logger.info("Starting PDF report generation")
        
        try:
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            logger.info("Created PDF document template")

            # Build the story (content) of the PDF
            story = []

            # Title
            logger.info("Adding title section")
            story.append(Paragraph("Resume Analysis Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))

            # Match Score
            logger.info("Adding match score section")
            story.append(Paragraph("Match Score", self.styles['SectionHeader']))
            story.append(self._create_score_table(analysis_results['technical_match_score']))
            story.append(Spacer(1, 20))

            # Recommendation
            logger.info("Adding recommendation section")
            story.append(Paragraph("Recommendation", self.styles['SectionHeader']))
            story.append(Paragraph(analysis_results['recommendation'], self.styles['Normal']))
            story.append(Spacer(1, 20))

            # Key Findings
            logger.info("Adding key findings section")
            story.append(Paragraph("Key Findings", self.styles['SectionHeader']))
            for finding in analysis_results['key_findings']:
                story.append(Paragraph(f"• {finding}", self.styles['Normal']))
            story.append(Spacer(1, 20))

            # Interview Questions
            logger.info("Adding interview questions section")
            story.append(Paragraph("Recommended Interview Questions", self.styles['SectionHeader']))
            for question in analysis_results['interview_questions']:
                story.append(Paragraph(f"• {question}", self.styles['Normal']))
            story.append(Spacer(1, 20))

            # Concerns
            if analysis_results.get('concerns'):
                logger.info("Adding concerns section")
                story.append(Paragraph("Potential Concerns", self.styles['SectionHeader']))
                for concern in analysis_results['concerns']:
                    story.append(Paragraph(f"• {concern}", self.styles['Normal']))

            # Build the PDF
            logger.info("Building final PDF document")
            doc.build(story)
            logger.info("Successfully generated PDF report")
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
            raise
        finally:
            buffer.close()
