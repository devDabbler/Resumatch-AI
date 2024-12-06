from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import io
from typing import Dict
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup enhanced custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            spaceAfter=30,
            textColor=HexColor('#1a237e'),
            alignment=1,  # Center alignment
            leading=32
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=15,
            spaceBefore=20,
            textColor=HexColor('#0d47a1'),
            leading=22
        ))

        # List item style with enhanced spacing
        self.styles.add(ParagraphStyle(
            name='ListItem',
            parent=self.styles['Normal'],
            fontSize=12,
            leftIndent=20,
            spaceAfter=10,
            bulletIndent=10,
            textColor=HexColor('#37474f'),
            leading=16
        ))

        # Score style with larger font
        self.styles.add(ParagraphStyle(
            name='Score',
            parent=self.styles['Normal'],
            fontSize=24,
            alignment=1,
            textColor=HexColor('#1b5e20'),
            leading=28
        ))

    def _create_score_table(self, score: int) -> Table:
        """Create an enhanced table displaying the match score"""
        # Determine color based on score
        if score >= 85:
            color = HexColor('#4caf50')  # Green
        elif score >= 75:
            color = HexColor('#2196f3')  # Blue
        elif score >= 50:
            color = HexColor('#ff9800')  # Orange
        else:
            color = HexColor('#f44336')  # Red

        data = [['Technical Match Score', f'{score}%']]
        table = Table(data, colWidths=[4*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 16),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e0e0e0')),
            ('ROUNDEDCORNERS', [10, 10, 10, 10]),
            ('BOX', (0, 0), (-1, -1), 2, color)
        ]))
        return table

    def _create_recommendation_section(self, recommendation: str) -> Table:
        """Create a styled recommendation section"""
        # Determine style based on recommendation
        if recommendation == "STRONG_MATCH":
            color = HexColor('#4caf50')
            text = "✅ Strong Match - Proceed with Recruiter Screen and Fast Track"
        elif recommendation == "GOOD_MATCH":
            color = HexColor('#2196f3')
            text = "👍 Good Match - Proceed with Recruiter Screen"
        elif recommendation == "POTENTIAL_MATCH":
            color = HexColor('#ff9800')
            text = "🤔 Potential Match - Additional Screening Required"
        else:
            color = HexColor('#f44336')
            text = "⚠️ Not a Match - Do Not Proceed"

        data = [[text]]
        table = Table(data, colWidths=[6*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, -1), color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('ROUNDEDCORNERS', [10, 10, 10, 10]),
            ('BOX', (0, 0), (-1, -1), 2, color)
        ]))
        return table

    def generate_report(self, analysis_results: Dict) -> bytes:
        """Generate an enhanced PDF report that mirrors the Streamlit UI"""
        buffer = io.BytesIO()
        
        try:
            # Create document with custom settings
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=50,
                leftMargin=50,
                topMargin=50,
                bottomMargin=50
            )

            story = []

            # Header with gradient background
            header_style = ParagraphStyle(
                'Header',
                parent=self.styles['CustomTitle'],
                fontSize=28,
                textColor=HexColor('#ffffff'),
                alignment=1,
                spaceAfter=20,
                backColor=HexColor('#1a237e')
            )
            
            story.append(Paragraph(
                """
                <para backColor="#1a237e" textColor="#ffffff">
                    🤖 Resumatch AI Analysis Report
                </para>
                """, 
                header_style
            ))
            story.append(Spacer(1, 20))

            # Match Score section with enhanced styling
            story.append(self._create_score_table(analysis_results['technical_match_score']))
            story.append(Spacer(1, 20))

            # Recommendation section with icon
            story.append(self._create_recommendation_section(analysis_results['recommendation']))
            story.append(Spacer(1, 20))

            # Key Findings section with enhanced styling
            if analysis_results['key_findings']:
                story.append(Paragraph("🎯 Key Findings", self.styles['SectionHeader']))
                for finding in analysis_results['key_findings']:
                    story.append(
                        Paragraph(
                            f"<bullet>•</bullet> {finding}",
                            self.styles['ListItem']
                        )
                    )
                story.append(Spacer(1, 15))

            # Skills Assessment section with modern table
            if analysis_results.get('skills_assessment'):
                story.append(Paragraph("💡 Skills Assessment", self.styles['SectionHeader']))
                skills_data = [[
                    Paragraph("<b>Skill</b>", self.styles['ListItem']),
                    Paragraph("<b>Proficiency</b>", self.styles['ListItem']),
                    Paragraph("<b>Years</b>", self.styles['ListItem'])
                ]]
                for skill in analysis_results['skills_assessment']:
                    skills_data.append([
                        Paragraph(skill['skill'], self.styles['ListItem']),
                        Paragraph(skill['proficiency'], self.styles['ListItem']),
                        Paragraph(str(skill['years']), self.styles['ListItem'])
                    ])
                
                skills_table = Table(skills_data, colWidths=[2.5*inch, 2*inch, 1*inch])
                skills_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f5f5f5')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1a237e')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 1, HexColor('#e0e0e0')),
                    ('PADDING', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ffffff')),
                    ('BOX', (0, 0), (-1, -1), 2, HexColor('#1a237e'))
                ]))
                story.append(skills_table)
                story.append(Spacer(1, 20))

            # Interview Questions section with enhanced styling
            if analysis_results['interview_questions']:
                story.append(Paragraph("💬 Recommended Interview Questions", self.styles['SectionHeader']))
                for i, question in enumerate(analysis_results['interview_questions'], 1):
                    story.append(
                        Paragraph(
                            f"<b>Q{i}.</b> {question}",
                            self.styles['ListItem']
                        )
                    )
                story.append(Spacer(1, 20))

            # Concerns section with warning styling
            if analysis_results.get('concerns'):
                concerns_style = ParagraphStyle(
                    'Concerns',
                    parent=self.styles['ListItem'],
                    textColor=HexColor('#dc3545')
                )
                story.append(Paragraph("⚠️ Potential Concerns", self.styles['SectionHeader']))
                for concern in analysis_results['concerns']:
                    story.append(
                        Paragraph(
                            f"<bullet>•</bullet> {concern}",
                            concerns_style
                        )
                    )
                story.append(Spacer(1, 20))

            # Add "Powered By" section
            story.append(Paragraph("Powered By", self.styles['SectionHeader']))
            powered_by_data = [
                ["🧠 Mixtral AI", "🤖 Gemini Pro", "⚡ Groq"],
                ["Advanced LLM", "Google AI", "Fast Inference"]
            ]
            powered_by_table = Table(powered_by_data, colWidths=[2*inch, 2*inch, 2*inch])
            powered_by_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f8f9fa')),
                ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#1a237e')),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('PADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#e0e0e0'))
            ]))
            story.append(powered_by_table)

            # Footer with timestamp
            footer_style = ParagraphStyle(
                'Footer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=HexColor('#666666'),
                alignment=1
            )
            story.append(Spacer(1, 30))
            story.append(Paragraph(
                f"Generated by Resumatch AI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                footer_style
            ))

            # Build the PDF
            doc.build(story)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
            raise
        finally:
            buffer.close()
