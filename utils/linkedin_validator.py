from typing import Dict, Tuple, List, Optional
import re
import asyncio
from playwright.async_api import async_playwright, Page
import logging
from datetime import datetime
from dateutil import parser
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class LinkedInValidator:
    def __init__(self):
        self.browser = None
        self.context = None
        
    async def __aenter__(self):
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.context.close()
        await self.browser.close()
    
    async def validate_profile(self, linkedin_url: str, resume_text: str) -> Tuple[bool, List[str]]:
        """
        Validate LinkedIn profile against resume content.
        Returns (is_valid, list_of_discrepancies)
        """
        if not linkedin_url:
            return True, []
            
        if not self._is_valid_linkedin_url(linkedin_url):
            return False, ["Invalid LinkedIn URL format"]
            
        try:
            profile_data = await self._scrape_public_profile(linkedin_url)
            return self._compare_with_resume(profile_data, resume_text)
        except Exception as e:
            logger.error(f"LinkedIn validation error: {str(e)}")
            return False, [f"Failed to validate LinkedIn profile: {str(e)}"]
    
    def _is_valid_linkedin_url(self, url: str) -> bool:
        """Check if the URL is a valid LinkedIn profile URL."""
        linkedin_pattern = r'^https?:\/\/(?:www\.)?linkedin\.com\/in\/[\w\-]+\/?$'
        return bool(re.match(linkedin_pattern, url))
    
    async def _scrape_public_profile(self, url: str) -> Dict:
        """
        Scrape public LinkedIn profile data.
        Note: Only scrapes publicly available data from profile page.
        """
        page = await self.context.new_page()
        try:
            await page.goto(url)
            await page.wait_for_selector("h1")  # Wait for name to load
            
            profile_data = {
                'name': await self._safe_get_text(page, "h1"),
                'headline': await self._safe_get_text(page, ".text-body-medium"),
                'experience': await self._get_experience_data(page),
                'education': await self._get_education_data(page),
                'skills': await self._get_skills_data(page)
            }
            
            return profile_data
            
        finally:
            await page.close()
    
    async def _safe_get_text(self, page: Page, selector: str) -> str:
        """Safely get text from an element, return empty string if not found."""
        try:
            element = await page.wait_for_selector(selector, timeout=5000)
            if element:
                return (await element.text_content()).strip()
            return ""
        except Exception as e:
            logger.debug(f"Error getting text for selector {selector}: {str(e)}")
            return ""
    
    async def _get_experience_data(self, page: Page) -> List[Dict]:
        """Extract experience data from LinkedIn profile."""
        experiences = []
        try:
            # Wait for experience section
            exp_section = await page.wait_for_selector('section[id*="experience"]')
            if exp_section:
                # Get all experience entries
                exp_items = await exp_section.query_selector_all("li")
                
                for item in exp_items:
                    # Extract title and company
                    title = await self._safe_get_text(item, "span[aria-hidden='true']")
                    company = await self._safe_get_text(item, "span.t-14")
                    
                    # Extract and parse dates
                    dates_text = await self._safe_get_text(item, "span.t-black--light")
                    start_date, end_date = self._parse_date_range(dates_text)
                    
                    # Extract description if available
                    description = await self._safe_get_text(item, ".experience-item__description")
                    
                    exp = {
                        'title': title,
                        'company': company,
                        'start_date': start_date,
                        'end_date': end_date,
                        'description': description
                    }
                    experiences.append(exp)
                    
        except Exception as e:
            logger.error(f"Error extracting experience data: {str(e)}")
            
        return experiences
    
    async def _get_education_data(self, page: Page) -> List[Dict]:
        """Extract education data from LinkedIn profile."""
        education = []
        try:
            # Wait for education section
            edu_section = await page.wait_for_selector('section[id*="education"]')
            if edu_section:
                # Get all education entries
                edu_items = await edu_section.query_selector_all("li")
                
                for item in edu_items:
                    # Extract school and degree info
                    school = await self._safe_get_text(item, "span[aria-hidden='true']")
                    degree = await self._safe_get_text(item, ".education__item-degree")
                    
                    # Extract and parse dates
                    dates_text = await self._safe_get_text(item, "span.t-black--light")
                    start_date, end_date = self._parse_date_range(dates_text)
                    
                    # Extract field of study if available
                    field = await self._safe_get_text(item, ".education__item-field")
                    
                    edu = {
                        'school': school,
                        'degree': degree,
                        'field': field,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                    education.append(edu)
                    
        except Exception as e:
            logger.error(f"Error extracting education data: {str(e)}")
            
        return education
    
    async def _get_skills_data(self, page: Page) -> List[Dict]:
        """Extract skills and endorsements from LinkedIn profile."""
        skills = []
        try:
            # Wait for skills section
            skills_section = await page.wait_for_selector('section[id*="skills"]')
            if skills_section:
                # Get all skill entries
                skill_items = await skills_section.query_selector_all("li")
                
                for item in skill_items:
                    skill_name = await self._safe_get_text(item, "span[aria-hidden='true']")
                    endorsements = await self._safe_get_text(item, ".skill-endorsements")
                    
                    if skill_name:
                        skills.append({
                            'name': skill_name,
                            'endorsements': self._parse_endorsements(endorsements)
                        })
                    
        except Exception as e:
            logger.error(f"Error extracting skills data: {str(e)}")
            
        return skills
    
    def _parse_date_range(self, date_text: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Parse date range text into start and end dates."""
        try:
            if not date_text:
                return None, None
                
            # Split date range
            parts = date_text.split(' - ')
            if len(parts) != 2:
                return None, None
                
            start_text, end_text = parts
            
            # Parse start date
            try:
                start_date = parser.parse(start_text)
            except:
                start_date = None
            
            # Parse end date
            if end_text.lower().strip() in ['present', 'current']:
                end_date = datetime.now()
            else:
                try:
                    end_date = parser.parse(end_text)
                except:
                    end_date = None
            
            return start_date, end_date
            
        except Exception as e:
            logger.error(f"Error parsing date range '{date_text}': {str(e)}")
            return None, None
    
    def _parse_endorsements(self, endorsement_text: str) -> int:
        """Parse endorsement count from text."""
        try:
            if not endorsement_text:
                return 0
            match = re.search(r'(\d+)', endorsement_text)
            return int(match.group(1)) if match else 0
        except:
            return 0
    
    def _compare_with_resume(self, profile_data: Dict, resume_text: str) -> Tuple[bool, List[str]]:
        """Compare LinkedIn profile data with resume content."""
        discrepancies = []
        
        # Extract resume sections
        resume_sections = self._extract_resume_sections(resume_text)
        
        # Compare name (using fuzzy matching)
        if profile_data['name']:
            first_line = resume_text.split('\n')[0]
            if not self._fuzzy_match(profile_data['name'], first_line):
                discrepancies.append("Name in LinkedIn profile doesn't match resume")
        
        # Compare experience
        resume_exp = resume_sections.get('experience', '')
        for exp in profile_data['experience']:
            # Check if experience exists in resume
            exp_found = False
            exp_str = f"{exp['title']} {exp['company']}".lower()
            
            # Try different variations of company name
            company_variations = self._get_company_variations(exp['company'])
            
            for company in company_variations:
                if self._fuzzy_match(exp_str, resume_exp.lower()):
                    exp_found = True
                    break
            
            if not exp_found:
                discrepancies.append(f"Experience not found in resume: {exp['title']} at {exp['company']}")
                continue
            
            # Compare dates if available
            if exp['start_date'] and exp['end_date']:
                date_str = f"{exp['start_date'].strftime('%Y')} - {exp['end_date'].strftime('%Y')}"
                if date_str not in resume_exp:
                    discrepancies.append(f"Date discrepancy for {exp['title']} at {exp['company']}")
        
        # Compare education
        resume_edu = resume_sections.get('education', '')
        for edu in profile_data['education']:
            # Check if education exists in resume
            edu_found = False
            edu_str = f"{edu['school']} {edu['degree']} {edu['field']}".lower()
            
            if self._fuzzy_match(edu_str, resume_edu.lower()):
                edu_found = True
            
            if not edu_found:
                discrepancies.append(f"Education not found in resume: {edu['degree']} from {edu['school']}")
                continue
            
            # Compare dates if available
            if edu['start_date'] and edu['end_date']:
                date_str = f"{edu['start_date'].strftime('%Y')} - {edu['end_date'].strftime('%Y')}"
                if date_str not in resume_edu:
                    discrepancies.append(f"Date discrepancy for education at {edu['school']}")
        
        # Compare skills
        resume_skills = resume_sections.get('skills', '')
        missing_skills = []
        for skill in profile_data['skills']:
            skill_name = skill['name'].lower()
            if not self._fuzzy_match(skill_name, resume_skills.lower()):
                # Only include highly endorsed skills
                if skill['endorsements'] >= 5:
                    missing_skills.append(skill['name'])
        
        if missing_skills:
            discrepancies.append(f"Skills mentioned in LinkedIn but not in resume: {', '.join(missing_skills)}")
        
        return len(discrepancies) == 0, discrepancies
    
    def _extract_resume_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from resume text."""
        sections = {}
        
        # Common section headers
        headers = {
            'experience': r'(?i)(work\s+experience|professional\s+experience|employment|experience)',
            'education': r'(?i)(education|academic|qualification)',
            'skills': r'(?i)(skills|technical\s+skills|expertise|competencies)'
        }
        
        # Find each section
        for section, pattern in headers.items():
            match = re.search(f"{pattern}.*?(?={pattern}|$)", text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[section] = match.group(0)
        
        return sections
    
    def _fuzzy_match(self, str1: str, str2: str, threshold: float = 0.85) -> bool:
        """Use fuzzy string matching to compare strings."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio() >= threshold
    
    def _get_company_variations(self, company: str) -> List[str]:
        """Generate variations of company names."""
        variations = [company]
        
        # Remove common suffixes
        suffixes = [' Inc.', ' LLC', ' Ltd.', ' Limited', ' Corp.', ' Corporation']
        base = company
        for suffix in suffixes:
            if company.endswith(suffix):
                base = company[:-len(suffix)]
                variations.append(base)
                break
        
        # Add variations with/without spaces
        if ' ' in base:
            variations.append(base.replace(' ', ''))
        
        return variations
