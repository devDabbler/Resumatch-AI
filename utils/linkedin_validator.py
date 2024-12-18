from typing import Dict, Tuple, List
import re
import asyncio
from playwright.async_api import async_playwright, Page
import logging

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
            await page.wait_for_selector(".top-card-layout__title")
            
            profile_data = {
                'name': await self._safe_get_text(page, ".top-card-layout__title"),
                'headline': await self._safe_get_text(page, ".top-card-layout__headline"),
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
                return await element.text_content()
            return ""
        except:
            return ""
    
    async def _get_experience_data(self, page: Page) -> List[Dict]:
        """Extract experience data from LinkedIn profile."""
        experiences = []
        try:
            exp_section = await page.wait_for_selector("#experience-section")
            if exp_section:
                exp_items = await exp_section.query_selector_all(".experience-item")
                
                for item in exp_items:
                    exp = {
                        'title': await self._safe_get_text(item, ".experience-item__title"),
                        'company': await self._safe_get_text(item, ".experience-item__subtitle"),
                        'duration': await self._safe_get_text(item, ".experience-item__duration")
                    }
                    experiences.append(exp)
                    
        except:
            pass
            
        return experiences
    
    async def _get_education_data(self, page: Page) -> List[Dict]:
        """Extract education data from LinkedIn profile."""
        education = []
        try:
            edu_section = await page.wait_for_selector("#education-section")
            if edu_section:
                edu_items = await edu_section.query_selector_all(".education__item")
                
                for item in edu_items:
                    edu = {
                        'school': await self._safe_get_text(item, ".education__school-name"),
                        'degree': await self._safe_get_text(item, ".education__item-degree-info"),
                        'duration': await self._safe_get_text(item, ".education__item-date-range")
                    }
                    education.append(edu)
                    
        except:
            pass
            
        return education
    
    async def _get_skills_data(self, page: Page) -> List[str]:
        """Extract skills from LinkedIn profile."""
        skills = []
        try:
            skills_section = await page.wait_for_selector("#skills-section")
            if skills_section:
                skill_items = await skills_section.query_selector_all(".skill-item")
                
                for item in skill_items:
                    skill_name = await self._safe_get_text(item, ".skill-item__name")
                    if skill_name:
                        skills.append(skill_name)
                    
        except:
            pass
            
        return skills
    
    def _compare_with_resume(self, profile_data: Dict, resume_text: str) -> Tuple[bool, List[str]]:
        """Compare LinkedIn profile data with resume content."""
        discrepancies = []
        
        # Compare name
        if profile_data['name']:
            first_line = resume_text.split('\n')[0].lower()
            if profile_data['name'].lower() not in first_line:
                discrepancies.append("Name in LinkedIn profile doesn't match resume")
        
        # Compare experience
        for exp in profile_data['experience']:
            exp_str = f"{exp['title']} {exp['company']}".lower()
            if exp_str not in resume_text.lower():
                discrepancies.append(f"Experience not found in resume: {exp['title']} at {exp['company']}")
        
        # Compare education
        for edu in profile_data['education']:
            edu_str = f"{edu['school']} {edu['degree']}".lower()
            if edu_str not in resume_text.lower():
                discrepancies.append(f"Education not found in resume: {edu['degree']} from {edu['school']}")
        
        # Compare skills (more lenient matching)
        resume_lower = resume_text.lower()
        missing_skills = []
        for skill in profile_data['skills']:
            if skill.lower() not in resume_lower:
                missing_skills.append(skill)
        
        if missing_skills:
            discrepancies.append(f"Skills mentioned in LinkedIn but not in resume: {', '.join(missing_skills)}")
        
        return len(discrepancies) == 0, discrepancies
