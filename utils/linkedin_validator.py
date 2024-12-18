from typing import Dict, Tuple, List
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

logger = logging.getLogger(__name__)

class LinkedInValidator:
    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        
    def validate_profile(self, linkedin_url: str, resume_text: str) -> Tuple[bool, List[str]]:
        """
        Validate LinkedIn profile against resume content.
        Returns (is_valid, list_of_discrepancies)
        """
        if not linkedin_url:
            return True, []
            
        if not self._is_valid_linkedin_url(linkedin_url):
            return False, ["Invalid LinkedIn URL format"]
            
        try:
            profile_data = self._scrape_public_profile(linkedin_url)
            return self._compare_with_resume(profile_data, resume_text)
        except Exception as e:
            logger.error(f"LinkedIn validation error: {str(e)}")
            return False, [f"Failed to validate LinkedIn profile: {str(e)}"]
    
    def _is_valid_linkedin_url(self, url: str) -> bool:
        """Check if the URL is a valid LinkedIn profile URL."""
        linkedin_pattern = r'^https?:\/\/(?:www\.)?linkedin\.com\/in\/[\w\-]+\/?$'
        return bool(re.match(linkedin_pattern, url))
    
    def _scrape_public_profile(self, url: str) -> Dict:
        """
        Scrape public LinkedIn profile data.
        Note: Only scrapes publicly available data from profile page.
        """
        driver = webdriver.Chrome(options=self.chrome_options)
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "top-card-layout__title"))
            )
            
            profile_data = {
                'name': self._safe_get_text(driver, ".top-card-layout__title"),
                'headline': self._safe_get_text(driver, ".top-card-layout__headline"),
                'experience': self._get_experience_data(driver),
                'education': self._get_education_data(driver),
                'skills': self._get_skills_data(driver)
            }
            
            return profile_data
            
        finally:
            driver.quit()
    
    def _safe_get_text(self, driver, selector: str) -> str:
        """Safely get text from an element, return empty string if not found."""
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            return element.text.strip()
        except NoSuchElementException:
            return ""
    
    def _get_experience_data(self, driver) -> List[Dict]:
        """Extract experience data from LinkedIn profile."""
        experiences = []
        try:
            exp_section = driver.find_element(By.ID, "experience-section")
            exp_items = exp_section.find_elements(By.CLASS_NAME, "experience-item")
            
            for item in exp_items:
                exp = {
                    'title': self._safe_get_text(item, ".experience-item__title"),
                    'company': self._safe_get_text(item, ".experience-item__subtitle"),
                    'duration': self._safe_get_text(item, ".experience-item__duration")
                }
                experiences.append(exp)
                
        except NoSuchElementException:
            pass
            
        return experiences
    
    def _get_education_data(self, driver) -> List[Dict]:
        """Extract education data from LinkedIn profile."""
        education = []
        try:
            edu_section = driver.find_element(By.ID, "education-section")
            edu_items = edu_section.find_elements(By.CLASS_NAME, "education__item")
            
            for item in edu_items:
                edu = {
                    'school': self._safe_get_text(item, ".education__school-name"),
                    'degree': self._safe_get_text(item, ".education__item-degree-info"),
                    'duration': self._safe_get_text(item, ".education__item-date-range")
                }
                education.append(edu)
                
        except NoSuchElementException:
            pass
            
        return education
    
    def _get_skills_data(self, driver) -> List[str]:
        """Extract skills from LinkedIn profile."""
        skills = []
        try:
            skills_section = driver.find_element(By.ID, "skills-section")
            skill_items = skills_section.find_elements(By.CLASS_NAME, "skill-item")
            
            for item in skill_items:
                skill_name = self._safe_get_text(item, ".skill-item__name")
                if skill_name:
                    skills.append(skill_name)
                    
        except NoSuchElementException:
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
