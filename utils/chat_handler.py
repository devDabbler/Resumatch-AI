import os
from typing import Dict, List, Optional
import yaml
import logging
from pathlib import Path
import requests
import json

logger = logging.getLogger(__name__)

class ChatHandler:
    def __init__(self, jobs_yaml_path: str):
        """Initialize the chat handler with the jobs YAML file path."""
        self.api_key = os.getenv('LLAMA_API_KEY')
        if not self.api_key:
            raise ValueError("LLAMA_API_KEY environment variable is not set")
            
        self.api_url = "https://api.llama-ai.com/v1/chat/completions"  # Replace with actual Llama API endpoint
        self.jobs_data = self._load_jobs_yaml(jobs_yaml_path)
        self.conversation_history = []
        
    def _load_jobs_yaml(self, yaml_path: str) -> Dict:
        """Load job requirements and skills from YAML file."""
        try:
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading jobs YAML: {str(e)}")
            return {}
            
    def _get_job_requirements(self, job_title: str) -> Dict:
        """Get requirements for a specific job title."""
        return self.jobs_data.get(job_title, {})
        
    def _build_system_prompt(self, resume_text: str, job_title: str) -> str:
        """Build system prompt with resume content and job requirements."""
        job_reqs = self._get_job_requirements(job_title)
        return f"""You are an AI assistant helping recruiters evaluate resumes. 
        You have access to the following information:
        
        Resume Content:
        {resume_text}
        
        Job Requirements:
        {yaml.dump(job_reqs, default_flow_style=False)}
        
        Base your responses on this information and provide specific, factual answers.
        Do not make assumptions about information not present in the resume or job requirements."""
        
    def _build_chat_prompt(self, query: str, resume_text: str, job_title: str) -> str:
        """Build the complete chat prompt including conversation history."""
        system_prompt = self._build_system_prompt(resume_text, job_title)
        
        # Format conversation history
        history = ""
        for msg in self.conversation_history:
            history += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
            
        return f"{system_prompt}\n\nConversation History:\n{history}\n\nUser: {query}\nAssistant:"
        
    def chat(self, query: str, resume_text: str, job_title: str, 
             comparison_resume: Optional[str] = None, 
             comparison_name: Optional[str] = None) -> str:
        """
        Process a chat query about a resume.
        """
        try:
            # Handle comparison queries
            if comparison_resume and comparison_name and "compare" in query.lower():
                system_prompt = f"""You are an AI assistant helping recruiters compare two candidates.
                
                Candidate 1 Resume:
                {resume_text}
                
                Candidate 2 ({comparison_name}) Resume:
                {comparison_resume}
                
                Job Requirements:
                {yaml.dump(self._get_job_requirements(job_title), default_flow_style=False)}
                
                Compare these candidates objectively based on their qualifications and the job requirements.
                """
                messages = [
                    {"role": "system", "content": self._build_system_prompt(resume_text, job_title)}
                ]

                # Add conversation history
                for msg in self.conversation_history:
                    messages.extend([
                        {"role": "user", "content": msg["user"]},
                        {"role": "assistant", "content": msg["assistant"]}
                    ])

                # Add current query
                messages.append({"role": "user", "content": query})

            # Make API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.5
            }
            
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            # Store conversation history
            self.conversation_history.append({
                'user': query,
                'assistant': answer
            })
            
            # Keep only last 5 exchanges to maintain context without overflow
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]
                
            return answer
            
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            return "I apologize, but I encountered an error while processing your query. Please try again."
            
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
