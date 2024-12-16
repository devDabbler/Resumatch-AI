from typing import Dict, List, Set, Optional, Any
import regex as re
from collections import defaultdict
import logging
from utils.fuzzy_matcher import FuzzyMatcher, FuzzyMatch

logger = logging.getLogger('pattern_matcher')

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.data = None
        self.case_insensitive_pattern = None

class PatternTrie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, pattern: str, data: Any = None):
        """Insert a pattern into the trie."""
        node = self.root
        pattern_lower = pattern.lower()
        
        for char in pattern_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
        node.is_end = True
        node.data = data
        # Create case-insensitive pattern for partial matches
        node.case_insensitive_pattern = re.compile(
            f"\\b{re.escape(pattern)}\\b", 
            re.IGNORECASE
        )

    def search(self, text: str) -> Set[tuple]:
        """
        Search for all patterns in the text.
        Returns set of (pattern, data) tuples.
        """
        results = set()
        text_lower = text.lower()
        
        def search_from_node(node: TrieNode, start_pos: int):
            if node.is_end:
                # Verify with regex for word boundaries
                if node.case_insensitive_pattern.search(
                    text[max(0, start_pos-1):start_pos+1]
                ):
                    results.add((text[start_pos-len(node.data):start_pos], node.data))
            
            if start_pos >= len(text_lower):
                return
                
            char = text_lower[start_pos]
            if char in node.children:
                search_from_node(node.children[char], start_pos + 1)
        
        # Start search from each position
        for i in range(len(text)):
            search_from_node(self.root, i)
            
        return results

class OptimizedPatternMatcher:
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.skill_trie = PatternTrie()
        self.company_trie = PatternTrie()
        self.school_trie = PatternTrie()
        self.compiled_patterns = {}
        self.context_window = self.config.get('context_window', 100)
        
        # Initialize fuzzy matcher with config
        self.fuzzy_matcher = FuzzyMatcher(config)
        
        # Store original patterns for matching
        self.skill_patterns = {}
        self.company_patterns = {}
        self.school_patterns = {}
        
        # Load scoring adjustments
        self.scoring_adjustments = self.config.get('scoring_adjustments', {
            'exact_match_bonus': 0.2,
            'fuzzy_match_penalty': 0.1,
            'phonetic_match_bonus': 0.15,
            'context_relevance_bonus': 0.1
        })

    def adjust_score_by_context(
        self,
        match: FuzzyMatch,
        text: str
    ) -> float:
        """Adjust match score based on surrounding context."""
        score = match.score
        context = match.context.lower()
        
        # Check for relevant context keywords
        context_keywords = self.config.get('context_keywords', {
            'positive': ['senior', 'expert', 'lead', 'advanced'],
            'negative': ['basic', 'beginner', 'learning']
        })
        
        # Adjust score based on context
        for keyword in context_keywords['positive']:
            if keyword in context:
                score += self.scoring_adjustments['context_relevance_bonus']
                
        for keyword in context_keywords['negative']:
            if keyword in context:
                score -= self.scoring_adjustments['context_relevance_bonus']
        
        return min(1.0, max(0.0, score))

    def build_tries(self, config: Dict[str, Any]):
        """Build tries and store patterns for fuzzy matching."""
        # Build skill trie and patterns
        skills = config.get('skills', {})
        skill_variations = config.get('skill_variations', {})
        
        for skill, data in skills.items():
            self.skill_trie.insert(skill, data)
            self.skill_patterns[skill] = data
            # Add variations
            variations = skill_variations.get(skill, [])
            for variation in variations:
                self.skill_trie.insert(variation, data)
                self.skill_patterns[variation] = data

        # Build company patterns
        companies = config.get('companies', [])
        for company in companies:
            self.company_trie.insert(company, company)
            self.company_patterns[company] = {'type': 'company', 'name': company}

        # Build school patterns
        schools = config.get('schools', [])
        for school in schools:
            self.school_trie.insert(school, school)
            self.school_patterns[school] = {'type': 'school', 'name': school}

    def compile_patterns(self, patterns: Dict[str, List[str]]):
        """Pre-compile regex patterns for better performance."""
        for category, pattern_list in patterns.items():
            if isinstance(pattern_list, list):
                # Combine patterns into a single regex where possible
                combined_pattern = '|'.join(
                    f'({pattern})' for pattern in pattern_list if pattern.strip()
                )
                try:
                    self.compiled_patterns[category] = re.compile(
                        combined_pattern,
                        re.IGNORECASE
                    )
                except Exception as e:
                    logger.error(f"Failed to compile patterns for {category}: {str(e)}")
                    # Fall back to individual patterns
                    self.compiled_patterns[category] = [
                        re.compile(pattern, re.IGNORECASE)
                        for pattern in pattern_list
                        if pattern.strip()
                    ]

    def extract_context(self, text: str, position: int) -> str:
        """Extract context around a match position."""
        start = max(0, position - self.context_window // 2)
        end = min(len(text), position + self.context_window // 2)
        return text[start:end].strip()

    def find_matches(self, text: str, category: str) -> List[Dict[str, Any]]:
        """Find matches for a specific category in text."""
        matches = []
        pattern = self.compiled_patterns.get(category)
        
        if not pattern:
            return matches

        if isinstance(pattern, list):
            # Handle individual patterns
            for p in pattern:
                for match in p.finditer(text):
                    matches.append({
                        'text': match.group(0),
                        'context': self.extract_context(text, match.start()),
                        'position': match.start()
                    })
        else:
            # Handle combined pattern
            for match in pattern.finditer(text):
                matches.append({
                    'text': match.group(0),
                    'context': self.extract_context(text, match.start()),
                    'position': match.start()
                })

        return matches

    def find_skills(
        self,
        text: str,
        use_fuzzy: bool = True,
        adjust_scores: bool = True
    ) -> Set[tuple]:
        """Find skills with enhanced matching and scoring."""
        # Get exact matches from trie
        exact_matches = self.skill_trie.search(text)
        matches = set(exact_matches)
        
        if use_fuzzy:
            # Get fuzzy matches
            fuzzy_matches = self.fuzzy_matcher.find_fuzzy_matches(
                text,
                self.skill_patterns
            )
            
            for match in fuzzy_matches:
                # Skip if we already have an exact match
                if match.pattern not in [m[0] for m in exact_matches]:
                    # Adjust score if requested
                    if adjust_scores:
                        score = self.fuzzy_matcher.adjust_match_score(match)
                        score = self.adjust_score_by_context(match, text)
                        
                        # Update match data with adjusted score
                        if isinstance(match.data, dict):
                            match.data = match.data.copy()
                            match.data['confidence'] = score
                            match.data['match_type'] = match.match_type
                    
                    matches.add((match.text, match.data))
        
        return matches

    def find_companies(
        self,
        text: str,
        use_fuzzy: bool = True,
        adjust_scores: bool = True
    ) -> Set[tuple]:
        """Find companies with enhanced matching and scoring."""
        exact_matches = self.company_trie.search(text)
        matches = set(exact_matches)
        
        if use_fuzzy:
            fuzzy_matches = self.fuzzy_matcher.find_fuzzy_matches(
                text,
                self.company_patterns
            )
            
            for match in fuzzy_matches:
                if match.pattern not in [m[0] for m in exact_matches]:
                    if adjust_scores:
                        score = self.fuzzy_matcher.adjust_match_score(match)
                        score = self.adjust_score_by_context(match, text)
                        
                        if isinstance(match.data, dict):
                            match.data = match.data.copy()
                            match.data['confidence'] = score
                            match.data['match_type'] = match.match_type
                    
                    matches.add((match.text, match.data))
        
        return matches

    def find_schools(
        self,
        text: str,
        use_fuzzy: bool = True,
        adjust_scores: bool = True
    ) -> Set[tuple]:
        """Find schools with enhanced matching and scoring."""
        exact_matches = self.school_trie.search(text)
        matches = set(exact_matches)
        
        if use_fuzzy:
            fuzzy_matches = self.fuzzy_matcher.find_fuzzy_matches(
                text,
                self.school_patterns
            )
            
            for match in fuzzy_matches:
                if match.pattern not in [m[0] for m in exact_matches]:
                    if adjust_scores:
                        score = self.fuzzy_matcher.adjust_match_score(match)
                        score = self.adjust_score_by_context(match, text)
                        
                        if isinstance(match.data, dict):
                            match.data = match.data.copy()
                            match.data['confidence'] = score
                            match.data['match_type'] = match.match_type
                    
                    matches.add((match.text, match.data))
        
        return matches

    def get_best_matches(
        self,
        text: str,
        category: str,
        top_n: int = 1
    ) -> Dict[str, List[FuzzyMatch]]:
        """Get top N best matches for a specific category."""
        patterns = {
            'skills': self.skill_patterns,
            'companies': self.company_patterns,
            'schools': self.school_patterns
        }.get(category, {})
        
        return self.fuzzy_matcher.find_best_matches(text, patterns, top_n)
