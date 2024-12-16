from typing import Dict, List, Set, Optional, Any, Tuple
import regex as re
from collections import defaultdict
import logging
from dataclasses import dataclass
from Levenshtein import distance, ratio
import difflib

logger = logging.getLogger('unified_matcher')

@dataclass
class MatchResult:
    """Represents a match result with context"""
    text: str
    pattern: str
    score: float
    context: str
    match_type: str  # 'exact', 'fuzzy', 'pattern'
    start_pos: int
    end_pos: int
    confidence: float

class UnifiedMatcher:
    """Unified matcher combining pattern and fuzzy matching capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the matcher with configuration"""
        self.config = config
        self.scoring_adjustments = {
            'context_relevance_bonus': 0.1,
            'length_penalty': 0.05,
            'position_bonus': 0.1,
            'frequency_bonus': 0.05
        }
        self.context_window = 100  # characters before/after match
        
    def find_matches(self, text: str, patterns: Dict[str, str]) -> Dict[str, List[MatchResult]]:
        """Find all matches using combined approach"""
        results = defaultdict(list)
        
        for skill_name, pattern in patterns.items():
            # Try exact pattern matching first
            pattern_matches = self._find_pattern_matches(text, pattern)
            
            # Try fuzzy matching if no pattern matches
            fuzzy_matches = []
            if not pattern_matches:
                fuzzy_matches = self._find_fuzzy_matches(text, pattern)
            
            # Combine and deduplicate matches
            all_matches = pattern_matches + fuzzy_matches
            unique_matches = self._deduplicate_matches(all_matches)
            
            # Store results
            if unique_matches:
                results[skill_name].extend(unique_matches)
        
        return dict(results)

    def _find_pattern_matches(self, text: str, pattern: str) -> List[MatchResult]:
        """Find matches using regex patterns"""
        matches = []
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                context = self._extract_context(text, start, end)
                
                result = MatchResult(
                    text=match.group(),
                    pattern=pattern,
                    score=1.0,
                    context=context,
                    match_type='pattern',
                    start_pos=start,
                    end_pos=end,
                    confidence=self._calculate_confidence(match.group(), pattern, context)
                )
                matches.append(result)
                
        except re.error as e:
            logger.error(f"Pattern matching error: {str(e)}")
            
        return matches

    def _find_fuzzy_matches(self, text: str, pattern: str) -> List[MatchResult]:
        """Find matches using fuzzy string matching"""
        matches = []
        words = text.split()
        
        for i, word in enumerate(words):
            score = ratio(word.lower(), pattern.lower())
            if score >= self.config.get('fuzzy_threshold', 0.8):
                # Calculate position in original text
                start = len(' '.join(words[:i]))
                end = start + len(word)
                context = self._extract_context(text, start, end)
                
                result = MatchResult(
                    text=word,
                    pattern=pattern,
                    score=score,
                    context=context,
                    match_type='fuzzy',
                    start_pos=start,
                    end_pos=end,
                    confidence=score * 0.9  # Slightly lower confidence for fuzzy matches
                )
                matches.append(result)
                
        return matches

    def _calculate_confidence(self, match_text: str, pattern: str, context: str) -> float:
        """Calculate confidence score for a match"""
        base_confidence = 0.8
        
        # Context relevance
        context_lower = context.lower()
        relevance_indicators = [
            'experience', 'expert', 'proficient', 'skilled',
            'developed', 'implemented', 'built', 'created'
        ]
        for indicator in relevance_indicators:
            if indicator in context_lower:
                base_confidence += self.scoring_adjustments['context_relevance_bonus']
        
        # Length similarity
        length_ratio = min(len(match_text), len(pattern)) / max(len(match_text), len(pattern))
        base_confidence += length_ratio * self.scoring_adjustments['length_penalty']
        
        # Cap confidence at 1.0
        return min(1.0, base_confidence)

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract surrounding context for a match"""
        context_start = max(0, start - self.context_window)
        context_end = min(len(text), end + self.context_window)
        return text[context_start:context_end].strip()

    def _deduplicate_matches(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Remove overlapping matches, keeping the highest scoring ones"""
        if not matches:
            return []
            
        # Sort by score and position
        sorted_matches = sorted(matches, key=lambda x: (-x.score, x.start_pos))
        
        # Keep track of used positions
        used_positions = set()
        unique_matches = []
        
        for match in sorted_matches:
            # Check if this match overlaps with any existing matches
            overlap = False
            for pos in range(match.start_pos, match.end_pos):
                if pos in used_positions:
                    overlap = True
                    break
                    
            if not overlap:
                # Add all positions covered by this match
                for pos in range(match.start_pos, match.end_pos):
                    used_positions.add(pos)
                unique_matches.append(match)
                
        return unique_matches

    def adjust_score_by_context(self, match: MatchResult) -> float:
        """Adjust match score based on surrounding context"""
        score = match.score
        context = match.context.lower()
        
        # Context keywords from config
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
