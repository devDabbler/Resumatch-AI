from typing import List, Tuple, Dict, Any, Optional
import regex as re
from Levenshtein import distance, ratio
import logging
from dataclasses import dataclass
from collections import defaultdict
import difflib  # Python standard library for sequence matching

logger = logging.getLogger('fuzzy_matcher')

@dataclass
class FuzzyMatch:
    text: str
    pattern: str
    score: float
    data: Any
    start: int
    end: int
    context: str
    match_type: str = 'exact'  # 'exact', 'fuzzy', 'sequence'
    phonetic_score: float = 0.0

class FuzzyMatcher:
    def __init__(
        self,
        config: Dict[str, Any] = None
    ):
        """
        Initialize FuzzyMatcher with configurable parameters.
        
        Args:
            config: Dictionary containing matching configuration
        """
        self.config = config or {}
        self.matching_config = self.config.get('matching_config', {})
        
        # Set defaults if not in config
        self.min_ratio = self.matching_config.get('min_ratio', 0.85)
        self.context_window = self.matching_config.get('context_window', 100)
        self.max_length_diff = self.matching_config.get('max_length_diff', 0.3)
        self.use_sequence_matcher = self.matching_config.get('use_sequence_matcher', True)
        self.sequence_weight = self.matching_config.get('sequence_weight', 0.3)
        self.fuzzy_weight = self.matching_config.get('fuzzy_weight', 0.7)
        
        self.word_boundaries = re.compile(r'\b\w+\b')

    def _preprocess_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Preprocess text to extract words with their positions.
        Returns list of (word, start, end) tuples.
        """
        return [
            (match.group(), match.start(), match.end())
            for match in self.word_boundaries.finditer(text)
        ]

    def _check_length_compatibility(self, str1: str, str2: str) -> bool:
        """Check if strings are within acceptable length difference."""
        max_len = max(len(str1), len(str2))
        min_len = min(len(str1), len(str2))
        return (max_len - min_len) / max_len <= self.max_length_diff

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not self._check_length_compatibility(str1, str2):
            return 0.0
        return ratio(str1.lower(), str2.lower())

    def _calculate_sequence_score(self, str1: str, str2: str) -> float:
        """Calculate similarity using difflib's SequenceMatcher."""
        if not self.use_sequence_matcher:
            return 0.0
            
        try:
            # Use difflib's SequenceMatcher for more sophisticated matching
            matcher = difflib.SequenceMatcher(None, str1.lower(), str2.lower())
            return matcher.ratio()
            
        except Exception as e:
            logger.warning(f"Sequence matching failed: {str(e)}")
            return 0.0

    def _calculate_combined_score(
        self,
        fuzzy_score: float,
        sequence_score: float
    ) -> float:
        """Calculate combined score from fuzzy and sequence matching."""
        if not self.use_sequence_matcher:
            return fuzzy_score
            
        return (
            fuzzy_score * self.fuzzy_weight +
            sequence_score * self.sequence_weight
        )

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context around match position."""
        context_start = max(0, start - self.context_window // 2)
        context_end = min(len(text), end + self.context_window // 2)
        return text[context_start:context_end].strip()

    def find_fuzzy_matches(
        self,
        text: str,
        patterns: Dict[str, Any],
        use_word_boundaries: bool = True
    ) -> List[FuzzyMatch]:
        """
        Find fuzzy matches in text with enhanced scoring.
        
        Args:
            text: Text to search in
            patterns: Dictionary of patterns with their associated data
            use_word_boundaries: Whether to match only at word boundaries
        """
        matches = []
        words = self._preprocess_text(text) if use_word_boundaries else [(text, 0, len(text))]
        
        for word, start, end in words:
            word_matches = []
            
            for pattern, data in patterns.items():
                # Skip if pattern is too different in length
                if not self._check_length_compatibility(word, pattern):
                    continue
                
                # Calculate fuzzy match score using Levenshtein
                fuzzy_score = self._calculate_similarity(word, pattern)
                
                # Calculate sequence match score using difflib
                sequence_score = self._calculate_sequence_score(word, pattern)
                
                # Calculate combined score
                combined_score = self._calculate_combined_score(
                    fuzzy_score,
                    sequence_score
                )
                
                if combined_score >= self.min_ratio:
                    match_type = 'exact' if fuzzy_score == 1.0 else (
                        'sequence' if sequence_score > fuzzy_score else 'fuzzy'
                    )
                    
                    word_matches.append(FuzzyMatch(
                        text=word,
                        pattern=pattern,
                        score=combined_score,
                        data=data,
                        start=start,
                        end=end,
                        context=self._extract_context(text, start, end),
                        match_type=match_type,
                        phonetic_score=sequence_score  # Reuse this field for sequence score
                    ))
            
            # Add best match for this word if any found
            if word_matches:
                best_match = max(word_matches, key=lambda x: x.score)
                matches.append(best_match)
        
        return matches

    def adjust_match_score(self, match: FuzzyMatch) -> float:
        """
        Adjust match score based on various factors.
        
        Args:
            match: FuzzyMatch object to adjust score for
            
        Returns:
            Adjusted score
        """
        score = match.score
        
        # Adjust based on match type
        type_multipliers = self.matching_config.get('type_multipliers', {
            'exact': 1.0,
            'fuzzy': 0.9,
            'sequence': 0.8
        })
        score *= type_multipliers.get(match.match_type, 0.7)
        
        # Adjust based on length
        length_ratio = len(match.text) / len(match.pattern)
        if length_ratio > 1:
            length_ratio = 1 / length_ratio
        score *= length_ratio
        
        # Apply confidence threshold
        confidence_threshold = self.matching_config.get('confidence_threshold', 0.6)
        if score < confidence_threshold:
            score *= 0.8  # Penalize low confidence matches
            
        return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1

    def group_matches_by_category(
        self,
        matches: List[FuzzyMatch]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group matches by their category.
        
        Args:
            matches: List of FuzzyMatch objects
            
        Returns:
            Dictionary of matches grouped by category
        """
        grouped = defaultdict(list)
        
        for match in matches:
            if isinstance(match.data, dict) and 'category' in match.data:
                category = match.data['category']
                grouped[category].append({
                    'text': match.text,
                    'pattern': match.pattern,
                    'score': match.score,
                    'context': match.context,
                    'data': match.data
                })
            else:
                grouped['uncategorized'].append({
                    'text': match.text,
                    'pattern': match.pattern,
                    'score': match.score,
                    'context': match.context,
                    'data': match.data
                })
        
        return dict(grouped)

    def find_best_matches(
        self,
        text: str,
        patterns: Dict[str, Any],
        top_n: int = 1
    ) -> Dict[str, List[FuzzyMatch]]:
        """
        Find top N best matches for each pattern.
        
        Args:
            text: Text to search in
            patterns: Dictionary of patterns with their data
            top_n: Number of top matches to return per pattern
            
        Returns:
            Dictionary of pattern to list of top matches
        """
        best_matches = defaultdict(list)
        
        for pattern, data in patterns.items():
            pattern_matches = []
            words = self._preprocess_text(text)
            
            for word, start, end in words:
                fuzzy_score = self._calculate_similarity(word, pattern)
                sequence_score = self._calculate_sequence_score(word, pattern)
                combined_score = self._calculate_combined_score(
                    fuzzy_score,
                    sequence_score
                )
                
                if combined_score >= self.min_ratio:
                    match_type = 'exact' if fuzzy_score == 1.0 else (
                        'sequence' if sequence_score > fuzzy_score else 'fuzzy'
                    )
                    
                    pattern_matches.append(FuzzyMatch(
                        text=word,
                        pattern=pattern,
                        score=combined_score,
                        data=data,
                        start=start,
                        end=end,
                        context=self._extract_context(text, start, end),
                        match_type=match_type,
                        phonetic_score=sequence_score
                    ))
            
            if pattern_matches:
                # Sort by score and take top N
                top_matches = sorted(
                    pattern_matches,
                    key=lambda x: x.score,
                    reverse=True
                )[:top_n]
                best_matches[pattern] = top_matches
        
        return dict(best_matches)
