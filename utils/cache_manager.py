import hashlib
import json
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger('cache_manager')

class CacheManager:
    """Enhanced cache manager with multi-level caching and batch operations"""
    
    def __init__(self, cache_dir: str = ".cache", cache_duration: int = 24):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            cache_duration: Cache duration in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_duration = timedelta(hours=cache_duration)
        self.memory_cache = {}  # In-memory cache for faster access
        self.memory_cache_hits = defaultdict(int)
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _generate_cache_key(self, data: Any) -> str:
        """Generate a unique cache key"""
        if isinstance(data, (str, bytes)):
            key_data = data
        else:
            key_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(str(key_data).encode()).hexdigest()
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for cache file"""
        return self.cache_dir / f"{cache_key}.json"
        
    def get(self, key_data: Any, category: str = "default") -> Optional[Dict]:
        """
        Get data from cache with memory-first strategy
        
        Args:
            key_data: Data to generate cache key from
            category: Category for cache statistics
        """
        cache_key = self._generate_cache_key(key_data)
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            self.memory_cache_hits[category] += 1
            return self.memory_cache[cache_key]
            
        # Try file cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    
                # Check if cache is still valid
                cached_time = datetime.fromisoformat(data['cached_at'])
                if datetime.now() - cached_time <= self.cache_duration:
                    # Add to memory cache
                    self.memory_cache[cache_key] = data['content']
                    return data['content']
                    
                # Remove expired cache
                os.remove(cache_path)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Cache read error: {str(e)}")
                
        return None
        
    def set(self, key_data: Any, content: Any, category: str = "default"):
        """
        Set data in both memory and file cache
        
        Args:
            key_data: Data to generate cache key from
            content: Content to cache
            category: Category for cache statistics
        """
        cache_key = self._generate_cache_key(key_data)
        
        # Set in memory cache
        self.memory_cache[cache_key] = content
        
        # Set in file cache
        cache_path = self._get_cache_path(cache_key)
        try:
            cache_data = {
                'content': content,
                'cached_at': datetime.now().isoformat(),
                'category': category
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.error(f"Cache write error: {str(e)}")
            
    def get_batch(self, key_data_list: List[Any], category: str = "default") -> Dict[str, Any]:
        """
        Get multiple items from cache
        
        Args:
            key_data_list: List of data to generate cache keys from
            category: Category for cache statistics
        """
        results = {}
        for key_data in key_data_list:
            cache_key = self._generate_cache_key(key_data)
            result = self.get(key_data, category)
            if result is not None:
                results[cache_key] = result
        return results
        
    def set_batch(self, data_dict: Dict[Any, Any], category: str = "default"):
        """
        Set multiple items in cache
        
        Args:
            data_dict: Dictionary of key_data to content mappings
            category: Category for cache statistics
        """
        for key_data, content in data_dict.items():
            self.set(key_data, content, category)
            
    def clear_category(self, category: str):
        """Clear all cache entries for a specific category"""
        # Clear from memory cache
        keys_to_remove = []
        for cache_key, content in self.memory_cache.items():
            cache_path = self._get_cache_path(cache_key)
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    if data.get('category') == category:
                        keys_to_remove.append(cache_key)
                        os.remove(cache_path)
            except:
                continue
                
        for key in keys_to_remove:
            del self.memory_cache[key]
            
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cache usage statistics"""
        stats = {
            'memory_hits': dict(self.memory_cache_hits),
            'cache_size': len(self.memory_cache),
            'categories': defaultdict(int)
        }
        
        # Count files by category
        for cache_path in self.cache_dir.glob('*.json'):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    category = data.get('category', 'default')
                    stats['categories'][category] += 1
            except:
                continue
                
        return dict(stats)
        
    def cleanup(self):
        """Clean up expired cache entries"""
        # Clear expired memory cache
        now = datetime.now()
        keys_to_remove = []
        for cache_key in self.memory_cache:
            cache_path = self._get_cache_path(cache_key)
            if not cache_path.exists():
                keys_to_remove.append(cache_key)
                continue
                
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    cached_time = datetime.fromisoformat(data['cached_at'])
                    if now - cached_time > self.cache_duration:
                        keys_to_remove.append(cache_key)
                        os.remove(cache_path)
            except:
                keys_to_remove.append(cache_key)
                
        for key in keys_to_remove:
            del self.memory_cache[key]
            
        # Clear expired file cache
        for cache_path in self.cache_dir.glob('*.json'):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    cached_time = datetime.fromisoformat(data['cached_at'])
                    if now - cached_time > self.cache_duration:
                        os.remove(cache_path)
            except:
                continue
