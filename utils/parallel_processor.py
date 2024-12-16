import concurrent.futures
import logging
import multiprocessing
import psutil
from typing import List, Dict, Any, Callable, Optional, Tuple, TypeVar
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger('parallel_processor')

T = TypeVar('T')  # Generic type for input data

@dataclass
class ProcessingStats:
    """Statistics for parallel processing"""
    total_items: int
    processed_items: int
    successful: int
    failed: int
    start_time: datetime
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> float:
        """Get processing duration in seconds"""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage"""
        if self.processed_items == 0:
            return 0.0
        return (self.successful / self.processed_items) * 100

class ParallelProcessor:
    def __init__(self, max_workers: int = None, memory_limit_gb: float = None):
        """
        Initialize parallel processor with resource management.
        
        Args:
            max_workers: Maximum number of worker processes. If None, uses CPU count
            memory_limit_gb: Maximum memory usage in GB. If None, uses 75% of system memory
        """
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        if memory_limit_gb is None:
            total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
            self.memory_limit_gb = total_memory * 0.75
        else:
            self.memory_limit_gb = memory_limit_gb
            
        self.stats = None

    def process_batch(self, 
                     items: List[T],
                     process_func: Callable[[T], Dict[str, Any]],
                     chunk_size: int = 10,
                     callback: Callable[[ProcessingStats], None] = None) -> Tuple[List[Dict[str, Any]], ProcessingStats]:
        """
        Process a batch of items in parallel with progress tracking.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            chunk_size: Number of items to process in each chunk
            callback: Optional callback function to receive processing stats updates
            
        Returns:
            Tuple of (processing results, final stats)
        """
        self.stats = ProcessingStats(
            total_items=len(items),
            processed_items=0,
            successful=0,
            failed=0,
            start_time=datetime.now()
        )
        
        results = []
        
        try:
            # Process items in chunks to manage memory
            for i in range(0, len(items), chunk_size):
                # Check system resources
                if not self._check_resources():
                    logger.warning("System resources low, reducing chunk size")
                    chunk_size = max(1, chunk_size // 2)
                    
                chunk = items[i:i + chunk_size]
                chunk_results = self._process_chunk(chunk, process_func)
                results.extend(chunk_results)
                
                if callback:
                    callback(self.stats)
                    
        finally:
            self.stats.end_time = datetime.now()
            
        return results, self.stats

    def _process_chunk(self, 
                    items: List[T],
                    process_func: Callable[[T], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a chunk of items in parallel with enhanced error handling."""
        chunk_results = []
        
        logger.info(f"Processing chunk of {len(items)} items")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(self._safe_process, process_func, item): item 
                            for item in items}
            logger.info(f"Submitted {len(future_to_item)} tasks to executor")
            
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    logger.info(f"Processing result for item: {getattr(item, 'name', str(item))}")
                    result = future.result()
                    
                    # Validate result structure
                    if not isinstance(result, dict):
                        logger.error(f"Invalid result type: {type(result)}")
                        result = {
                            'status': 'failed',
                            'error': f'Invalid result type: {type(result)}',
                            'filename': getattr(item, 'name', 'Unknown')
                        }
                    
                    # Log detailed result info
                    logger.info(f"Result status: {result.get('status')}")
                    logger.info(f"Result keys: {list(result.keys())}")
                    
                    if 'analysis' in result:
                        analysis = result['analysis']
                        logger.info(f"Analysis type: {type(analysis)}")
                        if hasattr(analysis, 'model_dump'):
                            logger.info("Analysis fields:")
                            for key, value in analysis.model_dump().items():
                                logger.info(f"- {key}: {type(value)}")
                    
                    chunk_results.append(result)
                    self.stats.processed_items += 1
                    
                    if result.get('status') == 'success':
                        self.stats.successful += 1
                        logger.info(f"Successfully processed: {result.get('filename', 'Unknown')}")
                    else:
                        self.stats.failed += 1
                        logger.error(f"Failed to process: {result.get('error')}")
                        
                except Exception as e:
                    self.stats.processed_items += 1
                    self.stats.failed += 1
                    error_msg = f"Error processing item: {str(e)}"
                    logger.error(error_msg)
                    logger.error(f"Processing error traceback: {traceback.format_exc()}")
                    chunk_results.append({
                        'error': error_msg,
                        'status': 'failed',
                        'filename': getattr(item, 'name', 'Unknown')
                    })
        
        logger.info(f"Chunk processing complete. Results: {len(chunk_results)} items")
        return chunk_results
    
    def _safe_process(self, process_func: Callable, item: T) -> Dict[str, Any]:
        """Safely execute the process function with error handling"""
        try:
            result = process_func(item)
            if isinstance(result, dict):
                result['status'] = result.get('status', 'success')
                return result
            else:
                return {
                    'error': 'Process function returned invalid result',
                    'status': 'failed'
                }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }
            
    def _check_resources(self) -> bool:
        """Check if system has enough resources to continue processing"""
        try:
            memory_usage_gb = psutil.Process().memory_info().rss / (1024 ** 3)
            return memory_usage_gb < self.memory_limit_gb
        except:
            return True  # Continue if we can't check resources
