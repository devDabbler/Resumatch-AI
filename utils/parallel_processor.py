import concurrent.futures
from typing import List, Callable, Any, Dict, Tuple
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Process multiple files in parallel using ThreadPoolExecutor."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize with maximum number of worker threads."""
        self.max_workers = max_workers
        
    def process_batch(self, items: List[Path], process_func: Callable) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a batch of items in parallel.
        
        Args:
            items: List of items (typically file paths) to process
            process_func: Function to process each item
            
        Returns:
            Tuple of (results list, statistics dict)
        """
        start_time = time.time()
        results = []
        errors = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(process_func, item): item for item in items}
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Successfully processed {item.name}")
                except Exception as e:
                    logger.error(f"Error processing {item.name}: {str(e)}")
                    errors.append({
                        'filename': item.name,
                        'error': str(e)
                    })
        
        # Calculate statistics
        end_time = time.time()
        stats = {
            'total_items': len(items),
            'successful': len(results),
            'failed': len(errors),
            'processing_time': end_time - start_time
        }
        
        # Add failed items to results
        results.extend(errors)
        
        return results, stats
