import asyncio
import hashlib
import json
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class APICache:
    """Intelligent caching system for API responses"""
    
    def __init__(self, cache_dir: str = ".api_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_hours = ttl_hours
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'evictions': 0
        }
        self._lock = threading.Lock()
        
        # Clean up expired cache on initialization
        self._cleanup_expired_cache()
    
    def _get_cache_key(self, request_data: Any) -> str:
        """Generate cache key from request data"""
        # Convert request to string and hash it
        if isinstance(request_data, dict):
            # Sort dict for consistent hashing
            sorted_data = json.dumps(request_data, sort_keys=True)
        else:
            sorted_data = str(request_data)
        
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cache file is expired"""
        if not cache_path.exists():
            return True
        
        # Check file age
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = modified_time + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def get(self, request_data: Any) -> Optional[Any]:
        """Get cached response"""
        cache_key = self._get_cache_key(request_data)
        
        with self._lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not self._is_cache_entry_expired(entry):
                    self.cache_stats['hits'] += 1
                    return entry['data']
                else:
                    del self.memory_cache[cache_key]
            
            # Check disk cache
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists() and not self._is_expired(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Store in memory cache
                    self.memory_cache[cache_key] = {
                        'data': cached_data,
                        'timestamp': datetime.now()
                    }
                    
                    self.cache_stats['hits'] += 1
                    return cached_data
                
                except Exception as e:
                    print(f"⚠️ Cache read error: {e}")
                    # Remove corrupted cache file
                    cache_path.unlink(missing_ok=True)
            
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, request_data: Any, response_data: Any):
        """Cache response data"""
        cache_key = self._get_cache_key(request_data)
        
        with self._lock:
            # Store in memory
            self.memory_cache[cache_key] = {
                'data': response_data,
                'timestamp': datetime.now()
            }
            
            # Store on disk
            cache_path = self._get_cache_path(cache_key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(response_data, f)
                self.cache_stats['saves'] += 1
            except Exception as e:
                print(f"⚠️ Cache write error: {e}")
    
    def _is_cache_entry_expired(self, entry: Dict) -> bool:
        """Check if memory cache entry is expired"""
        expiry_time = entry['timestamp'] + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def _cleanup_expired_cache(self):
        """Remove expired cache files"""
        expired_count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            if self._is_expired(cache_file):
                cache_file.unlink()
                expired_count += 1
        
        if expired_count > 0:
            print(f"🧹 Cleaned up {expired_count} expired cache files")
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            print("🧹 Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len(list(self.cache_dir.glob("*.cache")))
        }


class APIBatcher:
    """Batching system for efficient API requests"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 1.0, max_workers: int = 5):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_workers = max_workers
        self.pending_requests = []
        self.batch_futures = {}
        self._lock = threading.Lock()
        self._batch_timer = None
        self.stats = {
            'total_requests': 0,
            'batched_requests': 0,
            'individual_requests': 0,
            'batches_processed': 0
        }
    
    def add_request(self, request_func, *args, **kwargs) -> Any:
        """Add request to batch or execute immediately"""
        with self._lock:
            self.stats['total_requests'] += 1
            
            # For non-batchable requests, execute immediately
            if not self._is_batchable(request_func):
                self.stats['individual_requests'] += 1
                return request_func(*args, **kwargs)
            
            # Add to pending batch
            request_id = len(self.pending_requests)
            self.pending_requests.append({
                'id': request_id,
                'func': request_func,
                'args': args,
                'kwargs': kwargs
            })
            self.stats['batched_requests'] += 1
            
            # Start batch timer if this is the first request
            if len(self.pending_requests) == 1:
                self._start_batch_timer()
            
            # Process batch if it's full
            if len(self.pending_requests) >= self.batch_size:
                return self._process_batch()
            
            # Wait for batch to complete
            return self._wait_for_batch(request_id)
    
    def _is_batchable(self, func) -> bool:
        """Check if function can be batched"""
        # Add logic to determine if function supports batching
        batchable_functions = [
            'embed_documents',  # Embedding functions are typically batchable
            'encode'  # SentenceTransformer encode
        ]
        
        func_name = getattr(func, '__name__', str(func))
        return any(batch_func in func_name for batch_func in batchable_functions)
    
    def _start_batch_timer(self):
        """Start timer to process batch after timeout"""
        if self._batch_timer:
            self._batch_timer.cancel()
        
        self._batch_timer = threading.Timer(self.batch_timeout, self._process_batch)
        self._batch_timer.start()
    
    def _process_batch(self):
        """Process accumulated batch"""
        with self._lock:
            if not self.pending_requests:
                return None
            
            batch = self.pending_requests.copy()
            self.pending_requests.clear()
            self.stats['batches_processed'] += 1
            
            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None
        
        # Group requests by function
        function_groups = {}
        for req in batch:
            func_name = str(req['func'])
            if func_name not in function_groups:
                function_groups[func_name] = []
            function_groups[func_name].append(req)
        
        results = {}
        
        # Process each function group
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_group = {}
            
            for func_name, requests in function_groups.items():
                if len(requests) == 1:
                    # Single request - execute normally
                    req = requests[0]
                    future = executor.submit(req['func'], *req['args'], **req['kwargs'])
                    future_to_group[future] = [req['id']]
                else:
                    # Multiple requests - try to batch
                    future = executor.submit(self._execute_batch, requests)
                    future_to_group[future] = [req['id'] for req in requests]
            
            # Collect results
            for future in as_completed(future_to_group):
                request_ids = future_to_group[future]
                try:
                    batch_results = future.result()
                    if isinstance(batch_results, list) and len(batch_results) == len(request_ids):
                        # Batched results
                        for req_id, result in zip(request_ids, batch_results):
                            results[req_id] = result
                    else:
                        # Single result
                        results[request_ids[0]] = batch_results
                except Exception as e:
                    # Handle errors
                    for req_id in request_ids:
                        results[req_id] = {'error': str(e)}
        
        # Store results for waiting requests
        for req_id, result in results.items():
            self.batch_futures[req_id] = result
        
        return results
    
    def _execute_batch(self, requests: List[Dict]) -> List[Any]:
        """Execute a batch of similar requests"""
        if not requests:
            return []
        
        # Try to combine arguments for batch execution
        first_req = requests[0]
        func = first_req['func']
        
        try:
            # For embedding functions, combine all texts
            if hasattr(func, '__self__') and hasattr(func.__self__, 'encode'):
                # SentenceTransformer-style batching
                all_texts = []
                for req in requests:
                    if req['args']:
                        all_texts.extend(req['args'])
                
                batch_results = func(all_texts)
                
                # Split results back to individual requests
                results = []
                start_idx = 0
                for req in requests:
                    end_idx = start_idx + len(req['args'])
                    results.append(batch_results[start_idx:end_idx])
                    start_idx = end_idx
                
                return results
            
            else:
                # Execute individually if batching not supported
                results = []
                for req in requests:
                    result = req['func'](*req['args'], **req['kwargs'])
                    results.append(result)
                return results
                
        except Exception as e:
            # Fallback to individual execution
            results = []
            for req in requests:
                try:
                    result = req['func'](*req['args'], **req['kwargs'])
                    results.append(result)
                except Exception as req_e:
                    results.append({'error': str(req_e)})
            return results
    
    def _wait_for_batch(self, request_id: int, timeout: float = 30.0) -> Any:
        """Wait for batch containing request to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.batch_futures:
                result = self.batch_futures.pop(request_id)
                return result
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting
        
        # Timeout - execute request individually
        print(f"⚠️ Batch timeout for request {request_id}, executing individually")
        with self._lock:
            # Find and remove the request
            for i, req in enumerate(self.pending_requests):
                if req['id'] == request_id:
                    removed_req = self.pending_requests.pop(i)
                    return removed_req['func'](*removed_req['args'], **removed_req['kwargs'])
        
        raise TimeoutError(f"Request {request_id} timed out")
    
    def get_stats(self) -> Dict:
        """Get batching statistics"""
        total = self.stats['total_requests']
        batch_efficiency = self.stats['batched_requests'] / total if total > 0 else 0
        
        return {
            **self.stats,
            'batch_efficiency': batch_efficiency,
            'pending_requests': len(self.pending_requests),
            'avg_batch_size': self.stats['batched_requests'] / max(self.stats['batches_processed'], 1)
        }


class OptimizedAPIClient:
    """Combined caching and batching API client"""
    
    def __init__(self, cache_ttl_hours: int = 24, batch_size: int = 10, batch_timeout: float = 1.0):
        self.cache = APICache(ttl_hours=cache_ttl_hours)
        self.batcher = APIBatcher(batch_size=batch_size, batch_timeout=batch_timeout)
        self.enabled = True
        
        print(f"🚀 Optimized API client initialized")
        print(f"   Cache TTL: {cache_ttl_hours} hours")
        print(f"   Batch size: {batch_size}")
        print(f"   Batch timeout: {batch_timeout}s")
    
    def execute_with_optimization(self, func, cache_key_data=None, *args, **kwargs):
        """Execute function with caching and batching optimizations"""
        if not self.enabled:
            return func(*args, **kwargs)
        
        # Use provided cache key data or function arguments
        if cache_key_data is None:
            cache_key_data = {'func': str(func), 'args': args, 'kwargs': kwargs}
        
        # Check cache first
        cached_result = self.cache.get(cache_key_data)
        if cached_result is not None:
            return cached_result
        
        # Execute with batching
        result = self.batcher.add_request(func, *args, **kwargs)
        
        # Cache the result
        if not isinstance(result, dict) or 'error' not in result:
            self.cache.set(cache_key_data, result)
        
        return result
    
    def toggle_optimization(self, enabled: bool):
        """Enable/disable optimizations"""
        self.enabled = enabled
        print(f"🔧 API optimizations {'enabled' if enabled else 'disabled'}")
    
    def get_stats(self) -> Dict:
        """Get combined optimization statistics"""
        return {
            'cache_stats': self.cache.get_stats(),
            'batch_stats': self.batcher.get_stats(),
            'optimization_enabled': self.enabled
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()


# Utility functions for integration
def create_optimized_wrapper(original_func, api_client: OptimizedAPIClient):
    """Create an optimized wrapper for any function"""
    def optimized_wrapper(*args, **kwargs):
        return api_client.execute_with_optimization(original_func, None, *args, **kwargs)
    
    return optimized_wrapper


class OptimizedLLMWrapper:
    """Wrapper for LangChain LLM that preserves all methods while adding optimizations"""
    
    def __init__(self, base_llm, api_client: OptimizedAPIClient):
        self.base_llm = base_llm
        self.api_client = api_client
        
        # Preserve all attributes of the base LLM
        for attr_name in dir(base_llm):
            if not attr_name.startswith('_') and not hasattr(self, attr_name):
                attr_value = getattr(base_llm, attr_name)
                if callable(attr_value):
                    # Wrap callable methods with optimization
                    setattr(self, attr_name, self._create_optimized_method(attr_name, attr_value))
                else:
                    # Copy non-callable attributes
                    setattr(self, attr_name, attr_value)
    
    def _create_optimized_method(self, method_name: str, original_method):
        """Create optimized version of a method"""
        def optimized_method(*args, **kwargs):
            # For invoke and similar methods, use optimization
            if method_name in ['invoke', 'generate', 'predict']:
                cache_key = {
                    'method': method_name,
                    'args': args,
                    'kwargs': kwargs,
                    'model': getattr(self.base_llm, 'model_name', str(self.base_llm))
                }
                return self.api_client.execute_with_optimization(
                    original_method, cache_key, *args, **kwargs
                )
            else:
                # For other methods, call directly
                return original_method(*args, **kwargs)
        
        return optimized_method