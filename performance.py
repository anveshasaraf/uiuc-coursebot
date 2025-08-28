import time
import json
import statistics
from typing import Dict, List, Callable
from datetime import datetime
from pathlib import Path

class PerformanceBenchmark:
    """Performance benchmarking system for measuring latency improvements"""
    
    def __init__(self):
        self.metrics = {
            'query_times': [],
            'embedding_times': [],
            'retrieval_times': [],
            'llm_times': [],
            'total_times': [],
            'optimization_enabled': False
        }
        self.baseline_metrics = {}
        self.results_file = "performance_results.json"
    
    def time_function(self, func: Callable, *args, **kwargs) -> tuple:
        """Time a function execution and return result + duration"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, duration
    
    def benchmark_query(self, chatbot, query: str) -> Dict:
        """Benchmark a complete query execution"""
        print(f"🔍 Benchmarking query: '{query[:50]}...'")
        
        # Time the complete query
        start_total = time.perf_counter()
        
        # Time individual components if possible
        component_times = {}
        
        try:
            # Time document retrieval
            start_retrieval = time.perf_counter()
            retrieved_docs = chatbot.rag_pipeline.retriever.invoke(query)
            retrieval_time = (time.perf_counter() - start_retrieval) * 1000
            component_times['retrieval'] = retrieval_time
            
            # Time document filtering
            start_filter = time.perf_counter()
            filtered_docs = chatbot.rag_pipeline._filter_and_rank_docs(retrieved_docs, query)
            filter_time = (time.perf_counter() - start_filter) * 1000
            component_times['filtering'] = filter_time
            
            # Time context preparation
            start_context = time.perf_counter()
            context = chatbot.rag_pipeline._prepare_context(filtered_docs)
            context_time = (time.perf_counter() - start_context) * 1000
            component_times['context_prep'] = context_time
            
            # Time LLM generation
            start_llm = time.perf_counter()
            response = chatbot.rag_pipeline._generate_response(query, context)
            llm_time = (time.perf_counter() - start_llm) * 1000
            component_times['llm_generation'] = llm_time
            
            total_time = (time.perf_counter() - start_total) * 1000
            
            return {
                'query': query,
                'total_time': total_time,
                'component_times': component_times,
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'docs_retrieved': len(retrieved_docs),
                'docs_used': len(filtered_docs),
                'response_length': len(response)
            }
            
        except Exception as e:
            total_time = (time.perf_counter() - start_total) * 1000
            return {
                'query': query,
                'total_time': total_time,
                'component_times': component_times,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_benchmark_suite(self, chatbot, test_queries: List[str], runs_per_query: int = 3) -> Dict:
        """Run comprehensive benchmarks"""
        print(f"\n🚀 Running benchmark suite with {len(test_queries)} queries, {runs_per_query} runs each")
        
        all_results = []
        query_summaries = {}
        
        for query in test_queries:
            print(f"\n📊 Testing: '{query}'")
            query_results = []
            
            for run in range(runs_per_query):
                print(f"  Run {run + 1}/{runs_per_query}...")
                result = self.benchmark_query(chatbot, query)
                query_results.append(result)
                all_results.append(result)
                
                if result['success']:
                    print(f"    ✅ {result['total_time']:.1f}ms total")
                else:
                    print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Calculate statistics for this query
            successful_runs = [r for r in query_results if r['success']]
            if successful_runs:
                times = [r['total_time'] for r in successful_runs]
                query_summaries[query] = {
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                    'success_rate': len(successful_runs) / len(query_results),
                    'runs': len(successful_runs)
                }
        
        # Calculate overall statistics
        successful_results = [r for r in all_results if r['success']]
        overall_stats = {}
        
        if successful_results:
            total_times = [r['total_time'] for r in successful_results]
            retrieval_times = [r['component_times'].get('retrieval', 0) for r in successful_results]
            llm_times = [r['component_times'].get('llm_generation', 0) for r in successful_results]
            
            overall_stats = {
                'total_queries': len(all_results),
                'successful_queries': len(successful_results),
                'success_rate': len(successful_results) / len(all_results),
                'avg_total_time': statistics.mean(total_times),
                'avg_retrieval_time': statistics.mean([t for t in retrieval_times if t > 0]),
                'avg_llm_time': statistics.mean([t for t in llm_times if t > 0]),
                'p95_total_time': self._percentile(total_times, 95),
                'p99_total_time': self._percentile(total_times, 99)
            }
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'optimization_enabled': self.metrics['optimization_enabled'],
            'overall_stats': overall_stats,
            'query_summaries': query_summaries,
            'detailed_results': all_results
        }
        
        # Save results
        self._save_results(benchmark_results)
        
        return benchmark_results
    
    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def compare_with_baseline(self, current_results: Dict) -> Dict:
        """Compare current results with baseline"""
        if not self.baseline_metrics:
            print("📝 No baseline found - saving current results as baseline")
            self.baseline_metrics = current_results['overall_stats']
            return {'improvement': None, 'message': 'Baseline established'}
        
        current_stats = current_results['overall_stats']
        baseline_stats = self.baseline_metrics
        
        if not current_stats or not baseline_stats:
            return {'improvement': None, 'message': 'Insufficient data for comparison'}
        
        # Calculate improvements
        improvements = {}
        
        # Total time improvement
        if baseline_stats.get('avg_total_time') and current_stats.get('avg_total_time'):
            old_time = baseline_stats['avg_total_time']
            new_time = current_stats['avg_total_time']
            improvement = ((old_time - new_time) / old_time) * 100
            improvements['total_time'] = {
                'baseline': old_time,
                'current': new_time,
                'improvement_percent': improvement
            }
        
        # Retrieval time improvement
        if baseline_stats.get('avg_retrieval_time') and current_stats.get('avg_retrieval_time'):
            old_time = baseline_stats['avg_retrieval_time']
            new_time = current_stats['avg_retrieval_time']
            improvement = ((old_time - new_time) / old_time) * 100
            improvements['retrieval_time'] = {
                'baseline': old_time,
                'current': new_time,
                'improvement_percent': improvement
            }
        
        # LLM time improvement
        if baseline_stats.get('avg_llm_time') and current_stats.get('avg_llm_time'):
            old_time = baseline_stats['avg_llm_time']
            new_time = current_stats['avg_llm_time']
            improvement = ((old_time - new_time) / old_time) * 100
            improvements['llm_time'] = {
                'baseline': old_time,
                'current': new_time,
                'improvement_percent': improvement
            }
        
        return {
            'improvements': improvements,
            'overall_improvement': improvements.get('total_time', {}).get('improvement_percent', 0)
        }
    
    def _save_results(self, results: Dict):
        """Save benchmark results to file"""
        try:
            # Load existing results
            results_file = Path(self.results_file)
            all_results = []
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    all_results = json.load(f)
            
            # Add new results
            all_results.append(results)
            
            # Save back
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"💾 Results saved to {self.results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def print_summary(self, results: Dict):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("🏁 BENCHMARK SUMMARY")
        print("="*60)
        
        stats = results.get('overall_stats', {})
        if not stats:
            print("❌ No statistics available")
            return
        
        print(f"📊 Total Queries: {stats.get('total_queries', 0)}")
        print(f"✅ Success Rate: {stats.get('success_rate', 0):.1%}")
        print(f"⏱️  Average Total Time: {stats.get('avg_total_time', 0):.1f}ms")
        print(f"🔍 Average Retrieval Time: {stats.get('avg_retrieval_time', 0):.1f}ms")
        print(f"🤖 Average LLM Time: {stats.get('avg_llm_time', 0):.1f}ms")
        print(f"📈 95th Percentile: {stats.get('p95_total_time', 0):.1f}ms")
        print(f"📊 99th Percentile: {stats.get('p99_total_time', 0):.1f}ms")
        
        # Show comparison if available
        comparison = self.compare_with_baseline(results)
        if comparison.get('improvements'):
            print("\n🚀 PERFORMANCE IMPROVEMENTS:")
            improvements = comparison['improvements']
            
            if 'total_time' in improvements:
                imp = improvements['total_time']['improvement_percent']
                print(f"   Total Time: {imp:+.1f}% ({improvements['total_time']['baseline']:.1f}ms → {improvements['total_time']['current']:.1f}ms)")
            
            if 'retrieval_time' in improvements:
                imp = improvements['retrieval_time']['improvement_percent']
                print(f"   Retrieval: {imp:+.1f}% ({improvements['retrieval_time']['baseline']:.1f}ms → {improvements['retrieval_time']['current']:.1f}ms)")
            
            if 'llm_time' in improvements:
                imp = improvements['llm_time']['improvement_percent']
                print(f"   LLM Generation: {imp:+.1f}% ({improvements['llm_time']['baseline']:.1f}ms → {improvements['llm_time']['current']:.1f}ms)")
        
        print("="*60)

# Test queries for benchmarking
DEFAULT_TEST_QUERIES = [
    "What are the prerequisites for CS 225?",
    "Tell me about CS 128",
    "What courses should I take before CS 374?",
    "Is CS 357 difficult?",
    "What are the requirements for the CS major?",
    "What is the difference between CS 225 and CS 173?",
    "Should I take MATH 241 before CS 225?",
    "What programming languages are used in CS 128?",
    "How hard is CS 233?",
    "What are good electives for CS students?"
]