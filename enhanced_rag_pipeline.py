import re
import time
from typing import List, Dict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import Config
from performance import PerformanceBenchmark
from custom_embeddings import CustomEmbeddingOptimizer, OptimizedEmbeddingWrapper
from token_optimizer import TokenAwarePromptOptimizer
from api_optimizer import OptimizedAPIClient, OptimizedLLMWrapper

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with performance optimizations"""
    
    def __init__(self, retriever, enable_optimizations: bool = True):
        self.retriever = retriever
        self.enable_optimizations = enable_optimizations
        
        # Initialize optimization components
        if enable_optimizations:
            print("🚀 Initializing enhanced RAG pipeline with optimizations...")
            self._initialize_optimizations()
        else:
            print("📝 Initializing basic RAG pipeline...")
            self._initialize_basic_components()
        
        print("✅ Enhanced RAG pipeline ready!")
    
    def _initialize_optimizations(self):
        """Initialize all optimization components"""
        # Performance benchmarking
        self.benchmark = PerformanceBenchmark()
        
        # Token-aware prompt optimization
        self.token_optimizer = TokenAwarePromptOptimizer(Config.OPENAI_MODEL)
        
        # API optimization (caching + batching)
        self.api_client = OptimizedAPIClient(
            cache_ttl_hours=24,
            batch_size=5,  # Smaller batches for better responsiveness
            batch_timeout=0.5
        )
        
        # Custom embedding optimization
        self.embedding_optimizer = CustomEmbeddingOptimizer()
        
        # Initialize optimized LLM with caching
        self._setup_optimized_llm()
        
        # Create optimized prompt templates
        self._setup_optimized_prompts()
    
    def _initialize_basic_components(self):
        """Initialize basic components without optimizations"""
        self.benchmark = None
        self.token_optimizer = None
        self.api_client = None
        self.embedding_optimizer = None
        
        # Basic LLM setup
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
            max_tokens=Config.OPENAI_MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Basic prompt
        self.prompt_template = self._create_basic_prompt()
    
    def _setup_optimized_llm(self):
        """Setup LLM with optimizations"""
        # Create base LLM
        base_llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
            max_tokens=Config.OPENAI_MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Wrap with optimizations that preserve LangChain interface
        self.llm = OptimizedLLMWrapper(base_llm, self.api_client)
        
        print("🤖 LLM optimized with caching and batching")
    
    def _setup_optimized_prompts(self):
        """Setup token-optimized prompt templates"""
        # The token optimizer will select the best template dynamically
        self.prompt_templates = self.token_optimizer.prompt_templates
        print("📝 Token-aware prompt templates ready")
    
    def _create_basic_prompt(self):
        """Create basic prompt template"""
        template = """You are an expert UIUC academic advisor with comprehensive knowledge of computer science courses, prerequisites, and student experiences.

Use the provided context to answer the student's question accurately and helpfully. Synthesize information from multiple sources when available.

Context Information:
{context}

Student Question: {question}

Guidelines for your response:
1. Provide direct, actionable answers
2. Include specific course numbers and names when relevant
3. Mention prerequisites when discussing courses
4. If discussing difficulty, provide context about what makes courses challenging
5. If you don't have enough information, say so clearly
6. Keep responses focused and well-structured

Response:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str) -> Dict:
        """Process query through enhanced RAG pipeline"""
        if self.enable_optimizations:
            return self._query_with_optimizations(question)
        else:
            return self._query_basic(question)
    
    def _query_with_optimizations(self, question: str) -> Dict:
        """Enhanced query processing with all optimizations"""
        start_time = time.perf_counter()
        
        try:
            # Step 1: Retrieve relevant documents
            retrieve_start = time.perf_counter()
            retrieved_docs = self.retriever.invoke(question)
            retrieve_time = (time.perf_counter() - retrieve_start) * 1000
            
            # Step 2: Filter and rank documents  
            filter_start = time.perf_counter()
            filtered_docs = self._filter_and_rank_docs(retrieved_docs, question)
            filter_time = (time.perf_counter() - filter_start) * 1000
            
            # Step 3: Optimize context and prompt for tokens
            optimize_start = time.perf_counter()
            optimized_prompt, optimization_stats = self.token_optimizer.optimize_full_prompt(
                filtered_docs, question
            )
            optimize_time = (time.perf_counter() - optimize_start) * 1000
            
            # Step 4: Generate response with optimized LLM
            generate_start = time.perf_counter()
            
            response = self.llm.invoke(optimized_prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            generate_time = (time.perf_counter() - generate_start) * 1000
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Calculate performance stats
            performance_stats = {
                'total_time_ms': total_time,
                'retrieve_time_ms': retrieve_time,
                'filter_time_ms': filter_time,
                'optimize_time_ms': optimize_time,
                'generate_time_ms': generate_time,
                'optimization_enabled': True
            }
            
            return {
                'answer': answer.strip(),
                'sources': [doc.metadata.get('course_codes', []) for doc in filtered_docs],
                'retrieved_docs_count': len(retrieved_docs),
                'used_docs_count': len(filtered_docs),
                'performance_stats': performance_stats,
                'optimization_stats': optimization_stats
            }
            
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'retrieved_docs_count': 0,
                'used_docs_count': 0,
                'performance_stats': {
                    'total_time_ms': total_time,
                    'optimization_enabled': True,
                    'error': str(e)
                }
            }
    
    def _query_basic(self, question: str) -> Dict:
        """Basic query processing without optimizations"""
        start_time = time.perf_counter()
        
        try:
            retrieved_docs = self.retriever.invoke(question)
            filtered_docs = self._filter_and_rank_docs(retrieved_docs, question)
            context = self._prepare_context(filtered_docs)
            
            chain = self.prompt_template | self.llm | StrOutputParser()
            response = chain.invoke({
                "question": question,
                "context": context
            })
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'answer': response.strip(),
                'sources': [doc.metadata.get('course_codes', []) for doc in filtered_docs],
                'retrieved_docs_count': len(retrieved_docs),
                'used_docs_count': len(filtered_docs),
                'performance_stats': {
                    'total_time_ms': total_time,
                    'optimization_enabled': False
                }
            }
            
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'retrieved_docs_count': 0,
                'used_docs_count': 0,
                'performance_stats': {
                    'total_time_ms': total_time,
                    'optimization_enabled': False,
                    'error': str(e)
                }
            }
    
    def _filter_and_rank_docs(self, docs: List[Document], question: str) -> List[Document]:
        """Enhanced filtering with course-specific matching"""
        question_words = set(question.lower().split())
        question_courses = self._extract_course_codes(question)
        scored_docs = []
        
        for doc in docs:
            content_words = set(doc.page_content.lower().split())
            content = doc.page_content.lower()
            
            # Calculate word overlap
            overlap = len(question_words.intersection(content_words))
            
            # Boost for exact course code matches
            course_boost = 0
            if question_courses:
                for course in question_courses:
                    course_pattern = course.replace(' ', r'\\s*')
                    if re.search(course_pattern, content, re.IGNORECASE):
                        course_boost += 3
            
            # Quality boost for important content types
            quality_boost = {
                'course_description': 2,
                'prerequisites': 3,
                'student_advice': 1
            }.get(doc.metadata.get('content_type', 'general'), 0)
            
            # Calculate final score
            score = overlap + quality_boost + course_boost
            
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # For course-specific queries, prioritize all chunks of the target course
        if question_courses:
            target_course = question_courses[0]
            course_specific_docs = []
            other_docs = []
            
            for score, doc in scored_docs:
                doc_courses = doc.metadata.get('course_codes', [])
                doc_course_code = doc.metadata.get('course_code', '')
                
                if target_course in doc_courses or target_course == doc_course_code:
                    course_specific_docs.append((score, doc))
                else:
                    other_docs.append((score, doc))
            
            final_docs = course_specific_docs + other_docs
            return [doc for score, doc in final_docs[:8]]
        
        return [doc for score, doc in scored_docs[:8]]
    
    def _extract_course_codes(self, text: str) -> List[str]:
        """Extract course codes from text"""
        course_pattern = r'\\b([A-Z]{2,4})\\s*(\\d{3})\\b'
        matches = re.findall(course_pattern, text, re.IGNORECASE)
        return [f"{dept.upper()} {num}" for dept, num in matches]
    
    def _prepare_context(self, docs: List[Document]) -> str:
        """Prepare context string from documents"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            course_codes = doc.metadata.get('course_codes', [])
            content_type = doc.metadata.get('content_type', 'general')
            
            context_part = f"Source {i} ({content_type}):"
            if course_codes:
                context_part += f" [Courses: {', '.join(course_codes)}]"
            context_part += f"\\n{content}\\n"
            
            context_parts.append(context_part)
        
        return "\\n".join(context_parts)
    
    def benchmark_performance(self, test_queries: List[str], runs_per_query: int = 3) -> Dict:
        """Run performance benchmarks"""
        if not self.benchmark:
            return {"error": "Benchmarking not available - optimizations disabled"}
        
        return self.benchmark.run_benchmark_suite(self, test_queries, runs_per_query)
    
    def toggle_optimizations(self, enabled: bool):
        """Enable/disable optimizations"""
        if enabled and not self.enable_optimizations:
            # Re-initialize with optimizations
            self.enable_optimizations = True
            self._initialize_optimizations()
        elif not enabled and self.enable_optimizations:
            # Switch to basic mode
            self.enable_optimizations = False
            self._initialize_basic_components()
        
        print(f"🔧 Enhanced RAG optimizations {'enabled' if enabled else 'disabled'}")
    
    def get_optimization_stats(self) -> Dict:
        """Get comprehensive optimization statistics"""
        if not self.enable_optimizations:
            return {"optimization_enabled": False}
        
        stats = {
            "optimization_enabled": True,
            "token_optimizer": self.token_optimizer.get_optimization_summary(),
            "api_client": self.api_client.get_stats(),
        }
        
        if self.embedding_optimizer:
            stats["embedding_optimizer"] = self.embedding_optimizer.get_optimization_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear all caches"""
        if self.api_client:
            self.api_client.clear_cache()
        if self.embedding_optimizer:
            self.embedding_optimizer.optimization_cache.clear()
            self.embedding_optimizer.embeddings_cache.clear()
        print("🧹 All caches cleared")