import os
from typing import Dict, List
from dotenv import load_dotenv

from config import Config
from embeddings import EmbeddingManager
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline
from enhanced_rag_pipeline import EnhancedRAGPipeline
from performance import DEFAULT_TEST_QUERIES

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

class UIUCChatBot:
    """
    UIUC course assistant with RAG capabilities.
    Modular architecture with failover logic and vector search.
    """
    
    def __init__(self):
        print("Initializing UIUC ChatBot...")
        
        # Validate configuration
        Config.validate()
        
        # Initialize components
        self.embedding_manager = EmbeddingManager()
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = None
        self.rag_pipeline = None
        
        self._setup_components()
        
        # Display model info
        model_info = self.embedding_manager.get_model_info()
        print(f"📊 Using {model_info['type']} embeddings ({model_info['dimension']}D)")
        print("ChatBot initialized and ready!")
    
    def _setup_components(self):
        """Initialize all components in sequence"""
        # Process documents
        documents = self.document_processor.load_and_process()
        
        # Setup vector store
        self.vector_store_manager = VectorStoreManager(
            self.embedding_manager.embeddings,
            self.embedding_manager.dimension
        )
        
        # Add documents if new index was created
        if documents:
            self.vector_store_manager.add_documents(documents)
        
        # Setup enhanced RAG pipeline with optimizations
        retriever = self.vector_store_manager.as_retriever()
        self.rag_pipeline = EnhancedRAGPipeline(retriever, enable_optimizations=True)
        
        print("🚀 Enhanced RAG pipeline with performance optimizations enabled")
    
    
    def query(self, question: str) -> Dict:
        """Process query through enhanced RAG pipeline"""
        return self.rag_pipeline.query(question)
    
    def benchmark_performance(self, test_queries: List[str] = None, runs_per_query: int = 3) -> Dict:
        """Run performance benchmarks"""
        if test_queries is None:
            test_queries = DEFAULT_TEST_QUERIES[:5]  # Use first 5 default queries
        
        return self.rag_pipeline.benchmark_performance(test_queries, runs_per_query)
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics"""
        return self.rag_pipeline.get_optimization_stats()
    
    def toggle_optimizations(self, enabled: bool):
        """Enable/disable performance optimizations"""
        self.rag_pipeline.toggle_optimizations(enabled)
    

    def run_interactive_mode(self):
        """Run the chatbot in interactive mode"""
        print("\n" + "="*60)
        print("🎓 UIUC Course Advisor ChatBot - Interactive Mode")
        print("="*60)
        print("Ask me anything about UIUC computer science courses!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🙋 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thanks for using UIUC Course Advisor! Good luck with your studies!")
                    break
                
                # Skip empty inputs
                if not user_input:
                    print("Please enter a question about UIUC courses.")
                    continue
                
                print("\n🤔 Thinking...")
                
                # Get response from chatbot
                result = self.query(user_input)
                
                # Display response
                print(f"\n🤖 ChatBot: {result['answer']}")
                print(f"\n📊 Confidence: {result['confidence']:.2f} | "
                      f"Sources used: {result['used_docs_count']}/{result['retrieved_docs_count']}")
                
                # Show course codes if available
                course_codes = [code for sublist in result['sources'] for code in sublist if code]
                if course_codes:
                    unique_courses = list(set(course_codes))[:5]  # Show up to 5 unique courses
                    print(f"📚 Related courses: {', '.join(unique_courses)}")
                
            except KeyboardInterrupt:
                print("\n\n👋 ChatBot interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again with a different question.")

# Main execution
if __name__ == "__main__":
    try:
        # Initialize bot
        bot = UIUCChatBot()
        
        # Run in interactive mode
        bot.run_interactive_mode()
        
    except Exception as e:
        print(f"\n❌ Failed to initialize chatbot: {str(e)}")
        print("Please check your API keys and data files.")