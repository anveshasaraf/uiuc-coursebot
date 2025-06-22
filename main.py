import os
import json
from datetime import datetime
import time
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import re

# Fixing tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

class UIUCChatBot:
    """
    Production-ready RAG chatbot for UIUC course information.
    Features:
    - Intelligent document processing and chunking
    - Vector similarity search with relevance scoring
    - LLM-powered response synthesis
    - Response quality evaluation
    - Scalable architecture for course catalog updates
    """
    
    def __init__(self):
        print("Initializing UIUC ChatBot...")
        
        self.embedding_dimension = 1024
        
        # Initializing components
        self.setup_embeddings()  # ADD THIS LINE!
        self.setup_llm()
        self.load_and_process_documents()
        self.setup_vector_store()
        self.setup_rag_pipeline()
        
        print("ChatBot initialized and ready!")
    
    def setup_embeddings(self):
        """Initialize embedding model for vector representations"""
        try:
            # Try the advanced model first
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={'device': 'cpu'}
            )
            print("✅ Advanced embeddings model loaded (BGE-large)")
        except Exception as e:
            print(f"⚠️  Advanced model failed: {e}")
            try:
                # Fallback to reliable model
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                print("✅ Fallback embeddings model loaded (MiniLM)")
                # Update dimension for Pinecone
                self.embedding_dimension = 384
            except Exception as e2:
                print(f"❌ Both embedding models failed: {e2}")
                raise Exception("Could not initialize any embedding model")
    
    def setup_llm(self):
        """Initialize language model for response generation"""
        if not os.getenv("OPENAI_API_KEY"):
            raise Exception("❌ OPENAI_API_KEY not found in environment variables. Please set up your OpenAI API key.")
        
        try:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=300,  # Reduced to save tokens
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            print("✅ OpenAI LLM initialized")
        except Exception as e:
            print(f"❌ OpenAI initialization failed: {e}")
            print("💡 Please check:")
            print("   1. Your API key is correct")
            print("   2. You have sufficient quota/credits")
            print("   3. Your billing is set up at https://platform.openai.com/")
            raise Exception(f"Could not initialize OpenAI LLM: {e}")
    
    def load_and_process_documents(self):
        """Load and intelligently process course documents"""
        print("📚 Processing course documents...")
        
        # Load raw documents
        raw_documents = self._load_raw_documents()
        
        # Process and clean documents
        self.processed_docs = self._process_documents(raw_documents)
        
        # Create semantic chunks
        self.chunked_docs = self._create_semantic_chunks(self.processed_docs)
        
        print(f"📊 Document Processing Complete:")
        print(f"   - Raw documents: {len(raw_documents)}")
        print(f"   - Processed documents: {len(self.processed_docs)}")
        print(f"   - Final chunks: {len(self.chunked_docs)}")
    
    def _load_raw_documents(self) -> List[Dict]:
        """Load documents from multiple sources"""
        file_paths = [
            "data/course_chunks.json",
            "data/courserequirements_chunks.json",
            "data/redditchunks.json"
        ]
        
        all_documents = []
        for path in file_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Add source metadata
                        for item in data:
                            item['source_file'] = path.split('/')[-1]
                        all_documents.extend(data)
                    print(f"   Loaded {len(data)} items from {path}")
                except Exception as e:
                    print(f"   Error loading {path}: {e}")
        
        return all_documents
    
    def _process_documents(self, raw_docs: List[Dict]) -> List[Dict]:
        """Clean and enhance document content"""
        processed = []
        
        for doc in raw_docs:
            text = doc.get('text', '').strip()
            
            # Quality filters
            if self._is_high_quality_content(text):
                enhanced_doc = self._enhance_document(doc)
                processed.append(enhanced_doc)
        
        return processed
    
    def _is_high_quality_content(self, text: str) -> bool:
        """Determine if content is worth including"""
        if len(text) < 30:
            return False
        
        # Filter out low-quality patterns
        low_quality_patterns = [
            r'^Title:\s*$',
            r'^[A-Z\s]+$',  # All caps text
            r'^\d+\.\s*$',  # Just numbers
            r'^(Which|What|How|Should I)\s+.*\?$'  # Questions without answers
        ]
        
        for pattern in low_quality_patterns:
            if re.match(pattern, text):
                return False
        
        # Require substantive content
        return len(text.split()) >= 10
    
    def _enhance_document(self, doc: Dict) -> Dict:
        """Add metadata and clean content"""
        text = doc.get('text', '').strip()
        
        # Extract course codes
        course_codes = re.findall(r'\b[A-Z]{2,4}\s*\d{3}\b', text)
        
        # Categorize content type
        content_type = self._categorize_content(text)
        
        # Extract difficulty indicators
        difficulty = self._extract_difficulty(text)
        
        enhanced = {
            'content': self._clean_text(text),
            'course_codes': course_codes[:3] if course_codes else [],
            'content_type': content_type,
            'difficulty': difficulty,
            'source': doc.get('source_file', 'unknown'),
            'original_source': str(doc.get('metadata', {}).get('source', 'unknown')),
            'original_type': str(doc.get('metadata', {}).get('type', 'unknown'))
        }
        
        return enhanced
    
    def _categorize_content(self, text: str) -> str:
        """Categorize content for better retrieval"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['prerequisite', 'prereq', 'required']):
            return 'prerequisites'
        elif any(word in text_lower for word in ['description', 'covers', 'introduces']):
            return 'course_description'
        elif any(word in text_lower for word in ['easy', 'hard', 'difficult', 'challenging']):
            return 'difficulty_review'
        elif any(word in text_lower for word in ['recommend', 'suggest', 'advice']):
            return 'student_advice'
        else:
            return 'general'
    
    def _extract_difficulty(self, text: str) -> str:
        """Extract difficulty level from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['very easy', 'extremely easy', 'trivial']):
            return 'very_easy'
        elif any(word in text_lower for word in ['easy', 'simple', 'manageable', 'gentle']):
            return 'easy'
        elif any(word in text_lower for word in ['moderate', 'medium', 'average']):
            return 'medium'
        elif any(word in text_lower for word in ['hard', 'difficult', 'challenging', 'tough']):
            return 'hard'
        elif any(word in text_lower for word in ['very hard', 'extremely difficult', 'brutal']):
            return 'very_hard'
        else:
            return 'unknown'
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
        return text.strip()
    
    def _create_semantic_chunks(self, docs: List[Dict]) -> List[Document]:
        """Create semantically meaningful chunks"""
        chunks = []
        
        for doc in docs:
            content = doc['content']
            
            # For longer content, create overlapping chunks
            if len(content.split()) > 200:
                text_chunks = self._chunk_text(content, chunk_size=150, overlap=50)
                for i, chunk in enumerate(text_chunks):
                    metadata = {
                        'chunk_id': i,
                        'total_chunks': len(text_chunks),
                        'content_type': doc.get('content_type', 'general'),
                        'difficulty': doc.get('difficulty', 'unknown'),
                        'source': doc.get('source', 'unknown'),
                        'original_source': doc.get('original_source', 'unknown'),
                        'original_type': doc.get('original_type', 'unknown'),
                        'course_codes': doc.get('course_codes', [])
                    }
                    chunks.append(Document(page_content=chunk, metadata=metadata))
            else:
                metadata = {
                    'chunk_id': 0,
                    'total_chunks': 1,
                    'content_type': doc.get('content_type', 'general'),
                    'difficulty': doc.get('difficulty', 'unknown'),
                    'source': doc.get('source', 'unknown'),
                    'original_source': doc.get('original_source', 'unknown'),
                    'original_type': doc.get('original_type', 'unknown'),
                    'course_codes': doc.get('course_codes', [])
                }
                chunks.append(Document(page_content=content, metadata=metadata))
        
        return chunks
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping text chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) >= 20:
                chunks.append(chunk)
        
        return chunks
    
    def setup_vector_store(self):
        """Initialize Pinecone vector store with processed documents"""
        print("🔗 Setting up vector store...")
        
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index_name = "uiuc-chatbot"
    
        # Check if index exists
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        
        if index_name in existing_indexes:
            print(f"   Found existing index: {index_name}")
            
            # Use existing index
            self.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings
        )
            print("✅ Connected to existing vector store")
        
        else:
            print(f"   Creating new index: {index_name}")
            # Create new index
            pc.create_index(
            name=index_name,
            dimension=getattr(self, 'embedding_dimension', 1024),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
            time.sleep(30)
        
            # Add documents in optimized batches
            self._add_documents_to_vector_store(index_name)
            print("✅ Vector store ready")
    
    def _add_documents_to_vector_store(self, index_name: str):
        """Add documents to vector store with batch optimization"""
        batch_size = 100
        total_docs = len(self.chunked_docs)
        
        for i in range(0, total_docs, batch_size):
            batch = self.chunked_docs[i:i + batch_size]
            
            if i == 0:
                self.vector_store = PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    index_name=index_name
                )
            else:
                self.vector_store.add_documents(batch)
            
            print(f"   Added batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
            time.sleep(1)
    
    def setup_rag_pipeline(self):
        """Set up the RAG pipeline with prompt engineering"""
        
        prompt_template = """You are an expert UIUC academic advisor with comprehensive knowledge of computer science courses, prerequisites, and student experiences.

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

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever with advanced configuration
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )
        
        print("✅ RAG pipeline configured")
    
    def query(self, question: str) -> Dict:
        """Process query and return comprehensive response"""
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(question)
            
            # Filter and rank documents
            filtered_docs = self._filter_and_rank_docs(retrieved_docs, question)
            
            # Prepare context
            context = self._prepare_context(filtered_docs)
            
            # Generate response using LLM
            response = self._generate_response(question, context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(filtered_docs, question)
            
            # Return comprehensive result
            return {
                'answer': response,
                'sources': [doc.metadata.get('course_codes', []) for doc in filtered_docs],
                'confidence': confidence,
                'retrieved_docs_count': len(retrieved_docs),
                'used_docs_count': len(filtered_docs)
            }
        
        except Exception as e:
            # Return error result with safe defaults
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'retrieved_docs_count': 0,
                'used_docs_count': 0
            }
    
    def _filter_and_rank_docs(self, docs: List[Document], question: str) -> List[Document]:
        """Filter out low-relevance docs and rank by relevance"""
        question_words = set(question.lower().split())
        
        scored_docs = []
        for doc in docs:
            content_words = set(doc.page_content.lower().split())
            overlap = len(question_words.intersection(content_words))
            
            # Boost score for high-quality content types
            quality_boost = 0
            content_type = doc.metadata.get('content_type', 'general')
            if content_type in ['course_description', 'prerequisites']:
                quality_boost = 2
            elif content_type == 'student_advice':
                quality_boost = 1
            
            score = overlap + quality_boost
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top docs
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:5]]
    
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
            context_part += f"\n{content}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using LLM"""
        try:
            # DEBUG: Print what context is being sent to LLM
            print(f"\n🔍 DEBUG - Context being sent to LLM:")
            print(f"Context length: {len(context)} characters")
            print(f"First 500 characters: {context[:500]}...")
            print("-" * 50)
            
            # Create the chain
            chain = self.prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = chain.invoke({
                "question": question,
                "context": context
            })
            
            return response.strip()
            
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def _calculate_confidence(self, docs: List[Document], question: str) -> float:
        """Calculate confidence score for the response"""
        if not docs:
            return 0.0
        
        question_words = set(question.lower().split())
        
        # Calculate relevance scores for each document
        relevance_scores = []
        for doc in docs:
            content_words = set(doc.page_content.lower().split())
            overlap = len(question_words.intersection(content_words))
            
            # Normalize by question length
            if len(question_words) > 0:
                relevance = overlap / len(question_words)
            else:
                relevance = 0
            
            relevance_scores.append(relevance)
        
        # Base confidence on average relevance
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            base_confidence = min(avg_relevance * 2, 1.0)
        else:
            base_confidence = 0.0
        
        # Boost for high-quality content types
        quality_docs = sum(1 for doc in docs 
                          if doc.metadata.get('content_type') in ['course_description', 'prerequisites'])
        quality_boost = min(quality_docs * 0.15, 0.3)
        
        # Penalty for very short or fragmented content
        content_lengths = [len(doc.page_content.split()) for doc in docs]
        avg_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        if avg_length < 10:
            length_penalty = 0.3
        elif avg_length < 20:
            length_penalty = 0.1
        else:
            length_penalty = 0
        
        final_confidence = max(0.0, min(base_confidence + quality_boost - length_penalty, 1.0))
        return final_confidence

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