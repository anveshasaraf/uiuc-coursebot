import re
from typing import List, Dict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config

class RAGPipeline:
    """Handles retrieval-augmented generation pipeline"""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = self._setup_llm()
        self.prompt = self._create_prompt()
    
    def _setup_llm(self):
        """Initialize OpenAI LLM"""
        return ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
            max_tokens=Config.OPENAI_MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def _create_prompt(self):
        """Create prompt template for RAG"""
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
        """Process query through RAG pipeline"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(question)
            
            # Filter and rank documents
            filtered_docs = self._filter_and_rank_docs(retrieved_docs, question)
            
            # Prepare context
            context = self._prepare_context(filtered_docs)
            
            # Generate response
            response = self._generate_response(question, context)
            
            return {
                'answer': response,
                'sources': [doc.metadata.get('course_codes', []) for doc in filtered_docs],
                'retrieved_docs_count': len(retrieved_docs),
                'used_docs_count': len(filtered_docs)
            }
            
        except Exception as e:
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'retrieved_docs_count': 0,
                'used_docs_count': 0
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
                    # Use regex for proper course code matching
                    course_pattern = course.replace(' ', r'\s*')
                    if re.search(course_pattern, content, re.IGNORECASE):
                        course_boost += 3
            
            # Quality boost for important content types
            quality_boost = {
                'course_description': 2,
                'prerequisites': 3,  # Higher boost for prereqs
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
            
            # Prioritize course-specific docs, then fill with others
            final_docs = course_specific_docs + other_docs
            return [doc for score, doc in final_docs[:8]]
        
        return [doc for score, doc in scored_docs[:8]]
    
    def _extract_course_codes(self, text: str) -> List[str]:
        """Extract course codes from text"""
        course_pattern = r'\b([A-Z]{2,4})\s*(\d{3})\b'
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
            context_part += f"\n{content}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using LLM"""
        chain = self.prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "context": context
        })
        
        return response.strip()