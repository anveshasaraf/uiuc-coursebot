import os
import json
import re
from typing import List, Dict
from langchain_core.documents import Document
from config import Config

class DocumentProcessor:
    """Handles document loading, processing, and chunking"""
    
    def __init__(self):
        self.raw_documents = []
        self.processed_docs = []
        self.chunked_docs = []
    
    def load_and_process(self) -> List[Document]:
        """Main processing pipeline"""
        print("📚 Processing course documents...")
        
        self.raw_documents = self._load_raw_documents()
        self.processed_docs = self._process_documents(self.raw_documents)
        self.chunked_docs = self._create_semantic_chunks(self.processed_docs)
        
        print(f"📊 Processing complete:")
        print(f"   - Raw: {len(self.raw_documents)}")
        print(f"   - Processed: {len(self.processed_docs)}")
        print(f"   - Chunks: {len(self.chunked_docs)}")
        
        return self.chunked_docs
    
    def _load_raw_documents(self) -> List[Dict]:
        """Load documents from data files"""
        all_documents = []
        
        for path in Config.DATA_FILES:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            item['source_file'] = path.split('/')[-1]
                        all_documents.extend(data)
                    print(f"   Loaded {len(data)} items from {path}")
                except Exception as e:
                    print(f"   Error loading {path}: {e}")
        
        return all_documents
    
    def _process_documents(self, raw_docs: List[Dict]) -> List[Dict]:
        """Clean and enhance documents"""
        processed = []
        
        for doc in raw_docs:
            text = doc.get('text', '').strip()
            
            if self._is_high_quality_content(text):
                enhanced_doc = self._enhance_document(doc)
                processed.append(enhanced_doc)
        
        return processed
    
    def _is_high_quality_content(self, text: str) -> bool:
        """Filter low-quality content"""
        if len(text) < Config.MIN_CONTENT_LENGTH:
            return False
        
        low_quality_patterns = [
            r'^Title:\s*$',
            r'^[A-Z\s]+$',
            r'^\d+\.\s*$',
            r'^(Which|What|How|Should I)\s+.*\?$'
        ]
        
        for pattern in low_quality_patterns:
            if re.match(pattern, text):
                return False
        
        return len(text.split()) >= Config.MIN_WORD_COUNT
    
    def _enhance_document(self, doc: Dict) -> Dict:
        """Add metadata and clean content"""
        text = doc.get('text', '').strip()
        
        course_codes = re.findall(r'\b[A-Z]{2,4}\s*\d{3}\b', text)
        content_type = self._categorize_content(text)
        difficulty = self._extract_difficulty(text)
        
        return {
            'content': self._clean_text(text),
            'course_codes': course_codes[:3] if course_codes else [],
            'content_type': content_type,
            'difficulty': difficulty,
            'source': doc.get('source_file', 'unknown'),
            'original_source': str(doc.get('metadata', {}).get('source', 'unknown')),
            'original_type': str(doc.get('metadata', {}).get('type', 'unknown'))
        }
    
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
        """Extract difficulty indicators"""
        text_lower = text.lower()
        
        difficulty_map = [
            (['very easy', 'extremely easy', 'trivial'], 'very_easy'),
            (['easy', 'simple', 'manageable', 'gentle'], 'easy'),
            (['moderate', 'medium', 'average'], 'medium'),
            (['hard', 'difficult', 'challenging', 'tough'], 'hard'),
            (['very hard', 'extremely difficult', 'brutal'], 'very_hard')
        ]
        
        for keywords, level in difficulty_map:
            if any(word in text_lower for word in keywords):
                return level
        
        return 'unknown'
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
        return text.strip()
    
    def _create_semantic_chunks(self, docs: List[Dict]) -> List[Document]:
        """Create overlapping chunks"""
        chunks = []
        
        for doc in docs:
            content = doc['content']
            
            if len(content.split()) > 200:
                text_chunks = self._chunk_text(content)
                for i, chunk in enumerate(text_chunks):
                    metadata = self._create_metadata(doc, i, len(text_chunks))
                    chunks.append(Document(page_content=chunk, metadata=metadata))
            else:
                metadata = self._create_metadata(doc, 0, 1)
                chunks.append(Document(page_content=content, metadata=metadata))
        
        return chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        """Create overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
            chunk = ' '.join(words[i:i + Config.CHUNK_SIZE])
            if len(chunk.split()) >= Config.MIN_CHUNK_WORDS:
                chunks.append(chunk)
        
        return chunks
    
    def _create_metadata(self, doc: Dict, chunk_id: int, total_chunks: int) -> Dict:
        """Create metadata for document chunks"""
        return {
            'chunk_id': chunk_id,
            'total_chunks': total_chunks,
            'content_type': doc.get('content_type', 'general'),
            'difficulty': doc.get('difficulty', 'unknown'),
            'source': doc.get('source', 'unknown'),
            'original_source': doc.get('original_source', 'unknown'),
            'original_type': doc.get('original_type', 'unknown'),
            'course_codes': doc.get('course_codes', [])
        }