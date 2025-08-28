import numpy as np
import pickle
import json
from typing import List, Dict, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import Config
import re

class CustomEmbeddingOptimizer:
    """Custom embedding optimization for course-specific queries"""
    
    def __init__(self):
        self.base_model = None
        self.pca_reducer = None
        self.scaler = StandardScaler()
        self.course_vocab = set()
        self.course_patterns = {}
        self.optimization_cache = {}
        self.embeddings_cache = {}
        self.model_path = ".custom_embeddings/"
        Path(self.model_path).mkdir(exist_ok=True)
        
        # Load or create optimization components
        self._load_or_initialize()
    
    def _load_or_initialize(self):
        """Load existing optimizations or initialize new ones"""
        try:
            # Try loading existing optimizations
            if Path(f"{self.model_path}optimizer.pkl").exists():
                with open(f"{self.model_path}optimizer.pkl", 'rb') as f:
                    saved_data = pickle.load(f)
                    self.pca_reducer = saved_data.get('pca_reducer')
                    self.scaler = saved_data.get('scaler', StandardScaler())
                    self.course_vocab = saved_data.get('course_vocab', set())
                    self.course_patterns = saved_data.get('course_patterns', {})
                print("✅ Loaded existing embedding optimizations")
            else:
                print("🔧 Initializing new embedding optimizations")
                self._build_course_vocabulary()
                
        except Exception as e:
            print(f"⚠️ Failed to load optimizations: {e}")
            print("🔧 Initializing fresh optimizations")
            self._build_course_vocabulary()
        
        # Initialize base model
        try:
            self.base_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Loaded base embedding model")
        except Exception as e:
            print(f"❌ Failed to load base model: {e}")
    
    def _build_course_vocabulary(self):
        """Build course-specific vocabulary from data"""
        print("📚 Building course vocabulary...")
        
        course_texts = []
        course_codes = set()
        
        # Load course data
        for data_file in Config.DATA_FILES:
            if Path(data_file).exists():
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for item in data:
                        text = item.get('text', '')
                        if text:
                            course_texts.append(text.lower())
                            
                            # Extract course codes
                            codes = re.findall(r'\\b([A-Z]{2,4})\\s*(\\d{3})\\b', text, re.IGNORECASE)
                            for dept, num in codes:
                                course_codes.add(f"{dept.upper()} {num}")
                
                except Exception as e:
                    print(f"⚠️ Error loading {data_file}: {e}")
        
        # Build vocabulary
        self.course_vocab = set()
        
        # Add course codes
        self.course_vocab.update(course_codes)
        
        # Add common course-related terms
        course_terms = [
            'prerequisite', 'prereq', 'required', 'corequisite',
            'credit', 'hours', 'semester', 'difficulty', 'hard', 'easy',
            'programming', 'mathematics', 'computer', 'science',
            'algorithm', 'data', 'structure', 'theory', 'systems',
            'software', 'engineering', 'design', 'analysis'
        ]
        self.course_vocab.update(course_terms)
        
        # Build course-specific patterns
        self._build_course_patterns(course_texts)
        
        print(f"📖 Built vocabulary with {len(self.course_vocab)} terms and {len(course_codes)} course codes")
    
    def _build_course_patterns(self, texts: List[str]):
        """Build patterns for course-specific text enhancement"""
        self.course_patterns = {
            'prerequisite_indicators': [
                r'prerequisite[s]?[:\\s]+',
                r'prereq[s]?[:\\s]+',
                r'required[:\\s]+',
                r'must take',
                r'need.*before'
            ],
            'difficulty_indicators': [
                r'(very\\s+)?(easy|hard|difficult|challenging)',
                r'(not\\s+)?(too\\s+)?(tough|simple|manageable)',
                r'workload',
                r'time[\\s-]consuming'
            ],
            'course_code_pattern': r'\\b([A-Z]{2,4})\\s*(\\d{3})\\b'
        }
    
    def optimize_embeddings_for_courses(self, texts: List[str], target_dim: int = 256) -> np.ndarray:
        """Create optimized embeddings for course-related texts"""
        if not self.base_model:
            raise Exception("Base model not initialized")
        
        print(f"🔧 Optimizing embeddings for {len(texts)} texts...")
        
        # Enhanced text preprocessing
        enhanced_texts = [self._enhance_text_for_courses(text) for text in texts]
        
        # Generate base embeddings
        base_embeddings = self.base_model.encode(enhanced_texts, show_progress_bar=True)
        
        # Apply course-specific optimizations
        optimized_embeddings = self._apply_course_optimizations(base_embeddings, enhanced_texts)
        
        # Dimensionality reduction if requested
        if target_dim and target_dim < optimized_embeddings.shape[1]:
            optimized_embeddings = self._reduce_dimensions(optimized_embeddings, target_dim)
        
        print("✅ Embedding optimization complete")
        return optimized_embeddings
    
    def _enhance_text_for_courses(self, text: str) -> str:
        """Enhance text with course-specific processing"""
        enhanced = text.lower()
        
        # Boost course codes
        course_codes = re.findall(self.course_patterns['course_code_pattern'], enhanced, re.IGNORECASE)
        for dept, num in course_codes:
            course_code = f"{dept.upper()} {num}"
            # Repeat course codes to boost their importance
            enhanced += f" {course_code} {course_code}"
        
        # Boost prerequisite contexts
        for pattern in self.course_patterns['prerequisite_indicators']:
            if re.search(pattern, enhanced, re.IGNORECASE):
                enhanced += " prerequisite requirement required"
        
        # Boost difficulty contexts
        for pattern in self.course_patterns['difficulty_indicators']:
            if re.search(pattern, enhanced, re.IGNORECASE):
                enhanced += " difficulty level challenge"
        
        return enhanced
    
    def _apply_course_optimizations(self, embeddings: np.ndarray, texts: List[str]) -> np.ndarray:
        """Apply course-specific optimizations to embeddings"""
        optimized = embeddings.copy()
        
        # Course-specific weighting
        for i, text in enumerate(texts):
            # Boost embeddings that contain course codes
            course_boost = len(re.findall(self.course_patterns['course_code_pattern'], text, re.IGNORECASE))
            if course_boost > 0:
                optimized[i] *= (1.0 + course_boost * 0.1)  # 10% boost per course code
            
            # Boost prerequisite information
            prereq_boost = sum(1 for pattern in self.course_patterns['prerequisite_indicators'] 
                             if re.search(pattern, text, re.IGNORECASE))
            if prereq_boost > 0:
                optimized[i] *= (1.0 + prereq_boost * 0.15)  # 15% boost for prerequisites
        
        # Normalize
        norms = np.linalg.norm(optimized, axis=1, keepdims=True)
        optimized = optimized / (norms + 1e-8)
        
        return optimized
    
    def _reduce_dimensions(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Reduce embedding dimensions while preserving course-relevant information"""
        if self.pca_reducer is None:
            print(f"🔄 Training PCA reducer for {target_dim}D...")
            self.pca_reducer = PCA(n_components=target_dim, random_state=42)
            reduced = self.pca_reducer.fit_transform(embeddings)
        else:
            reduced = self.pca_reducer.transform(embeddings)
        
        # Scale the reduced embeddings
        if not hasattr(self.scaler, 'scale_'):
            reduced = self.scaler.fit_transform(reduced)
        else:
            reduced = self.scaler.transform(reduced)
        
        return reduced
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create optimized embedding for a query"""
        if not self.base_model:
            raise Exception("Base model not initialized")
        
        # Check cache first
        cache_key = hash(query)
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        # Enhance query
        enhanced_query = self._enhance_text_for_courses(query)
        
        # Generate base embedding
        base_embedding = self.base_model.encode([enhanced_query])
        
        # Apply optimizations
        optimized = self._apply_course_optimizations(base_embedding, [enhanced_query])
        
        # Apply dimensionality reduction if trained
        if self.pca_reducer is not None:
            optimized = self.pca_reducer.transform(optimized)
            optimized = self.scaler.transform(optimized)
        
        result = optimized[0]
        
        # Cache the result
        self.embeddings_cache[cache_key] = result
        
        return result
    
    def save_optimizations(self):
        """Save trained optimizations"""
        try:
            save_data = {
                'pca_reducer': self.pca_reducer,
                'scaler': self.scaler,
                'course_vocab': self.course_vocab,
                'course_patterns': self.course_patterns
            }
            
            with open(f"{self.model_path}optimizer.pkl", 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"💾 Saved optimizations to {self.model_path}optimizer.pkl")
            
        except Exception as e:
            print(f"❌ Failed to save optimizations: {e}")
    
    def get_optimization_stats(self) -> Dict:
        """Get statistics about the optimizations"""
        return {
            'vocabulary_size': len(self.course_vocab),
            'patterns_count': len(self.course_patterns),
            'cache_size': len(self.embeddings_cache),
            'pca_trained': self.pca_reducer is not None,
            'target_dimensions': self.pca_reducer.n_components_ if self.pca_reducer else None,
            'variance_explained': self.pca_reducer.explained_variance_ratio_.sum() if self.pca_reducer else None
        }


class OptimizedEmbeddingWrapper:
    """Wrapper to integrate custom optimizations with existing embedding systems"""
    
    def __init__(self, base_embeddings, optimizer: CustomEmbeddingOptimizer = None):
        self.base_embeddings = base_embeddings
        self.optimizer = optimizer or CustomEmbeddingOptimizer()
        self.use_optimization = True
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with optimizations"""
        if self.use_optimization and self.optimizer:
            try:
                optimized = self.optimizer.optimize_embeddings_for_courses(texts)
                return optimized.tolist()
            except Exception as e:
                print(f"⚠️ Optimization failed, falling back to base: {e}")
                return self.base_embeddings.embed_documents(texts)
        else:
            return self.base_embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with optimizations"""
        if self.use_optimization and self.optimizer:
            try:
                optimized = self.optimizer.create_query_embedding(text)
                return optimized.tolist()
            except Exception as e:
                print(f"⚠️ Query optimization failed, falling back to base: {e}")
                return self.base_embeddings.embed_query(text)
        else:
            return self.base_embeddings.embed_query(text)
    
    def toggle_optimization(self, enabled: bool):
        """Enable/disable optimizations"""
        self.use_optimization = enabled
        print(f"🔧 Custom optimizations {'enabled' if enabled else 'disabled'}")