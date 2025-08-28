import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from config import Config

class EmbeddingManager:
    """Manages embedding models with Hugging Face integration and failover logic"""
    
    def __init__(self):
        self.embeddings = None
        self.dimension = None
        self.model_type = None
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """Initialize embedding model with HuggingFace priority and failover"""
        if Config.HF_MODEL_PRIORITY:
            # Try HuggingFace first
            if self._try_huggingface():
                return
            # Fall back to OpenAI
            if self._try_openai():
                return
            # Final fallback
            self._try_sentence_transformer()
        else:
            # Try OpenAI first
            if self._try_openai():
                return
            # Fall back to HuggingFace
            if self._try_huggingface():
                return
            # Final fallback
            self._try_sentence_transformer()
    
    def _try_huggingface(self) -> bool:
        """Try to initialize HuggingFace embeddings"""
        try:
            # Ensure model is available
            print(f"🤗 Loading HuggingFace model: {Config.HUGGINGFACE_EMBEDDING_MODEL}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.HUGGINGFACE_EMBEDDING_MODEL,
                cache_folder=Config.HF_CACHE_DIR,
                encode_kwargs={'normalize_embeddings': True}
            )
            self.dimension = Config.FALLBACK_DIMENSION  # Most HF models use 384D
            self.model_type = "huggingface"
            print(f"✅ HuggingFace embeddings loaded ({Config.HUGGINGFACE_EMBEDDING_MODEL})")
            return True
            
        except Exception as e:
            print(f"⚠️ HuggingFace embeddings failed: {e}")
            return False
    
    def _try_openai(self) -> bool:
        """Try to initialize OpenAI embeddings"""
        try:
            if not Config.OPENAI_API_KEY:
                print("⚠️ OpenAI API key not found, skipping")
                return False
                
            self.embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY
            )
            self.dimension = Config.PINECONE_DIMENSION
            self.model_type = "openai"
            print(f"✅ OpenAI embeddings loaded ({Config.EMBEDDING_MODEL})")
            return True
            
        except Exception as e:
            print(f"⚠️ OpenAI embeddings failed: {e}")
            return False
    
    def _try_sentence_transformer(self):
        """Final fallback to SentenceTransformers"""
        try:
            print("🔄 Using SentenceTransformer fallback...")
            model = SentenceTransformer(Config.FALLBACK_EMBEDDING_MODEL)
            self.embeddings = _SentenceTransformerWrapper(model)
            self.dimension = Config.FALLBACK_DIMENSION
            self.model_type = "sentence_transformer"
            print(f"✅ Fallback embeddings loaded ({Config.FALLBACK_EMBEDDING_MODEL})")
            
        except Exception as e:
            raise Exception(f"All embedding models failed: {e}")
    
    def get_model_info(self) -> dict:
        """Get information about the current embedding model"""
        return {
            'type': self.model_type,
            'dimension': self.dimension,
            'model_name': getattr(self.embeddings, 'model_name', 'unknown')
        }

class _SentenceTransformerWrapper:
    """Wrapper to make SentenceTransformer compatible with LangChain interface"""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.model_name = Config.FALLBACK_EMBEDDING_MODEL
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()