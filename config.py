import os
from typing import Dict, Any

class Config:
    """Configuration settings for UIUC CourseBot"""
    
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Model Configuration
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE = 0.1
    OPENAI_MAX_TOKENS = 300
    
    # Vector Store Configuration
    PINECONE_INDEX_NAME = "uiuc-chatbot"
    PINECONE_DIMENSION = 1536  # OpenAI ada-002
    PINECONE_METRIC = "cosine"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"
    
    # Embedding Configuration
    EMBEDDING_MODEL = "text-embedding-ada-002"
    HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    FALLBACK_DIMENSION = 384
    
    # Hugging Face Configuration
    HF_MODEL_PRIORITY = False  # Try OpenAI first, HF as fallback
    HF_CACHE_DIR = ".hf_cache"
    
    # Data Configuration
    DATA_FILES = [
        "data/processed/course_chunks.json",
        "data/processed/courserequirements_chunks.json", 
        "data/processed/redditchunks.json"
    ]
    
    # RAG Configuration
    RETRIEVAL_K = 12  # Get more chunks to ensure we capture split information
    RETRIEVAL_FETCH_K = 30  # Fetch more candidates
    RETRIEVAL_LAMBDA_MULT = 0.7
    CHUNK_SIZE = 150
    CHUNK_OVERLAP = 50
    BATCH_SIZE = 100
    
    # Confidence and Quality Control
    MIN_CONFIDENCE_THRESHOLD = 0.4  # Reject answers below this confidence (lowered for better recall)
    MIN_SOURCE_RELEVANCE = 0.2      # Minimum relevance score for sources (lowered)
    REQUIRE_EXACT_COURSE_MATCH = True  # For course-specific queries
    SHOW_SOURCES = True             # Always show source attribution
    
    # Quality Filters
    MIN_CONTENT_LENGTH = 30
    MIN_WORD_COUNT = 10
    MIN_CHUNK_WORDS = 20

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment")
        return True