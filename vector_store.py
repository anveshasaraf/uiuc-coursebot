import time
from typing import List
from langchain_core.documents import Document
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from config import Config

class VectorStoreManager:
    """Manages Pinecone vector store operations"""
    
    def __init__(self, embeddings, embedding_dimension: int):
        self.embeddings = embeddings
        self.embedding_dimension = embedding_dimension
        self.vector_store = None
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Initialize Pinecone vector store"""
        print("🔗 Setting up vector store...")
        
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        
        if Config.PINECONE_INDEX_NAME in existing_indexes:
            print(f"   Found existing index: {Config.PINECONE_INDEX_NAME}")
            self._validate_existing_index(pc)
            self.vector_store = PineconeVectorStore(
                index_name=Config.PINECONE_INDEX_NAME,
                embedding=self.embeddings
            )
            print("✅ Connected to existing vector store")
        else:
            print(f"   Creating new index: {Config.PINECONE_INDEX_NAME}")
            self._create_new_index(pc)
    
    def _validate_existing_index(self, pc: Pinecone):
        """Check if existing index has correct dimensions"""
        index_info = next(
            (idx for idx in pc.list_indexes() 
             if idx["name"] == Config.PINECONE_INDEX_NAME), 
            None
        )
        
        if index_info:
            existing_dimension = index_info.get("dimension", 0)
            if existing_dimension != self.embedding_dimension:
                print(f"🔄 Index dimension mismatch: {existing_dimension} vs {self.embedding_dimension}")
                print("🗑️ Deleting incompatible index...")
                pc.delete_index(Config.PINECONE_INDEX_NAME)
                time.sleep(10)
                self._create_new_index(pc)
    
    def _create_new_index(self, pc: Pinecone):
        """Create new Pinecone index"""
        pc.create_index(
            name=Config.PINECONE_INDEX_NAME,
            dimension=self.embedding_dimension,
            metric=Config.PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=Config.PINECONE_CLOUD, 
                region=Config.PINECONE_REGION
            )
        )
        print("⏳ Waiting for index to be ready...")
        time.sleep(30)
        print("✅ New index created")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store in batches"""
        if not self.vector_store:
            self.vector_store = PineconeVectorStore.from_documents(
                documents=documents[:Config.BATCH_SIZE],
                embedding=self.embeddings,
                index_name=Config.PINECONE_INDEX_NAME
            )
            documents = documents[Config.BATCH_SIZE:]
        
        # Add remaining documents in batches
        total_docs = len(documents)
        for i in range(0, total_docs, Config.BATCH_SIZE):
            batch = documents[i:i + Config.BATCH_SIZE]
            self.vector_store.add_documents(batch)
            
            batch_num = i // Config.BATCH_SIZE + 1
            total_batches = (total_docs + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
            print(f"   Added batch {batch_num}/{total_batches}")
            time.sleep(1)
        
        print("✅ All documents added to vector store")
    
    def as_retriever(self):
        """Get retriever with configured parameters"""
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": Config.RETRIEVAL_K,
                "fetch_k": Config.RETRIEVAL_FETCH_K,
                "lambda_mult": Config.RETRIEVAL_LAMBDA_MULT
            }
        )