"""
RAG (Retrieval-Augmented Generation) engine for the Cybersecurity AI Mentor.
Handles document ingestion, indexing, and query processing.
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Handles all RAG-related operations including document loading,
    indexing, and query processing.
    """
    
    def __init__(self):
        """Initialize the RAG engine with embeddings and settings."""
        self.embed_model = None
        self.index = None
        self.query_engine = None
        self._setup_embeddings()
        self._load_documents()
        
    def _setup_embeddings(self):
        """Setup the embedding model and LlamaIndex settings."""
        try:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL_NAME}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=Config.EMBEDDING_MODEL_NAME
            )
            
            # Configure LlamaIndex settings
            Settings.llm = None  # We use Groq for LLM, not LlamaIndex
            Settings.embed_model = self.embed_model
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error setting up embeddings: {e}")
            raise
    
    def _load_documents(self):
        """Load documents from the configured directory."""
        try:
            logger.info(f"Loading documents from: {Config.DOCS_DIRECTORY}")
            reader = SimpleDirectoryReader(Config.DOCS_DIRECTORY)
            documents = reader.load_data()
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Build the index
            self._build_index(documents)
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def _build_index(self, documents):
        """Build the vector index from documents."""
        try:
            logger.info("Building vector index...")
            self.index = VectorStoreIndex.from_documents(
                documents, 
                embed_model=self.embed_model
            )
            
            # Create query engine
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=Config.SIMILARITY_TOP_K
            )
            
            logger.info("Vector index built successfully")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: The retrieved context from documents
        """
        if not self.query_engine:
            raise RuntimeError("Query engine not initialized. Call _load_documents() first.")
        
        try:
            logger.info(f"Processing query: {question[:50]}...")
            response = self.query_engine.query(question)
            
            # Get the response content
            context = response.response if hasattr(response, 'response') else str(response)
            
            # Check if the response indicates no relevant information was found
            if self._is_low_relevance_response(context):
                logger.info("Low relevance response detected")
                return "No relevant cybersecurity information found in documents."
            
            return context
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error retrieving information: {e}"
    
    def _is_low_relevance_response(self, response: str) -> bool:
        """
        Check if the RAG response indicates low relevance to cybersecurity topics.
        
        Args:
            response (str): The response from the RAG system
            
        Returns:
            bool: True if the response seems to have low relevance
        """
        if not response or len(response.strip()) < 20:
            return True
        
        # Check for common low-relevance indicators
        low_relevance_phrases = [
            "I don't have information",
            "I cannot find",
            "not mentioned in the context",
            "no information available",
            "cannot answer based on the provided context",
            "no relevant information",
            "I don't know",
            "not specified in the documents"
        ]
        
        response_lower = response.lower()
        for phrase in low_relevance_phrases:
            if phrase in response_lower:
                return True
        
        # Check if response is too generic (might indicate poor retrieval)
        cybersecurity_keywords = [
            "security", "vulnerability", "attack", "malware", "encryption", 
            "authentication", "authorization", "firewall", "intrusion", 
            "threat", "risk", "exploit", "penetration", "network", "cyber"
        ]
        
        has_cyber_keywords = any(keyword in response_lower for keyword in cybersecurity_keywords)
        
        # If response is long but has no cybersecurity keywords, it might be irrelevant
        if len(response.strip()) > 100 and not has_cyber_keywords:
            return True
        
        return False
    
    def get_index_info(self) -> dict:
        """
        Get information about the current index.
        
        Returns:
            dict: Index information including document count, etc.
        """
        if not self.index:
            return {"status": "not_initialized"}
        
        try:
            # Get basic index information
            info = {
                "status": "initialized",
                "embedding_model": Config.EMBEDDING_MODEL_NAME,
                "similarity_top_k": Config.SIMILARITY_TOP_K,
                "docs_directory": Config.DOCS_DIRECTORY
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return {"status": "error", "error": str(e)}
    
    def refresh_index(self):
        """Refresh the index by reloading documents."""
        logger.info("Refreshing index...")
        self._load_documents()
        logger.info("Index refreshed successfully")

# Create a global instance
_rag_engine_instance = None

def get_rag_engine() -> RAGEngine:
    """
    Get the global RAG engine instance (singleton pattern).
    
    Returns:
        RAGEngine: The RAG engine instance
    """
    global _rag_engine_instance
    
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    
    return _rag_engine_instance