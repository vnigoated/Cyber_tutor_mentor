"""
Configuration settings for the Cybersecurity AI Mentor application.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class."""
    
    # API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_NAME = "llama-3.1-8b-instant"
    
    # RAG Configuration
    DOCS_DIRECTORY = "docs"
    SIMILARITY_TOP_K = 3
    MAX_CONTEXT_MESSAGES = 3
    
    # LLM Parameters
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    TOP_P = 0.9
    
    # UI Configuration
    CHAT_HEIGHT = 500
    THEME = "soft"
    
    # Export Configuration
    EXPORT_FILENAME_PREFIX = "cyber_mentor_chat"
    EXPORT_ENCODING = "utf-8"
    
    # User Levels
    USER_LEVELS = ["beginner", "intermediate", "expert"]
    DEFAULT_USER_LEVEL = "beginner"
    
    # Application Info
    APP_TITLE = "Cybersecurity AI Mentor"
    APP_DESCRIPTION = """
    Your personal cybersecurity learning companion with conversation memory and RAG-powered responses.
    Ask questions about cybersecurity topics and I'll use my knowledge base to help you learn!
    """
    
    SAMPLE_QUESTIONS = [
        "Explain SQL injection for beginners",
        "What are the OWASP Top 10?",
        "How does two-factor authentication work?",
        "What is a firewall and how does it work?"
    ]
    
    TIPS = [
        "Ask follow-up questions naturally",
        "Discuss real-world scenarios", 
        "I remember our conversation context!",
        "Questions outside cybersecurity may have limited responses"
    ]

# Validate configuration
def validate_config():
    """Validate that required configuration is present."""
    if not Config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    
    if not os.path.exists(Config.DOCS_DIRECTORY):
        raise FileNotFoundError(f"Documents directory '{Config.DOCS_DIRECTORY}' not found.")
    
    return True