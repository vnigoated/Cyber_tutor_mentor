"""
Cybersecurity AI Mentor - Main Entry Point

A ChatGPT-like cybersecurity learning companion with conversation memory
and RAG-powered responses using local documents.
"""
import logging
from config import Config, validate_config
from ui_interface import create_app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the Cybersecurity AI Mentor application.
    """
    try:
        # Validate configuration
        logger.info("Starting Cybersecurity AI Mentor...")
        validate_config()
        logger.info("Configuration validated successfully")
        
        # Create and launch the application
        logger.info("Creating Gradio interface...")
        app = create_app()
        
        logger.info("Launching application...")
        app.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,       # Default Gradio port
            share=False,            # Set to True to create a public link
            debug=False,            # Set to True for development
            show_error=True,        # Show errors in the interface
        )
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"❌ Configuration Error: {e}")
        print("Please check your .env file and ensure GROQ_API_KEY is set.")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ File Error: {e}")
        print("Please ensure the 'docs' directory exists and contains your cybersecurity documents.")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected Error: {e}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()
