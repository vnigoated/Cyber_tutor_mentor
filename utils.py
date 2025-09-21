"""
Utility functions for the Cybersecurity AI Mentor application.
"""
import os
from datetime import datetime
from typing import List, Dict
from config import Config
from conversation_manager import get_conversation_manager
import logging
import json
import re
import time
from typing import Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_conversation_to_file() -> str:
    """
    Export the current conversation to a text file.
    
    Returns:
        str: Status message about the export operation
    """
    conversation_manager = get_conversation_manager()
    conversation_history = conversation_manager.conversation_history
    
    if not conversation_history:
        return "No conversation to export."
    
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{Config.EXPORT_FILENAME_PREFIX}_{timestamp}.txt"
        
        # Create export content
        export_text = generate_export_content(conversation_history)
        
        # Write to file
        with open(filename, 'w', encoding=Config.EXPORT_ENCODING) as f:
            f.write(export_text)
        
        logger.info(f"Conversation exported to {filename}")
        return f"Conversation exported to {filename}"
        
    except Exception as e:
        error_msg = f"Error exporting conversation: {e}"
        logger.error(error_msg)
        return error_msg

def generate_export_content(conversation_history: List[Dict]) -> str:
    """
    Generate the content for conversation export.
    
    Args:
        conversation_history (List[Dict]): The conversation history to export
        
    Returns:
        str: Formatted export content
    """
    export_text = f"{Config.APP_TITLE} - Conversation Export\n"
    export_text += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += "=" * 60 + "\n\n"
    
    # Add conversation summary
    total_exchanges = len(conversation_history)
    levels_used = list(set(exchange["user_level"] for exchange in conversation_history))
    start_time = conversation_history[0]["timestamp"] if conversation_history else "N/A"
    
    export_text += f"Conversation Summary:\n"
    export_text += f"Total Exchanges: {total_exchanges}\n"
    export_text += f"User Levels: {', '.join(levels_used)}\n"
    export_text += f"Started: {start_time}\n"
    export_text += "=" * 60 + "\n\n"
    
    # Add individual exchanges
    for i, msg in enumerate(conversation_history, 1):
        export_text += f"Exchange {i} ({msg['user_level']} level)\n"
        export_text += f"Time: {msg['timestamp']}\n"
        export_text += f"User: {msg['user_message']}\n"
        export_text += f"Mentor: {msg['assistant_response']}\n"
        export_text += "-" * 40 + "\n\n"
    
    return export_text

def validate_user_input(message: str) -> tuple[bool, str]:
    """
    Validate user input for basic security and format checks.
    
    Args:
        message (str): The user's input message
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not message or not message.strip():
        return False, "Message cannot be empty"
    
    # Check for excessively long messages
    if len(message) > 2000:
        return False, "Message is too long (max 2000 characters)"
    
    # Basic content filtering (you can expand this)
    forbidden_patterns = [
        "<?php", "<script", "javascript:", "data:text/html"
    ]
    
    message_lower = message.lower()
    for pattern in forbidden_patterns:
        if pattern in message_lower:
            return False, "Message contains potentially unsafe content"
    
    return True, ""

def format_markdown_response(text: str) -> str:
    """
    Format a response text for better Markdown display.
    
    Args:
        text (str): The response text to format
        
    Returns:
        str: Formatted text with proper Markdown
    """
    # Add some basic formatting improvements
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue
            
        # Format numbered lists
        if line[0].isdigit() and '. ' in line[:5]:
            formatted_lines.append(f"\n{line}")
        # Format bullet points
        elif line.startswith('- ') or line.startswith('* '):
            formatted_lines.append(line)
        # Format headers (simple detection)
        elif line.isupper() and len(line) < 50:
            formatted_lines.append(f"\n## {line.title()}\n")
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def get_system_status() -> Dict:
    """
    Get the current system status for monitoring and debugging.
    
    Returns:
        Dict: System status information
    """
    from rag_engine import get_rag_engine
    
    conversation_manager = get_conversation_manager()
    rag_engine = get_rag_engine()
    
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "conversation": {
                "total_exchanges": conversation_manager.get_history_length(),
                "summary": conversation_manager.get_conversation_summary()
            },
            "rag_engine": rag_engine.get_index_info(),
            "config": {
                "embedding_model": Config.EMBEDDING_MODEL_NAME,
                "llm_model": Config.LLM_MODEL_NAME,
                "docs_directory": Config.DOCS_DIRECTORY,
                "docs_exist": os.path.exists(Config.DOCS_DIRECTORY)
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "error"
        }

def clean_old_exports(max_files: int = 10):
    """
    Clean up old export files to prevent disk space issues.
    
    Args:
        max_files (int): Maximum number of export files to keep
    """
    try:
        # Find all export files
        export_files = [
            f for f in os.listdir('.') 
            if f.startswith(Config.EXPORT_FILENAME_PREFIX) and f.endswith('.txt')
        ]
        
        if len(export_files) <= max_files:
            return
        
        # Sort by modification time (oldest first)
        export_files.sort(key=lambda x: os.path.getmtime(x))
        
        # Remove oldest files
        files_to_remove = export_files[:-max_files]
        
        for file in files_to_remove:
            try:
                os.remove(file)
                logger.info(f"Removed old export file: {file}")
            except Exception as e:
                logger.warning(f"Could not remove file {file}: {e}")
                
    except Exception as e:
        logger.error(f"Error cleaning old exports: {e}")

def log_user_activity(user_level: str, message: str, response_length: int):
    """
    Log user activity for analytics and monitoring.
    
    Args:
        user_level (str): User's expertise level
        message (str): User's message
        response_length (int): Length of the assistant's response
    """
    # This is a simple logger - in production you might want to send to analytics
    logger.info(
        f"User activity - Level: {user_level}, "
        f"Message length: {len(message)}, "
        f"Response length: {response_length}"
    )

# Add robust JSON parsing helper

def parse_json_from_model(text: str, max_attempts: int = 3, delay: float = 0.5) -> Optional[Any]:
    """Try to extract and parse JSON from model text.

    Returns parsed JSON (dict/list) or None.
    """
    if text is None:
        return None

    candidate = text.strip()
    if not candidate:
        return None

    # Remove leading/trailing triple-backtick fences
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)

    for attempt in range(max_attempts):
        try:
            return json.loads(candidate)
        except Exception:
            # Search for JSON array or object substring
            jmatch = re.search(r"(\[\s*\{[\s\S]*?\}\s*\])", candidate, flags=re.S)
            if jmatch:
                try:
                    return json.loads(jmatch.group(1))
                except Exception:
                    pass

            omatch = re.search(r"(\{[\s\S]*?\})", candidate, flags=re.S)
            if omatch:
                try:
                    return json.loads(omatch.group(1))
                except Exception:
                    pass

        time.sleep(delay)

    return None

def grade_quiz_struct(quiz: list, answers: list) -> dict:
    """Grade a quiz given the canonical quiz structure and a list of answers.

    quiz: list of question dicts with keys: question, options, answer, explanation
    answers: list of answers as letters or labeled strings (e.g., 'A' or 'A. option text')

    Returns a dict with score and per-question results.
    """
    results = []
    score = 0
    for i, q in enumerate(quiz):
        user = answers[i] if i < len(answers) else None
        sel = ''
        if isinstance(user, str) and user.strip():
            sel = user.strip()[0].upper()
        correct = (q.get('answer') or '').strip().upper()
        options = q.get('options', [])
        is_correct = (sel == correct and sel != '')
        if is_correct:
            score += 1
        correct_text = ''
        if correct and options:
            try:
                correct_text = options[ord(correct)-65]
            except Exception:
                correct_text = ''
        results.append({
            'question': q.get('question',''),
            'selected': sel,
            'correct': correct,
            'is_correct': is_correct,
            'correct_text': correct_text,
            'explanation': q.get('explanation','')
        })
    return {'score': score, 'total': len(quiz), 'details': results}