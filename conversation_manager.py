"""
Conversation manager for handling chat history, context, and conversation persistence.
"""
from datetime import datetime
from typing import List, Dict, Optional
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages conversation history, context extraction, and conversation persistence.
    """
    
    def __init__(self):
        """Initialize the conversation manager."""
        self.conversation_history: List[Dict] = []
        
    def add_exchange(self, user_message: str, assistant_response: str, user_level: str):
        """
        Add a new conversation exchange to the history.
        
        Args:
            user_message (str): The user's message
            assistant_response (str): The assistant's response
            user_level (str): The user's expertise level
        """
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "user_level": user_level
        }
        
        self.conversation_history.append(exchange)
        logger.info(f"Added conversation exchange (total: {len(self.conversation_history)})")
    
    def get_conversation_context(self, max_context: Optional[int] = None) -> str:
        """
        Get recent conversation context for the LLM.
        
        Args:
            max_context (int, optional): Maximum number of context messages to include.
                                       Defaults to Config.MAX_CONTEXT_MESSAGES.
        
        Returns:
            str: Formatted conversation context
        """
        if not self.conversation_history:
            return ""
        
        if max_context is None:
            max_context = Config.MAX_CONTEXT_MESSAGES
        
        # Get recent conversation exchanges
        context_messages = (
            self.conversation_history[-max_context:] 
            if len(self.conversation_history) > max_context 
            else self.conversation_history
        )
        
        context = "\n\nPrevious conversation context:\n"
        
        for msg in context_messages:
            context += f"User: {msg['user_message']}\n"
            context += f"Assistant: {msg['assistant_response']}\n\n"
        
        return context
    
    def get_chat_history_for_display(self) -> List[List[str]]:
        """
        Get conversation history formatted for Gradio chatbot display.
        
        Returns:
            List[List[str]]: List of [user_message, assistant_response] pairs
        """
        chat_display = []
        for msg in self.conversation_history:
            chat_display.append([msg["user_message"], msg["assistant_response"]])
        return chat_display
    
    def clear_history(self):
        """Clear all conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_history_length(self) -> int:
        """
        Get the number of exchanges in the conversation history.
        
        Returns:
            int: Number of conversation exchanges
        """
        return len(self.conversation_history)
    
    def get_last_exchange(self) -> Optional[Dict]:
        """
        Get the last conversation exchange.
        
        Returns:
            Dict or None: Last exchange or None if no history
        """
        if not self.conversation_history:
            return None
        return self.conversation_history[-1]
    
    def get_exchanges_by_level(self, user_level: str) -> List[Dict]:
        """
        Get all exchanges for a specific user level.
        
        Args:
            user_level (str): The user level to filter by
            
        Returns:
            List[Dict]: List of exchanges for the specified level
        """
        return [
            exchange for exchange in self.conversation_history
            if exchange["user_level"] == user_level
        ]
    
    def get_conversation_summary(self) -> Dict:
        """
        Get a summary of the conversation statistics.
        
        Returns:
            Dict: Summary statistics
        """
        if not self.conversation_history:
            return {
                "total_exchanges": 0,
                "levels_used": [],
                "start_time": None,
                "last_activity": None
            }
        
        levels_used = list(set(
            exchange["user_level"] for exchange in self.conversation_history
        ))
        
        return {
            "total_exchanges": len(self.conversation_history),
            "levels_used": levels_used,
            "start_time": self.conversation_history[0]["timestamp"],
            "last_activity": self.conversation_history[-1]["timestamp"]
        }
    
    def search_history(self, search_term: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Search conversation history for a specific term.
        
        Args:
            search_term (str): Term to search for
            case_sensitive (bool): Whether search should be case sensitive
            
        Returns:
            List[Dict]: List of exchanges containing the search term
        """
        if not case_sensitive:
            search_term = search_term.lower()
        
        matching_exchanges = []
        
        for exchange in self.conversation_history:
            user_msg = exchange["user_message"]
            assistant_msg = exchange["assistant_response"]
            
            if not case_sensitive:
                user_msg = user_msg.lower()
                assistant_msg = assistant_msg.lower()
            
            if search_term in user_msg or search_term in assistant_msg:
                matching_exchanges.append(exchange)
        
        return matching_exchanges

# Create a global instance
_conversation_manager_instance = None

def get_conversation_manager() -> ConversationManager:
    """
    Get the global conversation manager instance (singleton pattern).
    
    Returns:
        ConversationManager: The conversation manager instance
    """
    global _conversation_manager_instance
    
    if _conversation_manager_instance is None:
        _conversation_manager_instance = ConversationManager()
    
    return _conversation_manager_instance