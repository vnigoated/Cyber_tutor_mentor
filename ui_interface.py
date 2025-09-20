"""
Gradio UI interface for the Cybersecurity AI Mentor application.
"""
import gradio as gr
from groq import Groq
from config import Config
from rag_engine import get_rag_engine
from conversation_manager import get_conversation_manager
from utils import export_conversation_to_file, validate_user_input, format_markdown_response, log_user_activity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CyberMentorUI:
    """
    Handles the Gradio user interface for the Cybersecurity AI Mentor.
    """
    
    def __init__(self):
        """Initialize the UI with necessary components."""
        self.rag_engine = get_rag_engine()
        self.conversation_manager = get_conversation_manager()
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
    
    def chat_with_mentor(self, message: str, history: list, user_level: str) -> tuple:
        """
        Handle a chat message from the user.
        
        Args:
            message (str): User's message
            history (list): Chat history for display
            user_level (str): User's expertise level
            
        Returns:
            tuple: (updated_history, empty_message_box)
        """
        # Validate input
        is_valid, error_msg = validate_user_input(message)
        if not is_valid:
            logger.warning(f"Invalid input: {error_msg}")
            if history is None:
                history = []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ Error: {error_msg}"})
            return history, ""
        
        try:
            # Get RAG context
            logger.info("Querying RAG engine for context")
            rag_context = self.rag_engine.query(message)
            
            # Get conversation context
            conversation_context = self.conversation_manager.get_conversation_context()
            
            # Create enhanced prompt with conversation memory
            prompt = self._create_prompt(message, rag_context, conversation_context, user_level)
            
            # Get response from Groq
            logger.info("Getting response from Groq")
            assistant_response = self._get_groq_response(prompt)
            
            # Format the response
            formatted_response = format_markdown_response(assistant_response)
            
            # Update conversation history
            self.conversation_manager.add_exchange(message, formatted_response, user_level)
            
            # Update chat display
            if history is None:
                history = []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": formatted_response})
            
            # Log activity
            log_user_activity(user_level, message, len(formatted_response))
            
            return history, ""
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            logger.error(f"Error in chat_with_mentor: {e}")
            
            if history is None:
                history = []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    def _create_prompt(self, message: str, rag_context: str, conversation_context: str, user_level: str) -> str:
        """
        Create the prompt for the LLM with all necessary context.
        
        Args:
            message (str): User's current message
            rag_context (str): Retrieved context from documents
            conversation_context (str): Previous conversation context
            user_level (str): User's expertise level
            
        Returns:
            str: Complete prompt for the LLM
        """
        # Check if RAG context is meaningful (not empty or generic)
        has_relevant_context = (
            rag_context and 
            len(rag_context.strip()) > 50 and  # Has substantial content
            not rag_context.strip().startswith("I don't") and  # Not a "don't know" response
            not "no relevant" in rag_context.lower()  # Not a "no relevant info" response
        )
        
        if has_relevant_context:
            # Use RAG-based response for cybersecurity topics
            prompt = (
                f"You are a cybersecurity mentor for a {user_level} learner.\n"
                f"Use the following retrieved context from cybersecurity documents to answer the question.\n"
                f"Consider the conversation history to provide contextual responses.\n"
                f"Be educational, encouraging, and adapt your language to the user's level.\n\n"
                f"Retrieved Context:\n{rag_context}\n"
                f"{conversation_context}"
                f"Current Question: {message}\n\n"
                f"Instructions:\n"
                f"- Provide helpful, educational responses based on the retrieved context\n"
                f"- Use examples relevant to {user_level} level\n"
                f"- Format your response with proper markdown for readability\n"
                f"- Keep responses comprehensive but not overwhelming\n"
                f"- Focus on cybersecurity concepts from the provided context\n"
            )
        else:
            # Limited response for non-cybersecurity topics
            prompt = (
                f"You are a cybersecurity mentor for a {user_level} learner.\n"
                f"The user asked: {message}\n"
                f"{conversation_context}"
                f"\nThis question doesn't appear to be related to cybersecurity topics covered in your knowledge base.\n"
                f"Provide a brief, helpful response that:\n"
                f"1. Acknowledges their question\n"
                f"2. Explains that you specialize in cybersecurity topics\n"
                f"3. Suggests how they might rephrase the question to be cybersecurity-related\n"
                f"4. Offers to help with cybersecurity concepts instead\n"
                f"Keep the response short and redirect them to cybersecurity topics.\n"
            )
        
        return prompt
    
    def _get_groq_response(self, prompt: str) -> str:
        """
        Get response from Groq API.
        
        Args:
            prompt (str): The prompt to send to Groq
            
        Returns:
            str: The assistant's response
        """
        try:
            response = self.groq_client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                top_p=Config.TOP_P,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting Groq response: {e}")
            raise
    
    def clear_chat(self) -> list:
        """Clear the conversation history."""
        self.conversation_manager.clear_history()
        logger.info("Chat cleared by user")
        return []
    
    def export_chat(self) -> dict:
        """Export the conversation to a file."""
        result = export_conversation_to_file()
        return gr.update(value=result, visible=True)
    
    def create_interface(self) -> gr.Blocks:
        """
        Create and return the Gradio interface.
        
        Returns:
            gr.Blocks: The complete Gradio interface
        """
        # Choose theme based on config
        theme = getattr(gr.themes, Config.THEME.title(), gr.themes.Soft)()
        
        with gr.Blocks(title=Config.APP_TITLE, theme=theme) as demo:
            # Header
            gr.Markdown(f"# 🛡️ {Config.APP_TITLE}")
            gr.Markdown(Config.APP_DESCRIPTION)
            
            with gr.Row():
                with gr.Column(scale=4):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        height=Config.CHAT_HEIGHT,
                        show_label=False,
                        container=True,
                        bubble_full_width=False,
                        type="messages"  # Updated to new format to avoid deprecation warning
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask me anything about cybersecurity...",
                            show_label=False,
                            scale=4,
                            container=False,
                            max_lines=3
                        )
                        send_btn = gr.Button("Send", scale=1, variant="primary")
                
                with gr.Column(scale=1):
                    # Settings and controls
                    gr.Markdown("### ⚙️ Settings")
                    user_level = gr.Dropdown(
                        choices=Config.USER_LEVELS,
                        value=Config.DEFAULT_USER_LEVEL,
                        label="Your Level",
                        info="Select your cybersecurity expertise level"
                    )
                    
                    gr.Markdown("### 🎛️ Chat Controls")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    export_btn = gr.Button("📄 Export Chat", variant="secondary")
                    export_status = gr.Textbox(
                        label="Export Status",
                        interactive=False,
                        visible=False
                    )
                    
                    # Tips and sample questions
                    gr.Markdown("### 💡 Tips")
                    tips_text = "\n".join(f"- {tip}" for tip in Config.TIPS)
                    gr.Markdown(tips_text)
                    
                    gr.Markdown("### 🎯 Sample Questions")
                    samples_text = "\n".join(f'- "{question}"' for question in Config.SAMPLE_QUESTIONS)
                    gr.Markdown(samples_text)
                    
                    # System status (for debugging)
                    with gr.Accordion("🔧 System Info", open=False):
                        gr.Markdown(
                            f"**Model**: {Config.LLM_MODEL_NAME}\n"
                            f"**Embeddings**: {Config.EMBEDDING_MODEL_NAME}\n"
                            f"**Docs**: {Config.DOCS_DIRECTORY}"
                        )
            
            # Event handlers
            send_btn.click(
                self.chat_with_mentor,
                inputs=[msg_input, chatbot, user_level],
                outputs=[chatbot, msg_input]
            )
            
            msg_input.submit(
                self.chat_with_mentor,
                inputs=[msg_input, chatbot, user_level],
                outputs=[chatbot, msg_input]
            )
            
            clear_btn.click(
                self.clear_chat,
                outputs=[chatbot]
            )
            
            export_btn.click(
                self.export_chat,
                outputs=[export_status]
            )
        
        return demo

def create_app() -> gr.Blocks:
    """
    Create and return the Gradio application.
    
    Returns:
        gr.Blocks: The complete Gradio application
    """
    ui = CyberMentorUI()
    return ui.create_interface()