"""
Gradio UI interface for the Cybersecurity AI Mentor application.
"""
import gradio as gr
from groq import Groq
from config import Config
from rag_engine import get_rag_engine
from conversation_manager import get_conversation_manager
from utils import export_conversation_to_file, validate_user_input, format_markdown_response, log_user_activity, parse_json_from_model
import logging
import json
import re

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
        # Store the last generated quiz as a list of dicts: {question, options, answer, explanation}
        self.last_quiz: list[dict] = []
    
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

    def generate_mcqs(self, num_questions: int, user_level: str) -> str:
        """
        Generate multiple-choice questions (MCQs) based on the recent conversation context.

        Args:
            num_questions (int): Number of MCQs to generate
            user_level (str): The user's expertise level to tailor question difficulty

        Returns:
            str: Markdown-formatted MCQs suitable for display in the UI
        """
        # Get recent conversation context
        conversation_context = self.conversation_manager.get_conversation_context()

        if not conversation_context or conversation_context.strip() == "":
            return "**No conversation history available.** Chat with the mentor first, then take a quiz based on what you've learned."

        # Build a prompt for the LLM to generate MCQs
        prompt = (
            f"You are an educational cybersecurity mentor. Based on the conversation context below, "
            f"generate {int(num_questions)} multiple-choice questions tailored for a {user_level} learner.\n\n"
            f"Requirements:\n"
            f"- Provide {int(num_questions)} questions.\n"
            f"- Each question should have four options labeled A, B, C, D.\n"
            f"- After each question, include an Answer: <LETTER> line and a 1-2 sentence explanation for the correct answer.\n"
            f"- Format the output using Markdown so it is easy to display (use headings, numbered lists, and code/quote blocks if helpful).\n\n"
            f"Conversation Context:\n{conversation_context}\n\n"
            f"Generate the questions now."
        )

        try:
            logger.info("Generating MCQs from conversation context")
            raw_response = self._get_groq_response(prompt)
            formatted = format_markdown_response(raw_response)
            return formatted
        except Exception as e:
            logger.error(f"Error generating MCQs: {e}")
            return f"**Error generating quiz:** {e}"

    def generate_mcqs_struct(self, user_level: str,
                           r1, r2, r3, r4, r5) -> tuple:
        """
        Generate MCQs and populate the pre-created Radio components (fixed 5 questions).
        Returns updates for the 5 radio components and the grade output.
        """
        # Normalize user_level (dropdown multiselect may pass list)
        if isinstance(user_level, (list, tuple)):
            user_level_str = ', '.join(user_level) if user_level else Config.DEFAULT_USER_LEVEL
        else:
            user_level_str = user_level or Config.DEFAULT_USER_LEVEL

        conversation_context = self.conversation_manager.get_conversation_context()
        if not conversation_context or conversation_context.strip() == "":
            # Return a single visible message and hide all question components
            hidden_rad = [gr.update(visible=False) for _ in range(5)]
            return (*hidden_rad, gr.update(value="**No conversation history available.** Chat first before taking a quiz.", visible=True))

        # JSON-output prompt
        prompt = (
            f"You are an educational cybersecurity mentor. Based on the conversation context below, "
            f"generate 5 multiple-choice questions tailored for a {user_level_str} learner.\n\n"
            f"Return the output as JSON: a list of objects with keys 'question', 'options' (array of 4 strings), "
            f"'answer' (one of 'A','B','C','D'), and 'explanation'.\n\n"
            f"Conversation Context:\n{conversation_context}\n\n"
            f"Generate the questions now."
        )

        try:
            logger.info("Generating structured MCQs from conversation context")
            raw = self._get_groq_response(prompt)

            # Use the robust parser from utils
            parsed = parse_json_from_model(raw)
            logger.debug(f"Raw quiz generator output: {raw}")
            if not parsed:
                logger.error("Could not parse JSON from Groq response")
                hidden_rad = [gr.update(visible=False) for _ in range(5)]
                return (*hidden_rad, gr.update(value="**Error: Could not parse quiz from generator response.**", visible=True))

            # Normalize various possible parsed structures into a list of question dicts
            quiz_list = None
            if isinstance(parsed, list):
                quiz_list = parsed
            elif isinstance(parsed, dict):
                if "questions" in parsed and isinstance(parsed["questions"], list):
                    quiz_list = parsed["questions"]
                elif all(k in parsed for k in ("question", "options")):
                    quiz_list = [parsed]
                else:
                    for v in parsed.values():
                        if isinstance(v, list):
                            quiz_list = v
                            break
            elif isinstance(parsed, str):
                try:
                    maybe = json.loads(parsed)
                    if isinstance(maybe, list):
                        quiz_list = maybe
                    elif isinstance(maybe, dict) and "questions" in maybe:
                        quiz_list = maybe["questions"]
                except Exception:
                    quiz_list = None

            if not isinstance(quiz_list, list):
                logger.error("Parsed quiz is not a list; raw output: %s", raw)
                hidden_rad = [gr.update(visible=False) for _ in range(5)]
                return (*hidden_rad, gr.update(value="**Error: Unexpected quiz format from generator.**", visible=True))

            # Truncate or pad to 5
            quiz_list = quiz_list[:5]
            # Normalize each question dict to have keys: question, options (list), answer (letter), explanation
            normalized = []
            for q in quiz_list:
                try:
                    question_text = q.get('question') if isinstance(q, dict) else str(q)
                    options = q.get('options', []) if isinstance(q, dict) else []
                    answer = q.get('answer', '') if isinstance(q, dict) else ''
                    explanation = q.get('explanation', '') if isinstance(q, dict) else ''
                    options = [str(o) for o in (options[:4] or [])]
                    while len(options) < 4:
                        options.append('')
                    normalized.append({
                        'question': question_text,
                        'options': options,
                        'answer': answer.strip().upper() if isinstance(answer, str) else '',
                        'explanation': explanation
                    })
                except Exception:
                    continue

            self.last_quiz = normalized

            # Prepare updates for radio components
            rad_updates = []
            radio_components = [r1, r2, r3, r4, r5]

            for i in range(5):
                if i < len(normalized):
                    q = normalized[i]
                    # remove bold markers from question if present
                    q_text = f"Q{i+1}. {q.get('question','')}"
                    options = q.get('options', [])

                    # Format options with A,B,C,D labels
                    labeled_options = [f"{chr(65+idx)}. {opt}" for idx, opt in enumerate(options[:4])]
                    options_text = "\n".join(labeled_options)

                    # Show full labeled options in radio choices
                    rad_updates.append(gr.update(
                        label=q_text,
                        info=options_text,
                        choices=labeled_options,
                        value=None,
                        visible=True
                    ))
                else:
                    rad_updates.append(gr.update(visible=False))

            # Return tuple: 5 radio updates, grade_out cleared and hidden
            return (*rad_updates, gr.update(value="", visible=False))

        except Exception as e:
            logger.error(f"Error generating structured MCQs: {e}")
            hidden_rad = [gr.update(visible=False) for _ in range(5)]
            return (*hidden_rad, gr.update(value=f"**Error generating quiz:** {e}", visible=True))

    def grade_quiz(self, a1, a2, a3, a4, a5) -> gr.update:
        """
        Grade the submitted answers against `self.last_quiz` and return a Markdown report.
        Takes radio button values (A,B,C,D letters or labeled strings) as inputs and returns a formatted feedback string.
        """
        answers = [a1, a2, a3, a4, a5]
        if not self.last_quiz:
            return gr.update(value="**No quiz to grade.** Generate a quiz first.", visible=True)

        feedback_lines = []
        score = 0
        total = len(self.last_quiz)
        for i, q in enumerate(self.last_quiz):
            selected = answers[i] if i < len(answers) else None
            correct = (q.get('answer') or '').strip()
            options = q.get('options', [])

            # Normalize selected letter (handles both 'A' and 'A. option')
            sel_letter = ''
            if isinstance(selected, str) and selected.strip():
                sel_letter = selected.strip()[0]

            # Determine correct option text
            correct_text = ''
            if correct:
                try:
                    idx = ord(correct.upper()) - 65
                    if 0 <= idx < len(options):
                        correct_text = options[idx]
                except Exception:
                    correct_text = ''

            if sel_letter and correct and sel_letter.upper() == correct.upper():
                score += 1
                feedback_lines.append(f"Q{i+1}. Correct ✅ - {q.get('question','')}")
            else:
                # Unanswered
                if not sel_letter:
                    feedback_lines.append(f"Q{i+1}. No answer provided ❌ - {q.get('question','')}\nCorrect: **{correct}. {correct_text}** - {q.get('explanation','')}")
                else:
                    feedback_lines.append(f"Q{i+1}. Incorrect ❌ - {q.get('question','')}\nYour answer: **{sel_letter}**\nCorrect: **{correct}. {correct_text}** - {q.get('explanation','')}")

        summary = f"**Score: {score}/{total}**\n\n"
        return gr.update(value=summary + "\n\n".join(feedback_lines), visible=True)
    
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
                    # Quiz area placed under the chat for more visibility
                    with gr.Column():
                        gr.Markdown("---\n## 📝 Quiz")
                        # Single-column radio groups for exactly 5 questions
                        with gr.Row():
                            with gr.Column():
                                r1 = gr.Radio(choices=["A", "B", "C", "D"], label="Question 1", info="The full question text and options will appear here", visible=False)
                                r2 = gr.Radio(choices=["A", "B", "C", "D"], label="Question 2", info="The full question text and options will appear here", visible=False)
                                r3 = gr.Radio(choices=["A", "B", "C", "D"], label="Question 3", info="The full question text and options will appear here", visible=False)
                                r4 = gr.Radio(choices=["A", "B", "C", "D"], label="Question 4", info="The full question text and options will appear here", visible=False)
                                r5 = gr.Radio(choices=["A", "B", "C", "D"], label="Question 5", info="The full question text and options will appear here", visible=False)
                        # Buttons for the quiz moved into the quiz area for easier access
                        with gr.Row():
                            quiz_btn = gr.Button("Take a Quiz", variant="primary")
                            submit_btn = gr.Button("Submit Quiz", variant="primary")
                        grade_out = gr.Markdown("Your quiz results will appear here.", visible=False)
                
                with gr.Column(scale=1):
                    # Settings and controls
                    gr.Markdown("### ⚙️ Settings")
                    user_level = gr.Dropdown(
                        choices=Config.USER_LEVELS,
                        value=Config.DEFAULT_USER_LEVEL,
                        label="Your Level",
                        info="Select your cybersecurity expertise level",
                        interactive=None
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
            
            # Quiz event - generate structured quiz and populate the 5 pre-created components
            quiz_btn.click(
                self.generate_mcqs_struct,
                inputs=[user_level, r1, r2, r3, r4, r5],
                outputs=[r1, r2, r3, r4, r5, grade_out]
            )

            # Submit event - grade quiz
            submit_btn.click(
                self.grade_quiz,
                inputs=[r1, r2, r3, r4, r5],
                outputs=[grade_out]
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