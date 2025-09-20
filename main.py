import os
import json
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from groq import Groq
import gradio as gr

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embedding model (local HuggingFace to avoid OpenAI)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Tell LlamaIndex not to use any LLM by default
Settings.llm = None
Settings.embed_model = embed_model

def ingest_docs():
    reader = SimpleDirectoryReader("docs")
    return reader.load_data()

def build_index(docs):
    return VectorStoreIndex.from_documents(docs, embed_model=embed_model)

def gradio_interface():
    docs = ingest_docs()
    index = build_index(docs)

    query_engine = index.as_query_engine(similarity_top_k=3)
    client = Groq(api_key=GROQ_API_KEY)

    def mentor_fn(user_level, query):
        response = query_engine.query(query)
        context = response.response

        prompt = (
            f"You are a cybersecurity mentor for a {user_level} learner.\n"
            f"Use the following retrieved context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

        reply = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
            top_p=0.9,
        )
        return reply.choices[0].message.content

    def generate_quiz(quiz_topic):
        prompt = f"Generate 3 multiple-choice cybersecurity questions on {quiz_topic}. For each, include options, correct answer, and explanation in JSON format."
        reply = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
            top_p=0.9,
        )

        try:
            quiz = json.loads(reply.choices[0].message.content)
        except:
            quiz = [
                {
                    "question": "What does TCP stand for?",
                    "options": [
                        "Transfer Control Protocol",
                        "Transmission Control Protocol",
                        "Transport Communication Process",
                    ],
                    "answer": "Transmission Control Protocol",
                    "explanation": "TCP ensures reliable data transfer.",
                }
            ]
        return quiz

    def check_answers(user_answers, quiz_data):
        score = 0
        feedback = ""
        for i, (ans, q) in enumerate(zip(user_answers, quiz_data)):
            if ans == q["answer"]:
                score += 1
                feedback += f"✅ Q{i+1}: Correct! {q['explanation']}\n\n"
            else:
                feedback += f"❌ Q{i+1}: Wrong. Correct: {q['answer']} — {q['explanation']}\n\n"
        feedback += f"Final Score: {score}/{len(quiz_data)}"
        return feedback

    with gr.Blocks() as demo:
        gr.Markdown("# Cybersecurity Knowledge Mentor (RAG + Groq)")

        with gr.Tab("Ask Mentor"):
            user_level = gr.Dropdown(
                ["beginner", "intermediate", "expert"],
                label="Your expertise",
                value="beginner",
            )
            query = gr.Textbox(label="Ask a cybersecurity question")
            output = gr.Textbox(label="Mentor Response", lines=10)
            btn = gr.Button("Submit")
            btn.click(mentor_fn, inputs=[user_level, query], outputs=output)

        with gr.Tab("Quiz"):
            quiz_topic = gr.Textbox(label="Enter quiz topic")
            gen_btn = gr.Button("Generate Quiz")
            quiz_state = gr.State([])  # stores quiz data
            answer_state = gr.State([])  # stores radio components
            quiz_result = gr.Textbox(label="Results", lines=10)
            submit_btn = gr.Button("Submit Answers", visible=False)

            def load_quiz(topic):
                quiz = generate_quiz(topic)
                radios = []
                with gr.Column():
                    for q in quiz:
                        gr.Markdown(f"**{q['question']}**")
                        radios.append(gr.Radio(choices=q["options"], label="Answer"))
                return gr.update(visible=True), quiz, radios

            gen_btn.click(
                load_quiz,
                inputs=quiz_topic,
                outputs=[submit_btn, quiz_state, answer_state],
            )

            def evaluate(*args):
                # last arg is quiz_state, rest are answers
                *answers, quiz_data = args
                return check_answers(answers, quiz_data)

            submit_btn.click(
                evaluate,
                inputs=[answer_state, quiz_state],
                outputs=quiz_result,
            )

    demo.launch()

if __name__ == "__main__":
    gradio_interface()
