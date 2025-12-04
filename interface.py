"""Gradio web interface"""
import gradio as gr
from loguru import logger
from database import get_index_db
from retrieval import get_message_content
from generation import get_model_response_stream
from config import *
import random
import time


db_instance = None

FUNNY_MESSAGES = [
    "🕵️ Hacking government website to get an answer...",
    "📱 Texting Sadyr Japarov...",
    "🏛️ Breaking into the Parliament library...",
    "🚁 Sending a drone to Jogorku Kenesh...",
    "☎️ Calling the Supreme Court...",
    "🔐 Decrypting classified documents...",
    "🎩 Bribing a government official...",
    "🏃 Running to the Ministry of Justice...",
    "🛸 Stealing law book from White House...",
    "📚 Dusting off old law books...",
    "🔍 Hacking into legal databases...",
    "💰 Negotiating with Parliament members...",
    "🚪 Knocking on Supreme Court doors...",
]


def initialize_db():
    """Initialize database on application startup"""
    global db_instance
    if db_instance is None:
        db_instance = get_index_db()
    return db_instance


def truncate_at_sentence(text, max_length=200):
    """Truncate text at sentence boundary"""
    if len(text) <= max_length:
        return text
    
    # Try to find sentence end within max_length
    truncated = text[:max_length]
    
    # Look for sentence endings (., !, ?, ., !, ?)
    for delimiter in ['. ', '! ', '? ', '. ', '! ', '? ']:
        last_pos = truncated.rfind(delimiter)
        if last_pos > max_length * 0.5:  # At least 50% of max_length
            return truncated[:last_pos + 1]
    
    # If no sentence boundary found, truncate at word boundary
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space] + '...'
    
    return truncated + '...'


def process_question(question, history):
    """Process questions in Gradio interface"""
    if not question.strip():
        history.append({"role": "assistant", "content": "❌ Please enter a question"})
        return history
    
    try:
        history.append({"role": "user", "content": question})
        
        # Build conversation history (last 2 Q&A pairs, preserve full last exchange)
        conv_history = ""
        if len(history) > 3:
            # Include previous exchanges (truncated)
            for msg in history[-5:-2]:
                role = "User" if msg["role"] == "user" else "Assistant"
                truncated_content = truncate_at_sentence(msg['content'], 200)
                conv_history += f"{role}: {truncated_content}\n"
            
            # Include full last exchange (no truncation)
            if len(history) >= 2:
                for msg in history[-2:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conv_history += f"{role}: {msg['content']}\n"
        
        history.append({"role": "assistant", "content": random.choice(FUNNY_MESSAGES)})
        yield history
        
        # Start retrieval and generation in background
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            db = initialize_db()
            
            # Retrieval phase
            future = executor.submit(get_message_content, question, db, RETRIEVAL_K)
            while not future.done():
                time.sleep(1)
                history[-1]["content"] = random.choice(FUNNY_MESSAGES)
                yield history
            message_content, is_cached = future.result()
            
            # Keep rotating during generation startup (skip if cached)
            if not is_cached:
                for _ in range(4):
                    time.sleep(3)
                    history[-1]["content"] = random.choice(FUNNY_MESSAGES)
                    yield history
        
        # Start answer streaming
        answer = ""
        for chunk in get_model_response_stream(question, message_content, conv_history):
            answer += chunk
            history[-1]["content"] = answer
            yield history
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        error_msg = "❌ An error occurred while processing your request. Please try rephrasing your question."
        history[-1]["content"] = error_msg
        yield history


def create_gradio_interface():
    """Create Gradio interface"""
    initialize_db()
    
    with gr.Blocks(title="KR Laws Expert") as interface:
        gr.HTML("""
        <div class="header">
            <h1>Legal Expert on Kyrgyz Republic Laws</h1>
            <p>Ask your question about KR legislation and get an expert answer</p>
        </div>
        """)
        
        chatbot = gr.Chatbot(label="Dialogue with Expert")
        
        msg = gr.Textbox(
            label="Your legal question",
            placeholder="For example: What rights does an employee have upon dismissal?",
            lines=2
        )
        
        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
        
        gr.Examples(
            examples=[
                ["What documents are needed to register an LLC?"],
                ["What is the minimum wage in KR?"],
                ["What rights does a consumer have when purchasing goods?"],
                ["What is the statute of limitations for civil cases?"]
            ],
            inputs=msg,
            label="💡 Example Questions"
        )
        
        gr.Markdown("""
        ---
        ### Information:
        - The system analyzes the laws of the Kyrgyz Republic and provides answers based on official documents
        - Answers are for informational purposes and do not replace professional legal advice
        - For accurate legal assistance, consult a qualified lawyer
        """)
        
        def clear_chat():
            return []
        
        def submit_and_clear(message, history):
            for updated_history in process_question(message, history):
                yield updated_history, ""
        
        submit_btn.click(
            submit_and_clear,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            submit_and_clear,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot]
        )
    
    return interface
