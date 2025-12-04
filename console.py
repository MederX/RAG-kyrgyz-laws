"""Console chat interface"""
from loguru import logger
from database import get_index_db
from retrieval import get_message_content
from generation import get_model_response
from config import *
import random

FUNNY_MESSAGES = [
    "🕵️ Hacking government website...",
    "📱 Texting the Sadyr Japarov...",
    "🏛️ Breaking into the Parliament library...",
    "🚁 Sending a drone to Jogorku Kenesh...",
    "☎️ Calling the Supreme Court...",
    "🔐 Decrypting classified documents...",
    "🎩 Bribing a government official...",
    "🏃 Running to the Ministry of Justice...",
    "🧙 Summoning ancient legal spirits...",
    "🛸 Stealing law book from White House...",
    "📚 Dusting off old law books...",
    "🔍 Hacking into legal databases...",
    "💰 Negotiating with Parliament members...",
    "🎭 Disguising as a judge...",
    "🚪 Knocking on Supreme Court doors...",
]


def truncate_at_sentence(text, max_length=200):
    """Truncate text at sentence boundary"""
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    
    # Look for sentence endings
    for delimiter in ['. ', '! ', '? ', '. ', '! ', '? ']:
        last_pos = truncated.rfind(delimiter)
        if last_pos > max_length * 0.5:
            return truncated[:last_pos + 1]
    
    # Truncate at word boundary
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space] + '...'
    
    return truncated + '...'


def interactive_chat():
    """Interactive chat mode"""
    print("🏛️  Legal Expert Chatbot for Kyrgyz Republic Laws")
    print("📖 Type 'exit' to quit\n")
    
    db = get_index_db()
    conversation_history = []
    
    while True:
        topic = input("❓ Your legal question: ").strip()
        
        if topic.lower() in ['выход', 'exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not topic:
            print("❌ Please enter a question\n")
            continue
            
        try:
            print(random.choice(FUNNY_MESSAGES))
            
            message_content, is_cached, confidence_score = get_message_content(topic, db, RETRIEVAL_K)
            if is_cached:
                print("⚡ Using cached results...")
            
            # Build history (last 2 Q&A pairs, preserve full last exchange)
            history_text = ""
            if len(conversation_history) > 2:
                # Previous exchanges (truncated)
                for h in conversation_history[-4:-2]:
                    truncated = truncate_at_sentence(h['content'], 200)
                    history_text += f"{h['role']}: {truncated}\n"
                
                # Last exchange (full)
                for h in conversation_history[-2:]:
                    history_text += f"{h['role']}: {h['content']}\n"
            
            answer = get_model_response(topic, message_content, history_text)
            
            print(f"\n📋 Legal Expert Answer:")
            print(f"{'='*50}")
            print(answer)
            print(f"{'='*50}\n")
            
            # Save to history
            conversation_history.append({"role": "user", "content": topic})
            conversation_history.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print("❌ An error occurred. Please try rephrasing your question.\n")


def single_question():
    """Single question mode"""
    db = get_index_db()
    topic = input("Enter your legal question: ")
    message_content, _ = get_message_content(topic, db, RETRIEVAL_K)
    answer = get_model_response(topic, message_content)
    print("\n📋 Model Answer:")
    print(f"{'='*50}")
    print(answer)
    print(f"{'='*50}")
