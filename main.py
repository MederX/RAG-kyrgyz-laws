"""Main entry point for RAG Kyrgyz Laws chatbot"""
from loguru import logger
from interface import create_gradio_interface
from console import interactive_chat, single_question
from config import *

logger.add(LOG_PATH, format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")


def main():
    """Main application entry point"""
    print("Legal Expert on KR Laws")
    print("Choose launch mode:")
    print("1 - Gradio Web Interface")
    print("2 - Interactive Console Chat")
    print("3 - Single Question in Console")
    
    mode = input("Enter mode number (1-3): ").strip()
    
    if mode == "1" or mode == "":
        interface = create_gradio_interface()
        print("🌐 Launching web interface...")
        print(f"📱 Interface will be available at: http://{SERVER_HOST}:{SERVER_PORT}")
        print("⏹️  Press Ctrl+C to stop")
        
        interface.launch(
            server_name=SERVER_HOST,
            server_port=SERVER_PORT,
            share=True,
            debug=False,
            show_error=True,
            inbrowser=True
        )
    elif mode == "2":
        interactive_chat()
    else:
        single_question()


if __name__ == "__main__":
    main()
