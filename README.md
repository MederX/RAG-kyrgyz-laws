# RAG Chatbot for Kyrgyz Laws 🇰🇬

A Retrieval-Augmented Generation (RAG) based chatbot for querying and understanding Kyrgyz Republic laws. This project uses LangChain, FAISS, and Hugging Face embeddings to provide accurate legal information retrieval.

## 🌟 Features

- Multi-lingual support (Kyrgyz/Russian) using paraphrase-multilingual-MiniLM-L12-v2
- Hybrid search: Vector similarity (FAISS) + BM25 keyword search
- Cross-encoder reranking for better relevance
- Query expansion with legal term synonyms
- Self-consistency for reliable answers
- Efficient text chunking with article extraction
- Gradio-based user interface + Console modes
- Caching system for faster responses
- Logging system for debugging and monitoring

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/MederX/RAG-kyrgyz-laws.git
cd RAG-kyrgyz-laws
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r req.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
# Get your key from: https://aistudio.google.com/app/apikey
```

5. Pull the LLM model (optional, now using Gemini API):
```bash
ollama pull phi3:3.8b
```

## 📂 Project Structure

```
rag_kyrgyz_laws/
├── main.py              # Main entry point
├── config.py            # Configuration settings
├── database.py          # Database initialization & management
├── retrieval.py         # Hybrid search & retrieval logic
├── generation.py        # LLM response generation
├── interface.py         # Gradio web interface
├── console.py           # Console chat interface
├── laws/                # Text files containing laws
├── db/                  # FAISS vector database
├── log/                 # Log files
├── req.txt              # Project dependencies
└── README.md
```

## 🚀 Usage

1. Place your law text files in the `laws/` directory (UTF-8 encoded .txt files)

2. Run the application:
```bash
python main.py
```

3. Choose mode:
   - **1**: Gradio Web Interface (http://localhost:7860)
   - **2**: Interactive Console Chat
   - **3**: Single Question Mode

## 💻 Technical Details

### Models
- **LLM**: phi3:3.8b (Microsoft's reasoning model)
- **Embeddings**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2

### Search Strategy
- **Vector Search**: FAISS with max marginal relevance
- **Keyword Search**: BM25 for exact term matching
- **Hybrid Weighting**: 70% semantic + 30% keyword
- **Reranking**: Cross-encoder on top 20 results

### Text Processing
- **Chunk Size**: 600 characters
- **Chunk Overlap**: 100 characters
- **Semantic Splitting**: By articles (Статья)
- **Metadata**: Article numbers, law names, source files

## 🔧 Configuration

Edit `config.py` to customize:
- Model names and parameters
- Chunk sizes and retrieval settings
- Server host and port
- Cache size and paths

## 📝 Logging

Logs are stored in `log/kyrgyz_laws_rag.log` with the following format:
```
{time} {level} {message}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face](https://huggingface.co)
- [Gradio](https://gradio.app)
- [Ollama](https://ollama.ai)

## 📞 Contact

For questions and support, please open an issue in the GitHub repository.
