# RAG Chatbot for Kyrgyz Laws 🇰🇬

A Retrieval-Augmented Generation (RAG) based chatbot for querying and understanding Kyrgyz Republic laws. This project uses LangChain, FAISS, Hugging Face embeddings, and Google Gemini API to provide accurate legal information retrieval.

## 🌟 Features

- **Multi-lingual support** - Answers in the same language as your question (English, Russian, Kyrgyz)
- **Hybrid search** - Vector similarity (FAISS) + BM25 keyword search
- **Cross-encoder reranking** - Better relevance ranking
- **Query expansion** - Automatic legal term synonyms
- **Conversation history** - Remembers last 2-3 Q&A pairs for context
- **Answer validation** - Ensures responses are relevant and cite sources
- **Efficient text chunking** - Smart article extraction with metadata
- **Gradio web interface** + Console modes
- **Caching system** - Faster responses for repeated questions
- **Funny loading messages** - Entertaining wait experience
- **Comprehensive logging** - Debug and monitoring support

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

## 📂 Project Structure

```
rag_kyrgyz_laws/
├── main.py              # Main entry point
├── config.py            # Configuration settings
├── database.py          # Database initialization & management
├── retrieval.py         # Hybrid search & retrieval logic
├── generation.py        # LLM response generation with Gemini API
├── interface.py         # Gradio web interface
├── console.py           # Console chat interface
├── .env                 # Environment variables (API keys)
├── .env.example         # Environment variables template
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
- **LLM**: Google Gemini Flash (via API)
- **Embeddings**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2

### Search Strategy
- **Vector Search**: FAISS with max marginal relevance
- **Keyword Search**: BM25 for exact term matching
- **Hybrid Weighting**: 70% semantic + 30% keyword
- **Reranking**: Cross-encoder on top 15 results

### Text Processing
- **Chunk Size**: 600 characters
- **Chunk Overlap**: 100 characters
- **Semantic Splitting**: By articles (Статья)
- **Metadata**: Article numbers, law names, source files

### Answer Generation
- **Multi-language**: Detects question language and responds in same language
- **Conversation Context**: Remembers last 2-3 exchanges
- **Smart Truncation**: Preserves sentence boundaries in history
- **Validation**: Checks relevance and citation of sources
- **Streaming**: Real-time response generation

## 🔧 Configuration

Edit `config.py` to customize:
- Model names and parameters
- Chunk sizes and retrieval settings
- Server host and port
- Cache size and paths
- Feature toggles (reranking, BM25, self-consistency)

Edit `.env` for sensitive data:
- Gemini API key
- Other API credentials

## 📝 Logging

Logs are stored in `log/kyrgyz_laws_rag.log` with the following format:
```
{time} {level} {message}
```

Logs include:
- Query processing steps
- Retrieval performance metrics
- API call results
- Error traces

## 🔒 Security

- API keys stored in `.env` file (not committed to git)
- Input validation on user queries
- Answer validation before display
- SSL verification for API calls

## ⚡ Performance

- **Caching**: Instant responses for repeated questions
- **Lazy Loading**: Models loaded once and reused
- **Optimized Retrieval**: Top 8 most relevant chunks
- **Fast API**: Gemini Flash for quick responses (1-3 seconds)

**Response Times:**
- New questions: ~15-20 seconds (retrieval + generation)
- Cached questions: ~2-3 seconds (generation only)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) - RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Hugging Face](https://huggingface.co) - Embeddings and reranking models
- [Gradio](https://gradio.app) - Web interface
- [Google Gemini](https://ai.google.dev/) - LLM API

## 📞 Contact

For questions and support, please open an issue in the GitHub repository.

## 🔮 Future Improvements

- Persistent cache (Redis/SQLite)
- Source citations with clickable links
- PDF export of conversations
- Voice input/output
- Mobile optimization
- Analytics dashboard
- A/B testing for prompts
