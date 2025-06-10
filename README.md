# RAG Chatbot for Kyrgyz Laws 🇰🇬

A Retrieval-Augmented Generation (RAG) based chatbot for querying and understanding Kyrgyz Republic laws. This project uses LangChain, FAISS, and Hugging Face embeddings to provide accurate legal information retrieval.

## 🌟 Features

- Multi-lingual support (Kyrgyz/Russian) using paraphrase-multilingual-MiniLM-L12-v2
- Vector similarity search with FAISS
- Efficient text chunking and retrieval
- Gradio-based user interface
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
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
rag_kyrgyz_laws/
├── laws/              # Text files containing laws
├── db/                # FAISS vector database
├── log/              # Log files
├── rag_chat_final.py  # Main application file
├── requirements.txt   # Project dependencies
└── README.md
```

## 🚀 Usage

1. Place your law text files in the `laws/` directory (UTF-8 encoded .txt files)

2. Run the application:
```bash
python rag_chat_final.py
```

3. Access the Gradio interface in your browser at `http://localhost:7860`

## 💻 Technical Details

- **Embedding Model**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Vector Store**: FAISS
- **Text Chunking**: RecursiveCharacterTextSplitter (800 chars, 150 overlap)
- **UI Framework**: Gradio

## 🔧 Configuration

The system uses the following default settings:
- Chunk size: 800 characters
- Chunk overlap: 150 characters
- CUDA support enabled (can be modified for CPU)
- Log rotation: 100 KB with zip compression

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

## 📞 Contact

For questions and support, please open an issue in the GitHub repository.
