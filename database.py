"""Database initialization and management"""
import os
import ssl
import certifi
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from config import *

# Disable SSL verification warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set SSL context to not verify
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


def get_embeddings():
    """Initialize embedding model"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
    )


def get_index_db():
    """Load or create FAISS vector database"""
    logger.debug('...get_index_db')
    embeddings = get_embeddings()
    file_path = DB_PATH + "/index.faiss"
    
    if os.path.exists(file_path):
        logger.debug('Loading existing database')
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.debug('Creating new knowledge base from .txt files')
        documents = []
        loaded_files = []
        
        for root, dirs, files in os.walk(LAWS_DIR):
            for file in files:
                if file.endswith(".txt"):
                    try:
                        logger.debug(f'Loading file: {file}')
                        file_path = os.path.join(root, file)
                        loader = TextLoader(file_path, encoding='utf-8')
                        file_docs = loader.load()
                        
                        for doc in file_docs:
                            doc.metadata['source_file'] = file
                            doc.metadata['law_name'] = file.replace('.txt', '').replace('_', ' ')
                            # Extract article numbers
                            article_match = re.search(r'Статья\s+(\d+)', doc.page_content)
                            if article_match:
                                doc.metadata['article'] = article_match.group(1)
                        
                        documents.extend(file_docs)
                        loaded_files.append(file)
                    except Exception as e:
                        logger.error(f'Error loading file {file}: {e}')

        logger.info(f'Files loaded: {len(loaded_files)}')
        logger.info(f'Files: {loaded_files}')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\nСтатья", "\n\n", "\n", ". ", " "],
            keep_separator=True
        )
        source_chunks = text_splitter.split_documents(documents)
        
        logger.info(f'Chunks created: {len(source_chunks)}')

        db = FAISS.from_documents(source_chunks, embeddings)
        db.save_local(DB_PATH)

    return db


def get_bm25_index(db):
    """Create BM25 index for keyword search"""
    from rank_bm25 import BM25Okapi
    
    docs = db.docstore._dict.values()
    corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(corpus)
    return bm25, list(docs)
