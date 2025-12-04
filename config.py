"""Configuration settings for RAG system"""

# Model settings
GEMINI_MODEL_NAME = "gemini-flash-latest"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GEMINI_API_KEY = None

# Retrieval settings
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
RETRIEVAL_K = 8  # Reduced from 10 for speed
RERANK_TOP_N = 15  # Reduced from 20 for speed

# Generation settings
TEMPERATURES = [0.1, 0.2, 0.15]  # Multiple temps for self-consistency mode
STREAMING_TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 2048
TOP_P = 0.95
TOP_K = 40

# Response quality settings
MIN_ANSWER_LENGTH = 20
MAX_ANSWER_LENGTH = 1500
REQUIRE_LEGAL_REFERENCES = True  # Require article/law citations
MAX_GENERATION_RETRIES = 3

# Speed optimizations
USE_SELF_CONSISTENCY = False  # Disable for speed (enable for max quality)
USE_RERANKING = True  # Keep for quality
USE_BM25 = True  # Keep for quality
LAZY_LOAD_RERANKER = True  # Load once, reuse

# Paths
LAWS_DIR = "laws"
DB_PATH = "db/laws_db"
LOG_PATH = "log/kyrgyz_laws_rag.log"

# Cache settings
MAX_CACHE_SIZE = 100

# Server settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860
