"""Retrieval and search functionality"""
from loguru import logger
import re
from config import *

# Cache for query results
query_cache = {}

# Lazy load reranker
_reranker_cache = None

def get_reranker():
    """Get or create reranker instance"""
    global _reranker_cache
    if _reranker_cache is None and USE_RERANKING:
        from sentence_transformers import CrossEncoder
        _reranker_cache = CrossEncoder(RERANKER_MODEL)
    return _reranker_cache


def expand_query(query):
    """Expand query with synonyms and variations"""
    expansions = [query]
    
    # Add question variations
    if "что" in query.lower() or "what" in query.lower():
        expansions.append(query.replace("что", "какие").replace("what", "which"))
    
    # Add legal term variations
    legal_terms = {
        "права": ["право", "правомочия"],
        "обязанности": ["обязанность", "долг"],
        "ответственность": ["наказание", "санкция"],
        "rights": ["right", "entitlement"],
        "duties": ["duty", "obligation"]
    }
    
    for term, synonyms in legal_terms.items():
        if term in query.lower():
            for syn in synonyms:
                expansions.append(query.lower().replace(term, syn))
    
    return expansions[:3]


def get_message_content(topic, db, k):
    """Retrieve relevant context using hybrid search"""
    logger.debug('...get_message_content')
    
    # Check cache
    cache_key = f"{topic}_{k}"
    if cache_key in query_cache:
        logger.debug("Using cached results")
        return query_cache[cache_key], True  # Return with cache flag
    
    # Query expansion
    queries = expand_query(topic)
    all_docs = []
    
    # Hybrid search: Vector (70%) + BM25 (30%)
    for query in queries:
        try:
            vector_docs = db.max_marginal_relevance_search(query, k=k, fetch_k=k*2)
        except Exception:
            vector_docs = db.similarity_search(query, k=k)
        
        all_docs.extend(vector_docs)
    
    # BM25 keyword search
    if USE_BM25:
        try:
            from rank_bm25 import BM25Okapi
            from database import get_bm25_index
            bm25, corpus_docs = get_bm25_index(db)
            tokenized_query = topic.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k//3]
            bm25_docs = [corpus_docs[i] for i in top_bm25_indices]
            all_docs.extend(bm25_docs)
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
    
    # Remove duplicates
    seen = set()
    unique_docs = []
    for doc in all_docs:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
    
    # Rerank with cross-encoder
    if USE_RERANKING and len(unique_docs) > k:
        try:
            reranker = get_reranker()
            if reranker:
                pairs = [[topic, doc.page_content] for doc in unique_docs[:RERANK_TOP_N]]
                scores = reranker.predict(pairs)
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                docs = [unique_docs[i] for i in ranked_indices[:k]]
                logger.debug(f"Reranked {len(unique_docs)} docs to top {k}")
            else:
                docs = unique_docs[:k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original order")
            docs = unique_docs[:k]
    else:
        docs = unique_docs[:k]
    
    # Build context with metadata
    sources_content = {}
    for doc in docs:
        source = doc.metadata.get('source_file', 'unknown')
        law_name = doc.metadata.get('law_name', source)
        article = doc.metadata.get('article', '')
        
        if source not in sources_content:
            sources_content[source] = {
                'law_name': law_name,
                'chunks': []
            }
        
        chunk_text = doc.page_content
        if article:
            chunk_text = f"[Статья {article}] {chunk_text}"
        
        sources_content[source]['chunks'].append(chunk_text)
    
    # Compress context
    message_content = ""
    for source, data in sources_content.items():
        message_content += f"\n=== {data['law_name']} ===\n"
        for chunk in data['chunks'][:3]:
            clean_chunk = re.sub(r'\n+', ' ', chunk.strip())
            clean_chunk = re.sub(r'\s+', ' ', clean_chunk)
            message_content += f"{clean_chunk}\n"
        message_content += "\n"
    
    result = message_content.strip()
    
    # Cache result
    if len(query_cache) < MAX_CACHE_SIZE:
        query_cache[cache_key] = result
    
    logger.debug(f"Relevant sources: {len(sources_content)}, Total chunks: {len(docs)}")
    return result, False  # Return with cache flag
