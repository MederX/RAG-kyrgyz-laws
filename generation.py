"""LLM response generation"""
from loguru import logger
import google.generativeai as genai
from collections import Counter
import re
import os
from dotenv import load_dotenv
from config import *

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv('GEMINI_API_KEY') or GEMINI_API_KEY
if not api_key:
    logger.warning("GEMINI_API_KEY not found in .env file or config.py. Please set it.")
else:
    genai.configure(api_key=api_key)

# Lazy load LLM for reuse
_llm_cache = None

def get_llm(temperature=None):
    """Get or create LLM instance"""
    global _llm_cache
    if temperature is None:
        temperature = STREAMING_TEMPERATURE
    
    # Reuse cached LLM if same temperature
    if _llm_cache and _llm_cache[0] == temperature:
        return _llm_cache[1]
    
    # Safety settings to allow legal content
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        generation_config={
            'temperature': temperature,
            'max_output_tokens': 2048,
            'top_p': 0.95,
            'top_k': 40,
        },
        safety_settings=safety_settings
    )
    _llm_cache = (temperature, model)
    return model


def _generate_with_retry(model, prompt, max_retries=3, base_delay=1):
    """Generate content with exponential backoff retry logic"""
    import time
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response.text:  # Check if response has content
                return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Generation failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
    
    raise Exception("Failed to generate response after all retries")


def detect_language(text):
    """Detect language of the question"""
    # Count Cyrillic vs Latin characters
    cyrillic_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # Detect Kyrgyz-specific patterns
    kyrgyz_patterns = ['ң', 'ү', 'ө', 'Ң', 'Ү', 'Ө']
    has_kyrgyz = any(char in text for char in kyrgyz_patterns)
    
    if has_kyrgyz or (cyrillic_chars > latin_chars and cyrillic_chars > 5):
        # If Kyrgyz-specific characters or mostly Cyrillic
        return "Kyrgyz" if has_kyrgyz else "Russian"
    elif latin_chars > cyrillic_chars and latin_chars > 5:
        return "English"
    else:
        # Default to English for short or mixed text
        return "English"


def post_process_answer(answer):
    """Post-process and clean up the generated answer"""
    if not answer:
        return ""
    
    # Strip whitespace
    answer = answer.strip()
    
    # Normalize whitespace (replace multiple spaces with single space)
    answer = re.sub(r'\s+', ' ', answer)
    
    # Remove any markdown formatting artifacts if present
    answer = re.sub(r'^#+\s*', '', answer)  # Remove markdown headers
    answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)  # Remove bold markers
    
    # Ensure proper sentence spacing
    answer = re.sub(r'\.([А-ЯA-Z])', r'. \1', answer)  # Add space after period before capital
    
    # Remove redundant phrases
    redundant_phrases = [
        r'Based on the provided context[,:]?\s*',
        r'According to the context[,:]?\s*',
        r'From the legal texts?[,:]?\s*',
    ]
    for phrase in redundant_phrases:
        answer = re.sub(phrase, '', answer, flags=re.IGNORECASE)
    
    return answer.strip()


RAG_PROMPT = """You are a helpful legal assistant for Kyrgyz Republic laws. Your goal is to explain the law in simple, easy-to-understand language for a general audience, while keeping the legal citations accurate.

⚠️ CRITICAL LANGUAGE REQUIREMENT - READ CAREFULLY:
The question is asked in: {language}
YOU MUST answer in {language} ONLY. Do NOT switch to Russian or any other language.
- English question → Write your ENTIRE answer in English
- Russian question → Write your ENTIRE answer in Russian
- Kyrgyz question → Write your ENTIRE answer in Kyrgyz

The legal context below is in Russian, but you MUST translate and explain it in {language}.

CONTEXT FROM KR LAWS:
{context}

QUESTION: {question}

RECENT CONVERSATION:
{history}

INSTRUCTIONS:
1. **Base your answer ONLY on the provided context But do not specify it like "based on the provided context"**.
2. **Use VERY SIMPLE language - imagine explaining to someone with no legal knowledge**:
   - Replace ALL legal jargon with everyday words
   - Use short, clear sentences
   - Explain consequences in plain terms (e.g., "you could go to jail for 3 years" instead of "imprisonment for a term of three years")
   - Avoid phrases like "is classified as", "is punishable by", "pursuant to" - use simple verbs instead
   - Think: How would you explain this to a teenager or a friend?
3. **Structure**:
   - Start with the most important point in ONE simple sentence
   - Then give 2-3 key details in plain language
   - Mention the article/law name naturally (e.g., "The Criminal Code says..." or "According to Article 123...")
4. **Length**: 4-6 sentences maximum. Be concise.
5. **REMEMBER**: Write your answer in {language}, NOT in the language of the context.

EXAMPLE FORMAT (if question is in English):
If you steal something, you will face serious punishment. The Criminal Code says you could be fined or sent to prison. How long depends on what you stole and how you did it. The punishment can range from a fine to several years in jail.

ANSWER IN {language}:

ANSWER IN {language}:"""


def get_model_response(topic, message_content, history=""):
    """Generate response with optional self-consistency and retry logic"""
    logger.debug('...get_model_response')
    
    base_temp = TEMPERATURES[0]
    
    # Detect question language
    language = detect_language(topic)
    logger.info(f"Detected question language: {language}")
    
    if USE_SELF_CONSISTENCY:
        # Quality mode: multiple temperatures
        answers = []
        for temp in TEMPERATURES:
            model = get_llm(temp)
            prompt = RAG_PROMPT.format(context=message_content, question=topic, history=history, language=language)
            
            try:
                response = _generate_with_retry(model, prompt)
                answer = post_process_answer(response.text)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error at temp {temp}: {e}")
        
        if len(answers) > 0:
            answer_counts = Counter(answers)
            if answer_counts.most_common(1)[0][1] > 1:
                return answer_counts.most_common(1)[0][0]
            return answers[0]
    else:
        # Speed mode: single temperature with validation
        model = get_llm(base_temp)
        
        # Detect question language
        language = detect_language(topic)
        logger.info(f"Detected question language: {language}")
        
        prompt = RAG_PROMPT.format(context=message_content, question=topic, history=history, language=language)
        
        try:
            response = _generate_with_retry(model, prompt)
            answer = post_process_answer(response.text)
            
            # Validate answer
            is_valid, reason = validate_answer(answer, topic, message_content)
            if not is_valid:
                logger.warning(f"Answer validation failed: {reason}, retrying with adjusted prompt")
                # Retry with enhanced prompt
                enhanced_prompt = prompt + f"\n\nNote: Please provide a detailed response in {language} with specific article references from the context."
                response = _generate_with_retry(model, enhanced_prompt)
                answer = post_process_answer(response.text)
            
            return answer
        except Exception as e:
            logger.error(f"Error generating response: {e}")
    
    return "Sorry, an error occurred while processing your request."


def get_model_response_stream(topic, message_content, history=""):
    """Generate streaming response for Gradio with improved error handling"""
    logger.debug('...get_model_response_stream')
    
    temp = STREAMING_TEMPERATURE
    model = get_llm(temp)
    
    # Detect question language
    language = detect_language(topic)
    logger.info(f"Detected question language: {language}")
    
    prompt = RAG_PROMPT.format(context=message_content, question=topic, history=history, language=language)
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, stream=True)
            
            accumulated_text = ""
            has_content = False
            
            for chunk in response:
                if chunk.text:
                    has_content = True
                    accumulated_text += chunk.text
                    yield chunk.text
            
            if has_content:
                # Validate accumulated response
                is_valid, reason = validate_answer(accumulated_text, topic, message_content)
                if not is_valid:
                    logger.warning(f"Streamed answer validation failed: {reason}")
                return  # Success, exit retry loop
            else:
                if attempt < max_retries - 1:
                    logger.warning("No content generated, retrying...")
                    continue
                else:
                    yield "I couldn't generate a response. Please try rephrasing your question."
                    
        except Exception as e:
            logger.error(f"Error getting model response (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
            else:
                yield "Sorry, an error occurred while processing your request. Please try again."


def validate_answer(answer, question, context):
    """Validate answer quality with comprehensive checks"""
    # Check minimum length
    if len(answer.strip()) < 20:
        return False, "Answer too short"
    
    # Check maximum reasonable length (not a dump of context)
    if len(answer) > 1500:
        return False, "Answer too long (possible context dump)"
    
    # Check if answer is just an error message
    if "sorry" in answer.lower() and "error" in answer.lower():
        return False, "Error response"
    
    # Check for generic/evasive responses
    evasive_patterns = [
        r'i (do not|don\'t) have (enough )?information',
        r'the context (does not|doesn\'t) provide',
        r'без дополнительной информации',
        r'я не могу ответить',
    ]
    for pattern in evasive_patterns:
        if re.search(pattern, answer.lower()):
            return False, "Evasive or insufficient answer"
    
    # Check if answer seems relevant (contains key words from question)
    question_words = set(re.findall(r'\w{3,}', question.lower()))  # Words with 3+ chars
    answer_words = set(re.findall(r'\w{3,}', answer.lower()))
    common_words = question_words & answer_words
    
    # Remove common stop words (expanded list)
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'can', 'could', 'should', 'would',
        'i', 'you', 'in', 'on', 'at', 'for', 'with', 'from', 'about', 'this', 'that', 'these', 'those',
        'и', 'в', 'на', 'что', 'как', 'или', 'для', 'это', 'был', 'была', 'были', 'быть', 'можно',
        'который', 'которая', 'которое', 'может', 'также', 'если', 'при', 'где', 'когда'
    }
    relevant_common = common_words - stop_words
    
    if len(relevant_common) < 1:
        return False, "Answer doesn't seem related to question"
    
    # Check if answer references the context (mentions articles or laws)
    has_reference = bool(re.search(r'(article|статья|статьи|law|закон|кодекс|codex)\s*(№|#|\d+)', answer.lower()))
    
    # For legal questions, we expect references when context is substantial
    if not has_reference and len(context) > 200:
        # Check if the answer at least mentions legal concepts
        legal_terms = r'(права|обязанност|ответственност|наказан|штраф|санкц|right|duty|obligation|penalty|fine|liable)'
        if not re.search(legal_terms, answer.lower()):
            return False, "Answer lacks legal references or terminology"
    
    # Check sentence structure (should have complete sentences)
    sentences = re.split(r'[.!?]+', answer)
    meaningful_sentences = [s for s in sentences if len(s.split()) > 3]
    if len(meaningful_sentences) < 1:
        return False, "Answer lacks proper sentence structure"
    
    return True, "OK"
