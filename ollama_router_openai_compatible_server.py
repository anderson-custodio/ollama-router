from fastapi import FastAPI, Request
import requests
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import uvicorn
import logging
import time
import uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import argparse

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
ROUTER_MODEL = "llama3.1:8b-instruct-q8_0"
TRANSLATION_MODEL = "mistral-small:22b-instruct-2409-q4_K_M"

# Automatic translation configuration
ENABLE_TRANSLATION = True  # Default value, can be changed via command line


# Redirect root endpoint to /v1/chat/completions
@app.post("/")
async def root(request: Request):
    body = await request.json()
    return await chat_completions(ChatCompletionRequest(**body))


# OpenAI compatible models
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completions"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage


# Task type mapping to specific models
MODEL_MAP = {
    "code": "qwen2.5-coder:14b-base-q4_0",
    "math": "gemma3:4b-it-q4_K_M",
    "talk": "llama3.1:8b-instruct-q8_0",
    "reasoning": "deepseek-r1:8b",
    "other": "llama3.1:8b-instruct-q8_0",
}


def detect_language(prompt: str) -> str:
    """
    Detects the language used in the user's prompt.
    Returns the language code (e.g., 'en', 'pt-br', 'es', etc.)
    """
    # Limit the prompt size to save tokens
    sample_text = prompt[:500] if len(prompt) > 500 else prompt

    lang_prompt = f"""
    You are a language detector. Analyze the following text and determine what language it is written in.
    Return ONLY the language code, without any explanation or additional text.
    Use codes like 'en' for English, 'pt-br' for Brazilian Portuguese, 'es' for Spanish, 'fr' for French, etc.
    
    Text to analyze: {sample_text}
    
    Response (language code only):
    """

    response = requests.post(
        OLLAMA_ENDPOINT,
        json={"model": ROUTER_MODEL, "prompt": lang_prompt, "stream": False},
    )

    detected_lang = response.json().get("response", "en").strip().lower()
    logger.info(f"Language detected: '{detected_lang}'")

    return detected_lang


def should_translate(user_language: str, content: str) -> bool:
    """
    Checks if the response content needs to be translated.
    Compares the language of the user's prompt with the language of the response.
    Returns True if translation is needed, False otherwise.
    """
    if not ENABLE_TRANSLATION or not user_language or user_language == "en":
        return False
        
    # Limit the content size to save tokens
    sample_text = content[:500] if len(content) > 500 else content
    
    # Detect the language of the response
    response_language = detect_language(sample_text)
    logger.info(f"Response language: '{response_language}'")
    
    # If the response language is the same as the user's, we don't need to translate
    if response_language == user_language:
        logger.info(f"Response is already in the user's language ({user_language}), skipping translation")
        return False
        
    return True


def translate_text(text: str, target_language: str) -> str:
    """
    Translates the text to the specified language using the router model.
    """
    if not text or target_language == "en":
        return text

    # Map language codes to full names to improve instruction
    language_names = {
        "pt": "Portuguese",
        "pt-br": "Brazilian Portuguese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
    }

    # Get the full name of the language, or use the code if not mapped
    language_name = language_names.get(target_language, target_language)

    translate_prompt = f"""
    You are a professional translator specialized in technical and educational content.
    
    TRANSLATE the following text COMPLETELY from English to {language_name}.
    
    IMPORTANT RULES:
    1. DO NOT translate any code blocks (content between ``` or `).
    2. DO NOT translate programming code, variable names, function names or classes.
    3. DO NOT translate comments within code blocks - keep ALL code and code comments in English.
    4. ALL other text MUST be translated to {language_name} - especially explanations, step-by-step reasoning, and conclusions.
    5. Preserve all formatting, markdown, symbols and special characters.
    6. Ensure the ENTIRE response is translated consistently to {language_name} - do not leave parts in English.
    7. Make sure the final answer and conclusion are in {language_name}.
    
    Text to translate:
    {text}
    
    Translation (in {language_name}):
    """

    # Use lower temperature for more consistent translation
    response = requests.post(
        OLLAMA_ENDPOINT,
        json={
            "model": TRANSLATION_MODEL,
            "prompt": translate_prompt,
            "stream": False,
            "temperature": 0.1,  # Lower temperature for greater consistency
        },
    )

    translated_text = response.json().get("response", text).strip()
    logger.info(f"Text translated to '{target_language}'")

    return translated_text


def classify_prompt(prompt: str) -> str:
    """
    Classifies the user's prompt into a category to choose the most suitable model.
    Categories: code, math, talk, reasoning, other
    """
    router_prompt = f"""
    You are a precise classifier that determines the most suitable category for a prompt.

    The categories are:
    - code: When the user asks to write, debug, explain, or review code
    - math: Mathematical questions, calculations, statistics, or formal logic
    - talk: Simple conversations, basic explanations, definitions, or direct questions that don't require deep analysis
    - reasoning: Complex tasks requiring multi-step reasoning, detailed analysis, advanced creativity, complex problem-solving, strategic planning, scientific themes, or when the prompt contains multiple interconnected questions
    - other: Anything that doesn't fit into the categories above

    IMPORTANT: If there is any doubt between "talk" and "reasoning", choose "reasoning".
    Use "reasoning" for:
    - Long prompts with multiple questions
    - Tasks requiring elaborate context analysis
    - Complex problems that need to be broken down into steps
    - When the user asks for a strategy, plan, or architecture
    - When the user asks questions requiring detailed explanations
    - When the user asks for a scientific explanation
    Prompt to classify: {prompt}
    Respond ONLY with one of these options (no explanations): code, math, talk, reasoning, other
    """

    response = requests.post(
        OLLAMA_ENDPOINT,
        json={"model": ROUTER_MODEL, "prompt": router_prompt, "stream": False},
    )
    classification = response.json().get("response", "other").strip().lower()
    logger.info(f"Prompt classified: '{prompt[:100]}...' (truncated)")
    logger.info(f"Classification received: '{classification}'")
    return classification


def select_model(task_type: str) -> str:
    """
    Selects the most appropriate model based on the classified task type.
    Returns the name of the Ollama model to use.
    """
    return MODEL_MAP.get(task_type, "llama3.1:8b-instruct-q8_0")


def extract_prompt_from_messages(messages: List[Message]) -> str:
    """
    Extracts the user's prompt from the message list.
    Returns the content of the last user message.
    """
    # Very simple extraction - get the last user message
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Extract the prompt from the messages
    user_prompt = extract_prompt_from_messages(request.messages)

    # Detect language synchronously
    user_language = ""
    if ENABLE_TRANSLATION:
        user_language = detect_language(user_prompt)
        logger.info(f"Language detected for translation: '{user_language}'")

    # Classify the task type
    task_type = classify_prompt(user_prompt)

    # Select the appropriate model
    selected_model = select_model(task_type)
    logger.info(f"Task type: {task_type}")
    logger.info(f"Selected model: {selected_model}")

    # Format messages for Ollama
    full_prompt = ""
    for msg in request.messages:
        prefix = f"{msg.role.capitalize()}: " if msg.role != "system" else "System: "
        full_prompt += f"{prefix}{msg.content}\n"

    # Handle streaming if requested
    if request.stream:
        return StreamingResponse(
            stream_response(selected_model, full_prompt, request.model, user_language),
            media_type="text/event-stream",
        )

    # Non-streaming response (original implementation)
    gen_response = requests.post(
        OLLAMA_ENDPOINT,
        json={"model": selected_model, "prompt": full_prompt, "stream": False},
    )

    response_text = gen_response.json().get("response", "")
    logger.info(f"Original response received, size: {len(response_text)}")

    # Translate the response if needed
    original_response = response_text
    # Check if translation is necessary (if the language of the response is different from the user's)
    if should_translate(user_language, response_text):
        logger.info(f"Starting translation to '{user_language}'")
        response_text = translate_text(response_text, user_language)
        logger.info(f"Translation completed, size: {len(response_text)}")
        # Safety check - if translation fails, use the original response
        if not response_text or len(response_text) < 10:
            logger.warning("Translation seems to have failed, using original response")
            response_text = original_response

    # Estimate token counts (very rough estimate)
    prompt_tokens = len(full_prompt.split())
    completion_tokens = len(response_text.split())
    total_tokens = prompt_tokens + completion_tokens

    # Build OpenAI-like response
    result = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completions",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    logger.info(f"Sending final response, size: {len(response_text)}")
    return result


async def stream_response(
    ollama_model: str, prompt: str, client_model: str, target_language: str = "en"
):
    """
    Stream the response from Ollama in the OpenAI streaming format.
    Streams the original response in real-time followed by the translated response, both via streaming.
    """
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    
    # Determine if we need translation - this will be confirmed after receiving the response
    need_translation = (
        ENABLE_TRANSLATION and target_language and target_language != "en"
    )
    if need_translation:
        logger.info(f"Streaming with translation potentially enabled for language '{target_language}'")
    else:
        logger.info(f"Streaming without translation (language: '{target_language}')")
    
    # We'll collect the complete response for later translation
    collected_content = ""
    
    # Phase 1: Send the original response in streaming directly from Ollama
    # while collecting the complete content for translation
    logger.info(f"Starting direct streaming of original response")
    
    # Make the request to Ollama with streaming
    with requests.post(
        OLLAMA_ENDPOINT,
        json={"model": ollama_model, "prompt": prompt, "stream": True},
        stream=True,
    ) as r:
        for line in r.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("response", "")
                    if content:
                        # Collect for translation later
                        collected_content += content
                        
                        # Send chunk immediately to client
                        data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": client_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        
                except Exception as e:
                    logger.error(f"Error processing streaming response: {e}")
    
    logger.info(f"Original response completed: {len(collected_content)} characters")
    
    # Phase 2: Check if the response needs to be translated
    perform_translation = collected_content and need_translation and should_translate(target_language, collected_content)
    
    # Translate and send the translated response in streaming if necessary
    if perform_translation:
        # Add a clear line break between responses
        separator = "\n\n---\n\n"
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': client_model, 'choices': [{'index': 0, 'delta': {'content': separator}, 'finish_reason': None}]})}\n\n"
        
        logger.info(f"Starting translation of complete content to '{target_language}'")
        translated_content = translate_text(collected_content, target_language)
        logger.info(f"Translation completed: {len(translated_content)} characters")
        
        # Safety check - if translation fails, skip this step
        if translated_content and len(translated_content) >= 10:
            # Send the translated content in chunks to simulate streaming
            chunk_size = 10  # Size of each chunk for streaming
            translated_chunks = [translated_content[i:i+chunk_size] for i in range(0, len(translated_content), chunk_size)]
            logger.info(f"Starting to send translated response ({len(translated_chunks)} chunks)")
            
            for chunk in translated_chunks:
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": client_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
        else:
            logger.warning("Translation failed, sending only original content")
    else:
        logger.info("Translation not necessary, response is already in the user's language")
    
    # Signal the end of streaming
    data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": client_model,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}
        ],
    }
    yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"
    logger.info("Streaming completed")


# Add the original endpoint for backward compatibility
@app.post("/chat/completions")
async def original_chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    return await chat_completions(request)


# Endpoint for /v1/completions (used by some OpenAI clients)
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # Convert completions request to chat completions
    chat_request = ChatCompletionRequest(
        model=request.model,
        messages=[Message(role="user", content=request.prompt)],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

    # If streaming, return directly the streaming response
    if request.stream:
        return await chat_completions(chat_request)

    # For non-streaming, continue with the original implementation
    response = await chat_completions(chat_request)

    # Convert response to completions format (not chat)
    return {
        "id": response["id"],
        "object": "text_completion",
        "created": response["created"],
        "model": response["model"],
        "choices": [
            {
                "text": choice["message"]["content"],
                "index": choice["index"],
                "finish_reason": choice["finish_reason"],
            }
            for choice in response["choices"]
        ],
        "usage": response["usage"],
    }


# Original chat endpoint for backward compatibility
class PromptRequest(BaseModel):
    prompt: str


@app.post("/chat")
async def chat_route(payload: PromptRequest):
    user_prompt = payload.prompt

    # Detect language synchronously
    user_language = ""
    if ENABLE_TRANSLATION:
        user_language = detect_language(user_prompt)
        logger.info(f"Language detected for /chat: '{user_language}'")

    # Step 1: classify
    task_type = classify_prompt(user_prompt)
    selected_model = select_model(task_type)
    logger.info(f"Task type: {task_type}")
    logger.info(f"Selected model: {selected_model}")

    # Step 2: generate with the appropriate model
    gen_response = requests.post(
        OLLAMA_ENDPOINT,
        json={"model": selected_model, "prompt": user_prompt, "stream": False},
    )

    response_text = gen_response.json().get("response", "")
    logger.info(f"Response received in /chat, size: {len(response_text)}")

    # Translate the response if necessary
    original_response = response_text
    # Check if translation is necessary (if the language of the response is different from the user's)
    if should_translate(user_language, response_text):
        logger.info(f"Starting translation in /chat to '{user_language}'")
        response_text = translate_text(response_text, user_language)
        logger.info(f"Translation in /chat completed, size: {len(response_text)}")

        # Safety check - if translation fails, use the original response
        if not response_text or len(response_text) < 10:
            logger.warning("Translation in /chat failed, using original response")
            response_text = original_response

    result = {
        "model": selected_model,
        "task_type": task_type,
        "response": response_text,
        "original_language": "en",
        "translated_to": user_language if user_language != "en" else None,
    }

    logger.info(f"Sending final response from /chat, size: {len(response_text)}")
    return result


# Health check endpoint
@app.get("/v1/health")
async def health_check():
    return {"status": "ok"}


# Models list endpoint
@app.get("/v1/models")
async def list_models():
    # Fixed list of OpenAI-compatible models for client display
    available_models = ["dynamic"]
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "public",
            }
            for model_name in available_models
        ],
    }


# Command line configuration
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ollama Router compatible with OpenAI API"
    )
    parser.add_argument(
        "--disable-translation",
        action="store_true",
        help="Disable automatic translation to user's language",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8005,
        help="Port to run the server on (default: 8005)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on (default: 0.0.0.0)",
    )
    return parser.parse_args()


# Run locally
if __name__ == "__main__":
    args = parse_args()

    # Configure translation based on arguments
    ENABLE_TRANSLATION = not args.disable_translation

    logger.info(
        f"Server starting with automatic translation {'ENABLED' if ENABLE_TRANSLATION else 'DISABLED'}"
    )
    uvicorn.run(app, host=args.host, port=args.port)
