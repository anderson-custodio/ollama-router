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

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Adicionar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos
    allow_headers=["*"],  # Permite todos os headers
)

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
ROUTER_MODEL = "llama3.1:8b-instruct-q8_0"


# Redirecionar endpoint raiz para /v1/chat/completions
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


# Mapeamento de tipos de tarefas para modelos específicos
MODEL_MAP = {
    "code": "qwen2.5-coder:14b-base-q4_0",
    "math": "gemma3:4b-it-q4_K_M",
    "talk": "llama3.1:8b-instruct-q8_0",
    "reasoning": "deepseek-r1:8b",
    "other": "llama3.1:8b-instruct-q8_0",
}


def classify_prompt(prompt: str) -> str:
    """
    Classifica o prompt do usuário em uma categoria para escolher o modelo mais adequado.
    Categorias: code, math, talk, reasoning, other
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
    logger.info(f"Prompt classificado: '{prompt[:100]}...' (truncado)")
    logger.info(f"Classificação recebida: '{classification}'")
    return classification


def select_model(task_type: str) -> str:
    """
    Seleciona o modelo mais adequado com base no tipo de tarefa classificada.
    Retorna o nome do modelo Ollama a ser usado.
    """
    return MODEL_MAP.get(task_type, "llama3.1:8b-instruct-q8_0")


def extract_prompt_from_messages(messages: List[Message]) -> str:
    """
    Extrai o prompt do usuário da lista de mensagens.
    Retorna o conteúdo da última mensagem do usuário.
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
            stream_response(selected_model, full_prompt, request.model),
            media_type="text/event-stream",
        )

    # Non-streaming response (original implementation)
    gen_response = requests.post(
        OLLAMA_ENDPOINT,
        json={"model": selected_model, "prompt": full_prompt, "stream": False},
    )

    response_text = gen_response.json().get("response", "")

    # Estimate token counts (very rough estimate)
    prompt_tokens = len(full_prompt.split())
    completion_tokens = len(response_text.split())
    total_tokens = prompt_tokens + completion_tokens

    # Build OpenAI-like response
    return {
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


async def stream_response(ollama_model: str, prompt: str, client_model: str):
    """
    Stream the response from Ollama in the OpenAI streaming format.
    """
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Make streaming request to Ollama
    with requests.post(
        OLLAMA_ENDPOINT,
        json={"model": ollama_model, "prompt": prompt, "stream": True},
        stream=True,
    ) as r:
        collected_content = ""
        for line in r.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("response", "")

                    if content:
                        collected_content += content

                        # Format in OpenAI streaming format
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

                    # Check for done
                    if chunk.get("done", False):
                        # Send the final chunk with finish_reason
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

                except Exception as e:
                    logger.error(f"Error processing streaming response: {e}")
                    continue


# Add the original endpoint for backward compatibility
@app.post("/chat/completions")
async def original_chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    return await chat_completions(request)


# Endpoint para /v1/completions (usado por alguns clientes OpenAI)
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # Converter request de completions para chat completions
    chat_request = ChatCompletionRequest(
        model=request.model,
        messages=[Message(role="user", content=request.prompt)],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

    # Se for streaming, retornar diretamente a resposta de streaming
    if request.stream:
        return await chat_completions(chat_request)

    # Para não-streaming, continuar com a implementação original
    response = await chat_completions(chat_request)

    # Converter resposta para o formato de completions (não chat)
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

    # Etapa 1: classificar
    task_type = classify_prompt(user_prompt)
    selected_model = select_model(task_type)
    logger.info(f"Task type: {task_type}")
    logger.info(f"Selected model: {selected_model}")

    # Etapa 2: gerar com o modelo adequado
    gen_response = requests.post(
        OLLAMA_ENDPOINT,
        json={"model": selected_model, "prompt": user_prompt, "stream": False},
    )

    return {
        "model": selected_model,
        "task_type": task_type,
        "response": gen_response.json().get("response", ""),
    }


# Health check endpoint
@app.get("/v1/health")
async def health_check():
    return {"status": "ok"}


# Models list endpoint
@app.get("/v1/models")
async def list_models():
    # Lista fixa de modelos compatíveis com OpenAI para exibição ao cliente
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


# Rodar localmente
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
