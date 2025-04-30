# LLM Router

This project provides an OpenAI-compatible API for routing requests to local Ollama models.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## Setting Up Ollama Models

Make sure you have the models mentioned in the `ollama_router_openai_compatible_server.py` file installed in Ollama:

```bash
ollama pull llama3.1:8b-instruct-q8_0
ollama pull qwen2.5-coder:14b-base-q4_0
ollama pull gemma3:4b-it-q4_K_M
ollama pull deepseek-r1:8b
```

## Starting the Server

```bash
# In the project's root folder
python ollama_router_openai_compatible_server.py
```

This will start the server on port 8005.

## Using with Open WebUI

Open WebUI is a web interface that you can use to interact with the LLM Router.
To run it using Docker:

```bash
docker run -d -p 3000:8080 \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URLS="http://host.docker.internal:8005/v1;" \
  -e OPENAI_API_KEYS="keep-anything-here;" \
  -e WEBUI_AUTH=False \
  --restart always \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

After starting the container, access the web interface at:
```
http://localhost:3000
```

## Implemented Features

- Dynamic routing of requests to different Ollama models based on question type
- Compatibility with OpenAI's completions API
- Support for response streaming for an interactive experience
- Automatic classification of prompt type (code, math, talk, reasoning, other)
