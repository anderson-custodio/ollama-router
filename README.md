# LLM Router

Este projeto fornece uma API compatível com OpenAI para roteamento de solicitações para modelos Ollama locais.

## Pré-requisitos

- Python 3.8+
- [Ollama](https://ollama.ai/) instalado e rodando

## Instalação de Dependências

```bash
pip install -r requirements.txt
```

## Configurando Modelos Ollama

Certifique-se de ter os modelos mencionados no arquivo `ollama_router_openai_compatible_server.py` instalados no Ollama:

```bash
ollama pull llama3.1:8b-instruct-q8_0
ollama pull qwen2.5-coder:14b-base-q4_0
ollama pull gemma3:4b-it-q4_K_M
ollama pull deepseek-r1:8b
```

## Iniciando o Servidor

```bash
# Na pasta raiz do projeto
python ollama_router_openai_compatible_server.py
```

Isso iniciará o servidor na porta 8005.

## Utilizando com Open WebUI

O Open WebUI é uma interface web que você pode usar para interagir com o LLM Router. 
Para executá-lo usando Docker:

```bash
docker run -d -p 3000:8080 \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URLS="http://host.docker.internal:8005/v1;" \
  -e OPENAI_API_KEYS="dummy-key;" \
  -e WEBUI_AUTH=False \
  --restart always \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Após iniciar o contêiner, acesse a interface web em:
```
http://localhost:3000
```

## Funcionalidades Implementadas

- Roteamento dinâmico de solicitações para diferentes modelos Ollama baseado no tipo de pergunta
- Compatibilidade com a API de completions do OpenAI
- Suporte a streaming de resposta para uma experiência interativa
- Classificação automática do tipo de prompt (code, math, talk, reasoning, other)
