# RAG (Retrieval-augmented generation) ChatBot

[![CI](https://github.com/madhavmeesala/rag-chatbot/workflows/CI/badge.svg)](https://github.com/madhavmeesala/rag-chatbot/actions/workflows/ci.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Check out the todo list to see the next steps and improvements planned for this project [here](notes/todo.md).

> [IMPORTANT]
> Disclaimer:
> The code has been tested on:
>   * Ubuntu 22.04.2 LTS running on a Lenovo Legion 5 Pro with twenty 12th Gen Intel Core i7-12700H and
      an NVIDIA GeForce RTX 3060.
>   * MacOS Sonoma 14.3.1 running on a MacBook Pro M1 (2020).
>
> If you are using another Operating System or different hardware, and you can't load the models, please
> take a look at the official llama.cpp's GitHub issue tracker.

> [WARNING]
> It's important to note that the large language model sometimes generates hallucinations or false information.

## Table of contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
    - [Install Poetry](#install-poetry)
- [Bootstrap Environment](#bootstrap-environment)
    - [How to use the make file](#how-to-use-the-make-file)
    - [Environment](#environment)
    - [Set the Open-Source LLM Model](#set-the-open-source-llm-model)
    - [Set the Embedding Model](#set-the-embedding-model)
    - [Set the Response Synthesis strategy](#set-the-response-synthesis-strategy)
- [Build the memory index](#build-the-memory-index)
- [Run the Chatbot](#run-the-chatbot)
- [References](#references)

## Introduction

This project combines the power of llama.cpp and Chroma to build:

* a Conversation-aware Chatbot (ChatGPT like experience).
* a RAG (Retrieval-augmented generation) ChatBot.

The RAG Chatbot works by taking a collection of Markdown files as input and, when asked a question, provides the
corresponding answer based on the context provided by those files.

![rag-chatbot-architecture-1.png](images/rag-chatbot-architecture-1.png)

> [NOTE]
> I decided to refactor the "RecursiveCharacterTextSplitter" class from LangChain to effectively chunk
> Markdown files without adding LangChain as a dependency.

The "Memory Builder" component of the project loads Markdown pages from the "docs" folder.
It then divides these pages into smaller sections, calculates the embeddings (a numerical representation) of these
sections with the Semantic Search models from Sentence Transformers, and saves them in an embedding database called Chroma for later use.

When a user asks a question, the RAG ChatBot retrieves the most relevant sections from the Embedding database.
Since the original question can't be always optimal to retrieve for the LLM, we first prompt an LLM to rewrite the
question, then conduct retrieval-augmented reading.
The most relevant sections are then used as context to generate the final answer using a local language model (LLM).
Additionally, the chatbot is designed to remember previous interactions. It saves the chat history and considers the
relevant context from previous conversations to provide more accurate answers.

To deal with context overflows, I implemented two approaches:

* "Create And Refine the Context": synthesize a responses sequentially through all retrieved contents.
    * ![create-and-refine-the-context.png](images/create-and-refine-the-context.png)
* "Hierarchical Summarization of Context": generate an answer for each relevant section independently, and then
  hierarchically combine the answers.
    * ![hierarchical-summarization.png](images/hierarchical-summarization.png)

The "Memory Builder" builds the vector database in an incremental way, which means that when a document changes,
we only update the corresponding chunks in the vector store instead of rebuilding the whole index.

This is achieved through:
- Document-level metadata tracking: every chunk gets tagged with a source doc ID + version hash. When a doc changes, we regenerate chunks for that doc only, delete the old ones by metadata filter, and insert new ones. This is much more efficient than rebuilding the whole index.
- Incremental ingestion pipeline: the pipeline diffs source docs against what's already indexed (using those version hashes). Only changed/new docs get processed. Keeps compute costs reasonable as the corpus grows.
- Handling deletions: I keep a separate mapping table (doc_id -> chunk_ids) in a SQLite db so I can precisely target what to remove without scanning the whole store.

> [IMPORTANT]
> One thing to watch out for - if you ever swap embedding models, you must rebuild it from scratch since the vector spaces won't be compatible. Plan for that early.

## Prerequisites

* Python 3.12+
* GPU supporting CUDA 12.4+ or Apple Silicon M-series
* Poetry 2.3.0+
  * See [notes/poetry.md](notes/technical/poetry.md#install-poetry).
* Docker 24.0.6+ and Docker Compose 5.0.2+
* NVIDIA Container Toolkit installed (optional, for CUDA support)
  * See [notes/llama-server-docker.md](notes/technical/llamacpp/server-docker.md#installing-nvidia-container-toolkit).

For the UI:
* Node 22.12+
* Yarn 1.22+

## Bootstrap Environment

To easily install the dependencies and start the services, I have provided a make file.

### How to use the make file

> [IMPORTANT]
> Run "Setup" as your init command (or after "Clean").

* Check: "make check"
    * Use it to check that "which pip3" and "which python3" points to the right path.
* Setup:
    * Setup with NVIDIA CUDA acceleration: "make setup_cuda"
        * Creates an environment and installs all dependencies with NVIDIA CUDA acceleration.
    * Setup with Metal GPU acceleration: "make setup_metal"
        * Creates an environment and installs all dependencies with Metal GPU acceleration for macOS system only.
    * Both starts "llama.cpp" server locally via Docker compose.
* Start: "make start"
    * Start both the backend and frontend ensuring that the backend is running and ready before launching the frontend.
* Start llama.cpp server
    * on CUDA: "make start_llama_server_cuda"
    * on Metal: "make start_llama_server_metal"
    * Start the llama.cpp server locally via Docker compose.
    * It will be available at http://0.0.0.0:8080 (it will show the llama-ui).
* Stop "llama.cpp" Server: "make stop_llama_server"
    * Stop the llama.cpp server if it's running locally.
* Update: "make update"
    * Update an environment and installs all updated dependencies.
* Tidy up the code: "make tidy"
    * Run Ruff check and format.
* Clean: "make clean"
    * Removes the environment and all cached files.
* Test: "make test"
    * Runs all tests using pytest.

### Environment

Copy .env.example -> .env and fill it in.

Copy /frontend/.env.example -> .env and fill it in.

To install the UI dependencies, run:

```shell
cd frontend
nvm use
npm install -g yarn
yarn

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env
```

### Set the Open-Source LLM Model

"llama-cpp" serves as a C++ backend designed to work efficiently with transformer-based models, which runs either on a CPU or GPU.
It uses quantized models which are stored in GGML/GGUF format.

You can load any GGUF model from HuggingFace.

In the .env you need to set the "MODEL" variable with the name of the model you want to load, and the "MODEL_URL" variable with the URL of the model in GGUF format:
```
MODEL="Meta-Llama-3.1-8B-Instruct-Q4_K_M"
MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
```

> [IMPORTANT]
> The Chatbot must be restarted after changing the model.

The chosen model will be downloaded in the "/models" folder and loaded in the "llama.cpp" server.

> [NOTE]
> You can load models that fit your hardware capacity and speed requirements.
> To decide which hardware to use to host local LLMs, I recommend reading these benchmarks:
> - Performance of llama.cpp on Nvidia CUDA
> - Performance of llama.cpp on Apple Silicon M-series
>
> Decision model:
> - Memory capacity is the main limit. Check if the model fits in memory (with quantization).
> - Memory bandwidth mostly determines speed (tokens/sec). Check if the bandwidth gives an acceptable speed.
> - If not, upgrade hardware or optimize the model.

I recommend starting with "Qwen 3.5 9B" or "Meta Llama 3.2 Instruct 3B" since they are small enough to run on a standard GPU with 6GB of VRAM.

Recommended models to start:

| Model | Model Size | Max Context Window | Notes |
|-------|------------|--------------------|-------|
| Qwen 3.6 27B | 27B | 262k | Recommended model |
| Qwen 3.6 35B A3B | 35B (3B activated) | 262k | MoE architecture |
| Qwen 3.5 0.8B | 0.8B | 256k | Tiny and fast multimodal |
| Qwen 3.5 2B | 2B | 256k | Multimodal for lightweight agents |
| Qwen 3.5 4B | 4B | 256k | Balanced performance |
| Qwen 3.5 9B | 9B | 256k | Recommended model for complex tasks |
| Meta Llama 3.2 Instruct | 1B | 128k | Optimized for mobile/edge |
| Meta Llama 3.2 Instruct | 3B | 128k | Optimized for mobile/edge |
| Meta Llama 3.1 Instruct | 8B | 128k | Stable and reliable |
| DeepSeek R1 Distill Qwen 7B | 7B | 128k | Experimental reasoning model |

### Set the Embedding Model

For semantic search, I support all embedding models from Sentence Transformers.

In the .env you need to set the "EMBEDDING_MODEL" variable with the name of the model you want to load:
```
EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

To find the list of best embeddings models, consult the Massive Text Embedding Benchmark (MTEB) Leaderboard.
I recommend using the jina-embeddings-v5-text models for their performance and size.

| Embedding Model | Supported | Model Size | Max Tokens | Retrieval score (MTEB) |
|-----------------|-----------|------------|------------|------------------------|
| all-MiniLM-L6-v2 | Yes | 0.023B | 512 | 33.30 |
| all-MiniLM-L12-v2 | Yes | 0.033B | 256 | 33.37 |
| all-mpnet-base-v2 | Yes | 0.109B | 384 | 33.80 |
| jinaai/jina-embeddings-v5-text-small-retrieval | Yes | 0.596B | 32k | 64.88 |
| jinaai/jina-embeddings-v5-text-nano-retrieval | Yes | 0.212B | 8k | 63.26 |

### Set the Response Synthesis strategy

In the .env you need to set the "SYNTHESIS_STRATEGY" variable:
```
SYNTHESIS_STRATEGY="tree-summarization"
```

| Response Synthesis strategy | Supported | Notes |
|-----------------------------|-----------|-------|
| create-and-refine | Yes | Sequential synthesis |
| tree-summarization | Yes | Recommended - Hierarchical synthesis |


## Build the memory index

You can download Markdown pages (e.g., from an employee handbook) and place them under "docs".

Build the memory index by running:

```shell
make migrate_db
cd scripts && PYTHONPATH=.:../backend python memory_builder.py --model-name jinaai/jina-embeddings-v5-text-small-retrieval --chunk-size 1000 --chunk-overlap 50
```

## Run the Chatbot

The Chatbot has a UI built with Vite, React and TypeScript, and a backend built with FastAPI that serves the LLMs through llama.cpp server.

To start both the backend and frontend:

```shell
make start
```

The application will be available at http://localhost:5173, with the backend API at http://localhost:8000.

![conversation-aware-chatbot.gif](images/conversation-aware-chatbot.gif)

You can enable the RAG Mode feature in the UI to ask questions based on the context provided by the Markdown files you loaded and indexed.

![rag_chatbot_example.gif](images/rag_chatbot_example.gif)

You can also upload a Markdown file using the file uploader. Once uploaded, files are chunked, embedded, and upserted to Chroma automatically.

![rag_chatbot_load_doc_example.gif](images/rag_chatbot_load_doc_example.gif)

## References

* Large Language Models (LLMs):
    * LLMs as a repository of vector programs
    * GPT in 60 Lines of NumPy
    * Calculating GPU memory for serving LLMs
    * Introduction to Weight Quantization
    * Uncensor any LLM with abliteration
    * Understanding Multimodal LLMs
    * Direct preference optimization (DPO): Complete overview
* LLM Frameworks:
    * Deepval - A framework for evaluating LLMs
    * Structured Outputs (Outlines)
* LLM Datasets:
    * High-quality datasets
* Agents:
    * Agents (Huyen Chip)
    * Building effective agents (Anthropic)
* Agent Frameworks:
    * PydanticAI
    * Atomic Agents
    * agno - lightweight library for building Agents.
* Vector Databases:
    * Indexing algorithms (HNSW, IVF)
    * Chroma
    * Qdrant
* Retrieval Augmented Generation (RAG):
    * Building A Generative AI Platform
    * Rewrite-Retrieve-Read
    * Rerank
    * Building Response Synthesis from Scratch
    * Conversational awareness
* Text Processing and Cleaning:
    * clean-text
    * Fast Semantic Text Deduplication
* Inspirational Open Source Repositories:
    * lit-gpt
    * api-for-open-llm
    * AnythingLLM
    * Alpaca
    * LiteLLM

---

## Maintainer
**Madhav Meesala**
Software Engineer
Email: madhavmeesala@gmail.com

## About the Developer
Madhav Meesala is a Software Engineer with over 4 years of experience building scalable full-stack and distributed systems. He has a proven track record in financial services and digital payments, utilizing technologies like Java, C++, Python, and AWS. His recent focus includes the implementation of GenAI, LLMs, and RAG architectures to build high-performance, intelligent applications. This project represents his ongoing work in optimizing local LLM deployment and efficient document retrieval systems.