# Vir ChatBot

A RAG (Retrieval-Augmented Generation) chatbot specialized in **virology** and **bioinformatics**, designed to run locally. Use your own scientific documents (PDFs) to create a personalized knowledge base and interact with your data through an intelligent assistant.

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green?logo=chainlink)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)

---

## 📦 Prerequisites

- **Docker** and **Docker Compose** (recommended)
- **Python 3.12+** (for local development)
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager (for local development)
- An API key from **one** of the supported LLM providers:
  - **Google Gemini** API key (`GEMINI_API_KEY`) — default provider, also used for embeddings and summarization
  - **Anthropic** API key (`ANTHROPIC_API_KEY`) — optional, can be selected as the chat LLM via `LLM_PROVIDER=anthropic`

> ⚠️ Even when using Anthropic as the chat provider, a `GEMINI_API_KEY` is still required because embeddings and the optional PDF summarization step always run on Gemini.

---

## �🚀 Installation and Use

### Clone the repository

```bash
git clone https://github.com/vini8cs/Vir_ChatBot.git
cd Vir_ChatBot
```

### Install and Run via Docker Compose (Recommended)

Configure environment variables and edit the .env file with your credentials

```bash
cp .env.example .env
```
Start the services

```bash
docker compose up --build
```

Start with development tools (Redis Commander)
```bash
docker compose --profile dev up --build
```

To stop the services (Ctrl+C or in another terminal)
```bash
docker compose down
```

To stop and remove all data
```bash
docker compose down --rmi all --volumes --remove-orphans
```

### Local Development (without Docker Compose)


Install dependencies with uv

```bash
uv sync
```
Configure environment variables and edit the .env file with your credentials

```bash
cp .env.example .env
```
Start Redis (required for Celery)

```bash
docker run -d -p 6379:6379 --name redis-vir redis:7
```

In separate terminals, start:

1. Celery Worker

```bash
uv run celery -A agents.vir_chatbot.tasks worker -l info
```

2. FastAPI Backend
```bash
uv run uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

3. Streamlit Interface
```bash
cd frontend && uv run streamlit run app.py
```

---

## ✨ Features

- 🔍 **RAG (Retrieval-Augmented Generation)**: Retrieves relevant information from scientific documents to ground responses
- 📄 **PDF Processing**: Automatic upload and processing of scientific papers with OCR and image extraction
- 🧠 **Intelligent Summarization**: Automatic summaries of scientific texts and images using Gemini
- 💬 **Conversational Interface**: User-friendly Streamlit interface with conversation history
- 🔄 **Session Persistence**: Multiple conversation threads with persistent memory (SQLite)
- 🐳 **Containerized**: Simplified deployment via Docker Compose
- ⚡ **Asynchronous Processing**: Heavy tasks processed in background with Celery + Redis

---

## 🏗️ Architecture

```
┌────────────┐      ┌────────────┐      ┌────────────┐      ┌─────────────────┐
│  Streamlit │ ───► │  FastAPI   │ ───► │  LangGraph │ ───► │ Gemini /        │
│     UI     │      │  Backend   │      │   Agent    │      │ Anthropic  API  │
└────────────┘      └─────┬──────┘      └─────┬──────┘      └─────────────────┘
    :8501                 │                   │
                         │                   ▼
                         │            ┌────────────┐
                         │            │   FAISS    │
                         │            │ VectorStore│
                         │            └────────────┘
                         ▼
                   ┌────────────┐      ┌────────────┐
                   │   Redis    │ ◄──► │   Celery   │
                   │   Queue    │      │   Worker   │
                   └────────────┘      └────────────┘
                       :6379           PDF Processing
```

### Data Flow

1. **PDF Upload**: Documents are processed by Docling (OCR + extraction)
2. **Chunking**: Texts are split into semantic chunks
3. **Embeddings**: Google Gemini generates embeddings for each chunk
4. **Storage**: Chunks are indexed in FAISS for vector search
5. **Query**: User questions retrieve relevant chunks via similarity search
6. **Response**: The chat LLM (Gemini by default, or Anthropic when `LLM_PROVIDER=anthropic`) generates responses based on retrieved context

---

## � System Requirements

### Memory (RAM)

| Scenario | Minimum | Recommended |
|----------|---------|-------------|
| **Development** | 8 GB | 16 GB |
| **Production (light usage)** | 16 GB | 24 GB |
| **Production (heavy usage)** | 24 GB | 32 GB+ |

**Typical memory usage per container:**

| Container | Idle | Under Load |
|-----------|------|------------|
| **web** (FastAPI + LangChain) | ~500 MB | ~1-2 GB |
| **worker** (Celery + Docling/OCR) | ~500 MB | ~3-4 GB |
| **streamlit** | ~40 MB | ~100 MB |
| **redis** | ~20 MB | ~50-100 MB |

> ⚠️ **Note**: The Celery worker can consume significant memory spikes (~3-4 GB) during PDF processing with Docling and OCR. Processing large or complex documents may require additional memory.

### Disk Space

| Component | Size |
|-----------|------|
| **Docker images** (all services) | ~4-5 GB |
| **Python dependencies** | ~3 GB |
| **Base system** | ~500 MB |
| **Data volumes** (vectorstore, PDFs, cache) | Variable* |

> *Data volumes depend on the number and size of processed documents. Each PDF generates embeddings stored in the FAISS vectorstore.

**Minimum recommended disk space**: **15 GB** (excluding your PDF documents)

#### Supported Platforms for Local Development

| Platform | Architecture | Status |
|----------|--------------|--------|
| **Linux** | x86_64 (Intel/AMD 64-bit) | ✅ Fully supported |
| **Linux** | aarch64 (ARM64) | ✅ Supported |
| **Windows** | AMD64 (64-bit) | ✅ Supported |

> ⚠️ **Note**: macOS is not currently supported due to PyTorch CPU wheel availability constraints.

---

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

```env
# LLM provider selection (Optional - defaults to "gemini")
# Options: "gemini" or "anthropic"
LLM_PROVIDER="gemini"

# Gemini (Required - used for embeddings and optional summarization,
# and as the chat LLM when LLM_PROVIDER=gemini)
GEMINI_API_KEY="your_gemini_api_key"

# Anthropic (Required only when LLM_PROVIDER=anthropic)
ANTHROPIC_API_KEY="your_anthropic_api_key"

# Paths (adjust as needed)
VECTORSTORE_PATH="/path/to/vectorstore_folder_path"
CACHE_FOLDER_PATH="/path/to/cache_folder_path"
SQLITE_DB_DIR="/path/to/db_folder"

# Ports (Optional - defaults shown)
WEB_PORT=8000
REDIS_PORT=6379
STREAMLIT_PORT=8501

# Development only (used with --profile dev)
REDIS_COMMANDER_PORT=8081

# Streamlit UI Configuration
API_BASE_URL="http://localhost:8000"  # Must match WEB_PORT

# Container user permissions (Optional - defaults to 1000)
# PUID=1000
# PGID=1000

# LangSmith (Optional - for tracing/debugging)
# LANGSMITH_API_KEY="your_langsmith_key"
# LANGSMITH_TRACING_V2=true
# LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
# LANGSMITH_PROJECT="vir-chatbot"
```

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | ❌ | `gemini` | Chat LLM provider — `gemini` or `anthropic` |
| `GEMINI_API_KEY` | ✅ | — | Google Gemini API key (always required — used for embeddings and summarization, and as the chat LLM when `LLM_PROVIDER=gemini`) |
| `ANTHROPIC_API_KEY` | ⚠️ | — | Required only when `LLM_PROVIDER=anthropic` |
| `VECTORSTORE_PATH` | ✅ | — | Host path where FAISS vectorstore will be saved |
| `CACHE_FOLDER_PATH` | ✅ | — | Host path for caching processed documents |
| `SQLITE_DB_DIR` | ✅ | — | Host directory for SQLite database (conversation memory) and runtime config |
| `WEB_PORT` | ❌ | `8000` | Port for FastAPI backend |
| `REDIS_PORT` | ❌ | `6379` | Port for Redis |
| `STREAMLIT_PORT` | ❌ | `8501` | Port for Streamlit UI |
| `REDIS_COMMANDER_PORT` | ❌ | `8081` | Port for Redis Commander (dev profile only) |
| `API_BASE_URL` | ❌ | `http://localhost:8000` | Backend URL used by Streamlit (must match `WEB_PORT`) |
| `PUID` | ❌ | `1000` | User ID for container process (must match host folder owner for bind mounts) |
| `PGID` | ❌ | `1000` | Group ID for container process (must match host folder owner for bind mounts) |
| `LANGSMITH_API_KEY` | ❌ | — | LangSmith API key for tracing/debugging |
| `LANGSMITH_TRACING_V2` | ❌ | `false` | Enable LangSmith tracing |
| `LANGSMITH_ENDPOINT` | ❌ | — | LangSmith API endpoint |
| `LANGSMITH_PROJECT` | ❌ | — | LangSmith project name |

> **📝 Note about `PUID` and `PGID`:**  
> When using bind mounts, Docker preserves the host's numeric owner IDs (UID/GID). If the container user doesn't match, permission errors will occur. Set `PUID` and `PGID` to match your host user. Check your IDs with `id` command (e.g., `uid=1000(username) gid=1000(username)`).

### Model Configuration

The default model parameters are defined in [config.py](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `gemini-2.5-flash` (or `claude-sonnet-4-6` if `LLM_PROVIDER=anthropic`) | Chat LLM model — resolved from `LLM_PROVIDER` |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for PDF summarization (always Gemini, regardless of `LLM_PROVIDER`) |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Model for embeddings |
| `TEMPERATURE` | `0.1` | Model creativity (0-1) |
| `MAX_OUTPUT_TOKENS` | `2048` | Maximum output tokens |
| `TOKEN_SIZE` | `2048` | Maximum chunk size |
| `RETRIEVER_LIMIT` | `5` | Number of retrieved documents |
| `SUMMARIZE` | `false` | Enable/disable document summarization |

> **💡 Runtime Configuration via UI:**  
> Most of these parameters can be adjusted directly in the **Streamlit interface** without modifying the config file:
> - **🔧 LLM Settings** (in the sidebar): Configure `model`, `temperature`, `max_tokens`, `retriever_limit`, and `max_retries` for the chat.
> - **VectorStore Settings** (inside "Create VectorStore" expander): Configure `model`, `max_tokens`, and `summarize` for VectorStore creation.
> 
> Changes made in the UI are applied at runtime and do not require restarting the services.

---

## 📖 Usage

### Accessing the Interface

After starting the services, access (using default ports):

- **Chat Interface**: http://localhost:8501
- **REST API**: http://localhost:8000
- **Redis Commander** (debug): http://localhost:8081

> **💡 Custom Ports:** If you configured custom ports in `.env` (e.g., `STREAMLIT_PORT=3000`), use those instead.

### Interface Features

1. **Create VectorStore** (in the sidebar under "📚 Manage VectorStore"):
   - **Upload PDFs**: Select and upload individual PDF files (or `.docx`, `.txt`, `.tsv`, images) directly through the interface — ingestion runs asynchronously on the Celery worker
   - **VectorStore Settings**: Configure the model, max tokens, and enable/disable summarization before creating

2. **Chat with Documents**:
   - Create a new conversation thread
   - Ask questions about virology/bioinformatics
   - The bot will respond based on loaded documents
   - Use **🔧 LLM Settings** to adjust model, temperature, max tokens, retriever limit, and retries

3. **Thread Management**:
   - Create multiple conversations
   - Threads are automatically named after the first message
   - Switch between threads
   - Delete old conversations

4. **PDF Management** (in "📋 PDFs in VectorStore"):
   - View all PDFs indexed in the VectorStore
   - Search and filter PDFs
   - Delete selected PDFs from the VectorStore

### Example Questions

```
- "What are the main replication mechanisms of the dengue virus?"
- "Explain the viral capsid structure described in the documents"
- "What sequencing techniques are mentioned in the papers?"
- "Compare the viral detection methods presented"
```

---

## 📁 Project Structure

```
Vir_ChatBot/
├── agents/
│   └── vir_chatbot/
│       ├── vectorstore.py       # Vectorstore creation and management
│       ├── vir_chatbot.py       # Main LangGraph agent
│       └── tasks.py             # Celery tasks (background)
├── backend/
│   ├── api.py                   # FastAPI app entry point (lifespan, include_router)
│   ├── state.py                 # Shared app state (runtime_config, global_resources)
│   ├── models.py                # Pydantic models and shared types
│   └── routers/
│       ├── chat.py              # POST /chat/stream
│       ├── config.py            # GET|PUT /config, POST /config/reset*
│       ├── tasks.py             # GET|DELETE /tasks/{task_id}
│       ├── threads.py           # GET|POST|DELETE /threads/*
│       └── vectorstore.py       # POST /vectorstore/reload, upload, delete, list
├── frontend/
│   ├── app.py                   # Streamlit entry point (main + page config)
│   ├── api_client.py            # All HTTP calls to the backend
│   ├── session.py               # Session state helpers (threads, messages, tasks)
│   └── views/
│       ├── sidebar.py           # User setup + thread list sidebar
│       ├── chat.py              # Chat window and message streaming
│       ├── config.py            # LLM settings and system prompt UI
│       ├── tasks.py             # Background task progress bars
│       └── vectorstore.py       # Upload, delete, and list documents UI
├── llms/
│   ├── gemini.py                # Gemini API wrapper
│   ├── langgraph_functions.py   # Agent graph functions
│   └── tokenizer.py             # Tokenizer wrapper
├── templates/
│   ├── prompts.py               # System prompts
│   └── schemas.py               # Response schemas
├── config.py                    # Configuration and environment variables
├── compose.yaml                 # Docker Compose
├── Dockerfile                   # API Dockerfile
└── Dockerfile.streamlit         # Streamlit Dockerfile
```

---

## 🛠️ Technologies

| Category | Technology |
|----------|------------|
| **Chat LLM** | Google Gemini (2.5 Flash) or Anthropic Claude (Sonnet/Opus/Haiku) — selected via `LLM_PROVIDER` |
| **Embeddings** | Google Gemini Embedding |
| **Agent Framework** | LangGraph + LangChain |
| **Vector Store** | FAISS |
| **PDF Processing** | Docling (OCR + extraction) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Task Queue** | Celery + Redis |
| **Containerization** | Docker + Docker Compose |
| **Persistence** | SQLite (conversation memory) |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Vinicius** - [@vini8cs](https://github.com/vini8cs)

---

<p align="center">
  Developed for the virology and bioinformatics community
</p>
