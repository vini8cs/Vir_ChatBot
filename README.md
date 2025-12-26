# 🦠 Vir ChatBot

A RAG (Retrieval-Augmented Generation) chatbot specialized in **virology** and **bioinformatics**, designed to run locally. Use your own scientific documents (PDFs) to create a personalized knowledge base and interact with your data through an intelligent assistant.

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green?logo=chainlink)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [License](#-license)

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
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   FastAPI       │────▶│   LangGraph     │
│   (Frontend)    │     │   (Backend)     │     │   (Agent)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                        │
                                ▼                        ▼
                        ┌───────────────┐       ┌───────────────┐
                        │    Redis      │       │    FAISS      │
                        │   (Queue)     │       │ (VectorStore) │
                        └───────────────┘       └───────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │    Celery     │
                        │   (Worker)    │
                        └───────────────┘
```

### Data Flow

1. **PDF Upload**: Documents are processed by Docling (OCR + extraction)
2. **Chunking**: Texts are split into semantic chunks
3. **Embeddings**: Google Gemini generates embeddings for each chunk
4. **Storage**: Chunks are indexed in FAISS for vector search
5. **Query**: User questions retrieve relevant chunks via similarity search
6. **Response**: The LLM (Gemini) generates responses based on retrieved context

---

## 📦 Prerequisites

- **Docker** and **Docker Compose** (recommended)
- **Python 3.12+** (for local development)
- **Google Cloud Account** with access to Gemini/Vertex AI API
- **GCP Credentials** (service account JSON file)

---

## 🚀 Installation

### Via Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/vini8cs/Vir_ChatBot.git
cd Vir_ChatBot

# Configure environment variables
cp .env.example .env
# Edit the .env file with your credentials

# Start the services
docker compose up --build
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/vini8cs/Vir_ChatBot.git
cd Vir_ChatBot

# Install dependencies with uv
uv sync

# Start Redis (required for Celery)
docker run -d -p 6379:6379 redis:7

# In separate terminals, start:
# 1. Celery Worker
celery -A tasks worker -l info

# 2. FastAPI Backend
uvicorn streamlit_ui.api:app --host 0.0.0.0 --port 8000

# 3. Streamlit Interface
cd streamlit_ui && streamlit run app.py
```

---

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

```env
# Google Cloud / Gemini (Required)
GEMINI_API_KEY=your_gemini_api_key
GCP_CREDENTIALS=/path/to/your/credentials.json
GCP_PROJECT=your_gcp_project_id
GCP_REGION=us-central1

# Paths (adjust as needed)
PDF_FOLDER=/path/to/your/pdfs
VECTORSTORE_PATH=/path/to/vectorstore
CACHE_FOLDER_PATH=/path/to/cache
SQLITE_MEMORY_DATABASE=/path/to/memory.sqlite

# LangSmith (Optional - for tracing/debugging)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=vir-chatbot
```

### Model Configuration

Model parameters can be adjusted in [config.py](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GEMINI_MODEL` | `gemini-2.5-flash` | LLM model for chat and summarization |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Model for embeddings |
| `TEMPERATURE` | `0.1` | Model creativity (0-1) |
| `TOKEN_SIZE` | `2048` | Maximum chunk size |
| `RETRIEVER_LIMIT` | `5` | Number of retrieved documents |
| `LANGUAGES` | `["eng", "pt"]` | Languages for OCR |

---

## 📖 Usage

### Accessing the Interface

After starting the services, access:

- **Chat Interface**: http://localhost:8501
- **REST API**: http://localhost:8000
- **Redis Commander** (debug): http://localhost:8081

### Interface Features

1. **PDF Upload**: 
   - Go to the "Manage Documents" tab in the sidebar
   - Upload your scientific papers
   - Wait for processing (monitored in real-time)

2. **Chat with Documents**:
   - Create a new conversation thread
   - Ask questions about virology/bioinformatics
   - The bot will respond based on loaded documents

3. **Thread Management**:
   - Create multiple conversations
   - Switch between threads
   - Delete old conversations

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
│       ├── vectorstore.py    # Vectorstore creation and management
│       └── vir_chatbot.py    # Main LangGraph agent
├── streamlit_ui/
│   ├── api.py                # FastAPI endpoints
│   └── app.py                # Streamlit interface
├── config.py                 # Configuration and environment variables
├── gemini.py                 # Gemini API wrapper
├── langgraph_functions.py    # Agent graph functions
├── prompts.py                # System prompts
├── schemas.py                # Response schemas
├── tasks.py                  # Celery tasks (background)
├── tokenizer.py              # Tokenizer wrapper
├── compose.yaml              # Docker Compose
├── Dockerfile                # API Dockerfile
└── Dockerfile.streamlit      # Streamlit Dockerfile
```

---

## 🛠️ Technologies

| Category | Technology |
|----------|------------|
| **LLM** | Google Gemini (2.5 Flash) |
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
