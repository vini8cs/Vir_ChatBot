import logging
import os
import warnings

from google import genai
from pydantic_settings import BaseSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.cloud.aiplatform"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="unstructured"
)


class GeminiConnectionError(ConnectionError):
    """Custom exception for Gemini client connection errors."""


class Settings(BaseSettings):
    # Required when LLM_PROVIDER=gemini or EMBEDDING_PROVIDER=gemini.
    # Not needed when both providers are set to "local".
    GEMINI_API_KEY: str = ""

    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING_V2: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = ""

    UNSTRUCTURED_API: str = ""
    VECTORSTORE_PATH: str = "vectorstore"
    SQLITE_MEMORY_DATABASE: str = "memory.sqlite"
    CACHE_FOLDER_PATH: str = "cache.csv"
    API_BASE_URL: str = "http://localhost:8000"

    WEB_PORT: int = 8000
    REDIS_PORT: int = 6379
    REDIS_COMMANDER_PORT: int = 8081
    STREAMLIT_PORT: int = 8501

    # LLM provider settings
    LLM_PROVIDER: str = "gemini"  # gemini | openai | anthropic | local
    LLM_MODEL: str = "gemini-2.5-flash"

    # Embedding provider settings (can differ from LLM_PROVIDER)
    EMBEDDING_PROVIDER: str = "gemini"  # gemini | openai | local
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # Local inference settings
    # (used when LLM_PROVIDER=local or EMBEDDING_PROVIDER=local)
    LOCAL_MODEL_BITS: int = 4
    LOCAL_MODELS_PATH: str = "models"


settings = Settings(_env_file=".env", _env_file_encoding="utf-8")

if settings.GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
    os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY

if settings.LANGSMITH_API_KEY != "" and settings.LANGSMITH_PROJECT != "":
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING_V2
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT

CLIENT_GEMINI = None
if settings.GEMINI_API_KEY:
    try:
        logging.info("Initializing Gemini client...")
        CLIENT_GEMINI = genai.Client()
    except Exception as e:
        raise GeminiConnectionError(
            f"Error initializing Gemini client: {e}"
        ) from e
else:
    logging.info(
        "GEMINI_API_KEY not set — Gemini client skipped. "
        "Set LLM_PROVIDER=gemini and provide credentials to use Gemini."
    )

LLM_PROVIDER = settings.LLM_PROVIDER
LLM_MODEL = settings.LLM_MODEL
EMBEDDING_PROVIDER = settings.EMBEDDING_PROVIDER
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
LOCAL_MODEL_BITS = settings.LOCAL_MODEL_BITS
LOCAL_MODELS_PATH = settings.LOCAL_MODELS_PATH

# Legacy alias kept for backward compatibility with the Gemini summarization
# chain in llms/gemini.py which references GEMINI_MODEL by name.
GEMINI_MODEL = LLM_MODEL

TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 2048
TOKEN_SIZE = 2048
MAX_RETRIES = 3
TOKENIZER_MODEL = "mistralai/Mistral-7B-v0.1"
THREADS = 4
SUMMARIZE = False
SQLITE_MEMORY_DATABASE = settings.SQLITE_MEMORY_DATABASE
RUNTIME_CONFIG_PATH = os.path.join(
    os.path.dirname(settings.SQLITE_MEMORY_DATABASE), "runtime_config.json"
)
VECTORSTORE_PATH = settings.VECTORSTORE_PATH
CACHE_FOLDER = settings.CACHE_FOLDER_PATH
API_BASE_URL = settings.API_BASE_URL
RETRIEVER_LIMIT = 5
THREAD_NUMBER = 1

from templates.prompts import TOOL_CALLER_PROMPT  # noqa: E402

SYSTEM_PROMPT = TOOL_CALLER_PROMPT
USER_ID = 1
PDF_LIST_DEFAULT = []
