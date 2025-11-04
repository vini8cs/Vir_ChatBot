import logging
import os

from google import genai
from pydantic_settings import BaseSettings


class GeminiConnectionError(ConnectionError):
    """Custom exception for Gemini client connection errors."""


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GCP_CREDENTIALS: str
    GCP_PROJECT: str
    GCP_REGION: str
    TIMESCALE_SERVICE_URL: str

    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING_V2: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = ""

    UNSTRUCTURED_API: str = ""


settings = Settings(_env_file=".env", _env_file_encoding="utf-8")

os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GCP_CREDENTIALS
os.environ["GOOGLE_CLOUD_PROJECT"] = settings.GCP_PROJECT
os.environ["GOOGLE_CLOUD_REGION"] = settings.GCP_REGION

if settings.LANGSMITH_API_KEY != "" and settings.LANGSMITH_PROJECT != "":
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING_V2
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT

try:
    logging.info("Initializing Gemini client...")
    CLIENT_GEMINI = genai.Client()
except Exception as e:
    raise GeminiConnectionError(f"Error initializing Gemini client: {e}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 2048
CHUNK_SIZE = 4000
COMBINE_TEXT_UNDER_N_CHARS = 1500
NEW_AFTER_N_CHARACTERS = 3000
LANGUAGES = ["eng", "pt"]
MAX_RETRIES = 3
