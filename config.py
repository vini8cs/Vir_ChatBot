import logging
import os

from google import genai
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API: str
    GCP_CREDENTIALS: str
    GCP_PROJECT: str
    GCP_REGION: str
    TIMESCALE_SERVICE_URL: str

    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING_V2: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = ""

    UNSTRUCTURED_API: str = ""


settings = Settings(_env_file="../.env", _env_file_encoding="utf-8")

os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GCP_CREDENTIALS
os.environ["GOOGLE_CLOUD_PROJECT"] = settings.GCP_PROJECT
os.environ["GOOGLE_CLOUD_REGION"] = settings.GCP_REGION

if settings.LANGSMITH_API_KEY != "" and settings.LANGSMITH_PROJECT != "":
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING_V2
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT

CLIENT_GEMINI = genai.Client()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

PROMPT_IMAGE = """You are a scientific assistant specializing in virology. Analyze
 and describe the given image as if it comes from a peer-reviewed scientific paper. 
 Focus on identifying what type of figure it is (e.g., protein structure, 
viral capsid morphology, electron microscopy image, phylogenetic tree, genome map, or 
experimental graph). Provide a clear, detailed, and objective description of the key 
elements shown, including labels, axes, molecular/structural features, and overall context. 
If the image depicts data (graphs, heatmaps, bar plots, etc.), describe the variables 
measured, axes, units (if visible), and any notable trends or differences. If it shows 
biological structures (proteins, viral particles, host interactions), describe their 
shape, organization, colors (if relevant for distinguishing features), and potential
biological significance. Avoid speculation beyond what is visible, and keep the description
 concise but informative,in a way that would help researchers understand the figure without 
 seeing it."""  # noqa

PROMPT_TEXT = """
You are an assistant specialized in summarizing **scientific texts and tables related to viruses**.  
Your goal is to produce **compact but information-rich summaries** suitable for **semantic retrieval**.  

Guidelines:
- Begin directly with core scientific information. Do not add framing phrases.  
- Prioritize **completeness of scientific content**: include virus name(s), host(s), sample type, study aim, methods, main findings, geographic location, and quantitative results when available.  
- Keep summaries short by removing redundancy, not by omitting details.  
- For **texts**: capture study type (e.g., genomic, epidemiological, diagnostic), datasets, and key conclusions.  
- For **tables**: describe what each column or variable represents, relevant metrics, trends, or comparisons.  
- Maintain **neutral, factual, and terminology-accurate** language appropriate for scientific retrieval.  
- When uncertain, prefer including the information rather than excluding it.

# Content:
{element}

"""  # noqa


RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "isReference": {
                "type": "BOOLEAN",
                "description": "A boolean indicating whether the text is mainly references to a scientific paper (e.g. "
                "HO, Thien; TZANETAKIS, Ioannis E. Development of a virus detection and discovery pipeline using next "
                "generation sequencing. Virology, v. 471, p. 54-60, 2014.).",
                "nullable": False,
            },
            "summary": {
                "type": "STRING",
                "nullable": False,
            },
        },
        "required": ["summary", "isReference"],
    },
}

GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 2048
CHUNK_SIZE = 4000
COMBINE_TEXT_UNDER_N_CHARS = 1500
NEW_AFTER_N_CHARACTERS = 3000
LANGUAGES = ["eng", "pt"]
MAX_RETRIES = 3
