"""
LLM and Embeddings factory.

Maps provider names to concrete LangChain objects. To add a new provider:
  1. Add the package to pyproject.toml and run uv sync
  2. Import it at the top and write one builder function
  3. Register it in _LLM_BUILDERS or _EMBEDDING_BUILDERS
  4. Add the API key to .env

Supported LLM providers:        gemini | local
Supported embedding providers:  gemini | local
"""

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

import config as _
from llms.local_model import get_local_embeddings, get_local_llm

logger = logging.getLogger(__name__)


def _gemini_llm(
    model: str, temperature: float, max_retries: int
) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=model, temperature=temperature, max_retries=max_retries
    )


def _local_llm(
    model: str, temperature: float, max_retries: int
) -> BaseChatModel:
    # temperature and max_retries are ignored: they are baked into the
    # HuggingFace pipeline at load time (singleton cache).
    return get_local_llm(model_name=model, bits=_.LOCAL_MODEL_BITS)


_LLM_BUILDERS: dict[str, callable] = {
    "gemini": _gemini_llm,
    "local": _local_llm,
}


def _gemini_embeddings(model: str) -> Embeddings:
    return GoogleGenerativeAIEmbeddings(model=model)


def _local_embeddings(model: str) -> Embeddings:
    return get_local_embeddings(model_name=model)


_EMBEDDING_BUILDERS: dict[str, callable] = {
    "gemini": _gemini_embeddings,
    "local": _local_embeddings,
}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    """Return a LangChain chat model for the given provider."""
    provider = provider or _.LLM_PROVIDER
    model = model or _.LLM_MODEL
    temperature = temperature if temperature is not None else _.TEMPERATURE
    max_retries = max_retries if max_retries is not None else _.MAX_RETRIES

    builder = _LLM_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Valid options: {list(_LLM_BUILDERS)}"
        )

    logger.debug("Building %s LLM: %s", provider, model)
    return builder(model, temperature, max_retries)


def get_embeddings(
    provider: str | None = None,
    model: str | None = None,
) -> Embeddings:
    """
    Return a LangChain embeddings object for the given provider.

    Important: a FAISS index must always be queried with the same embedding
    model it was built with. Mixing providers silently returns wrong results.
    Keep EMBEDDING_PROVIDER and EMBEDDING_MODEL consistent between vectorstore
    creation and retrieval.
    """
    provider = provider or _.EMBEDDING_PROVIDER
    model = model or _.EMBEDDING_MODEL

    builder = _EMBEDDING_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. "
            f"Valid options: {list(_EMBEDDING_BUILDERS)}"
        )

    logger.debug("Building %s embeddings: %s", provider, model)
    return builder(model)
