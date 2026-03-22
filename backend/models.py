import logging
from typing import List

from pydantic import BaseModel, Field

import config as _


class RuntimeConfig(BaseModel):
    """Runtime configuration that can be modified via API."""

    gemini_model: str = _.GEMINI_MODEL
    temperature: float = _.TEMPERATURE
    max_output_tokens: int = _.MAX_OUTPUT_TOKENS
    max_retries: int = _.MAX_RETRIES
    retriever_limit: int = _.RETRIEVER_LIMIT
    summarize: bool = _.SUMMARIZE
    system_prompt: str = _.SYSTEM_PROMPT
    embedding_model: str = _.EMBEDDING_MODEL
    token_size: int = _.TOKEN_SIZE
    tokenizer_model: str = _.TOKENIZER_MODEL
    threads: int = _.THREADS


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration."""

    gemini_model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    max_retries: int | None = None
    retriever_limit: int | None = None
    summarize: bool | None = None
    system_prompt: str | None = None
    embedding_model: str | None = None
    token_size: int | None = None
    tokenizer_model: str | None = None
    threads: int | None = None


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


class DeleteFileRequest(BaseModel):
    filenames: List[str]


class ChatRequest(BaseModel):
    message: str
    thread_id: str = Field(..., description="Unique ID for the conversation thread")
    user_id: str = Field("default_user", description="User ID for personalization")


class CreateThreadRequest(BaseModel):
    user_id: str = Field(..., description="User ID to associate with the thread")


class ThreadResponse(BaseModel):
    thread_id: str
    user_id: str
