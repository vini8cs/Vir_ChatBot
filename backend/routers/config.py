import logging

from fastapi import APIRouter, HTTPException

import backend.state as state
from agents.vir_chatbot.vir_chatbot import load_global_vectorstore
from backend.models import ConfigUpdateRequest, RuntimeConfig

router = APIRouter(prefix="/config", tags=["config"])


@router.get("")
async def get_config():
    """Get current runtime configuration settings."""
    return state.runtime_config.model_dump()


@router.put("")
async def update_config(request: ConfigUpdateRequest):
    """
    Update runtime configuration. Only provided fields will be updated.
    Changes take effect immediately for new chat sessions.
    """
    update_data = request.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(
            status_code=400, detail="No configuration fields provided"
        )

    current_config = state.runtime_config.model_dump()
    current_config.update(update_data)
    state.runtime_config = RuntimeConfig(**current_config)
    state._save_persisted_config(state.runtime_config)

    return {
        "status": "success",
        "message": "Configuration updated successfully",
        "config": state.runtime_config.model_dump(),
    }


@router.post("/reset")
async def reset_config():
    """
    Reset configuration to default values from config.py.
    """
    state.runtime_config = RuntimeConfig()
    state._save_persisted_config(state.runtime_config)

    return {
        "status": "success",
        "message": "Configuration reset to defaults",
        "config": state.runtime_config.model_dump(),
    }


@router.post("/reset-and-reload")
async def reset_config_and_reload():
    """
    Reset configuration to defaults AND reload the vectorstore.
    Use this when you want a complete fresh start.
    """
    state.runtime_config = RuntimeConfig()
    state._save_persisted_config(state.runtime_config)

    try:
        state.global_resources["retriever"] = await load_global_vectorstore(
            retriever_limit=state.runtime_config.retriever_limit
        )
        return {
            "status": "success",
            "message": "Configuration reset and VectorStore reloaded",
            "config": state.runtime_config.model_dump(),
            "vectorstore_loaded": state.global_resources["retriever"]
            is not None,
        }
    except Exception as e:
        logging.error(f"Error reloading VectorStore: {e}")
        return {
            "status": "partial",
            "message": "Configuration reset but VectorStore reload failed",
            "config": state.runtime_config.model_dump(),
            "error": str(e),
        }
