import asyncio
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage

import backend.state as state
from agents.vir_chatbot.vir_chatbot import create_graph
from backend.models import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])


async def _create_graph_retriever(retriever, config, max_retries=3):
    """Create graph using runtime configuration."""
    for attempt in range(max_retries):
        try:
            return await create_graph(
                global_retriever=retriever,
                llm_model=config.gemini_model,
                temperature=config.temperature,
                max_retries=config.max_retries,
                system_prompt=config.system_prompt,
            )
        except Exception as e:
            if (
                "database is locked" in str(e).lower()
                and attempt < max_retries - 1
            ):
                await asyncio.sleep(3)
                continue
            return None
    return None


async def _chat_generator(user_input: str, thread_id: str, user_id: str):
    """
    Generates the streaming response and
    ensures the database connection is closed.
    Uses runtime_config for LLM settings.
    """
    retriever = state.global_resources.get("retriever")
    if not retriever:
        yield f"data: {json.dumps({'error': 'VectorStore not initialized. Please create or reload the VectorStore.'})}\n\n"  # noqa E501
        return

    result = await _create_graph_retriever(retriever, state.runtime_config, 3)

    if result is None:
        yield f"data: {json.dumps({'error': 'Error creating graph'})}\n\n"
        return

    graph, checkpointer_cm = result

    graph_config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id}
    }

    try:
        async for chunk, _ in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            graph_config,
            stream_mode="messages",
        ):
            if not isinstance(chunk, AIMessageChunk):
                continue
            if getattr(chunk, "tool_call_chunks", None):
                continue

            content = chunk.content
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "")
                    if isinstance(part, dict)
                    else str(part)
                    for part in content
                    if not (
                        isinstance(part, dict)
                        and part.get("type") == "thinking"
                    )
                ).strip()

            if content:
                payload = {"content": content, "type": "ai_response"}
                yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error during chat generation: {error_msg}")
        if "database is locked" in error_msg.lower():
            error_msg = "The system is busy updating. "
            "Please try again in a few seconds."
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

    finally:
        if checkpointer_cm:
            try:
                await checkpointer_cm.__aexit__(None, None, None)
            except Exception as close_err:
                logging.error(f"Error closing SQLite connection: {close_err}")


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Call the chat generator and return a streaming response."""
    return StreamingResponse(
        _chat_generator(request.message, request.thread_id, request.user_id),
        media_type="text/event-stream",
    )
