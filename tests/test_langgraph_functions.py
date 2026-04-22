"""Tests for llms/langgraph_functions.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from llms.langgraph_functions import make_retrieve_tool, query_or_respond


class TestMakeRetrieveTool:
    @pytest.mark.asyncio
    async def test_returns_callable_tool(self):
        retriever = MagicMock()
        tool = make_retrieve_tool(retriever)
        assert callable(tool.ainvoke)

    @pytest.mark.asyncio
    async def test_tool_encodes_retrieved_docs(self):
        doc = MagicMock()
        doc.id = "uuid-1"
        doc.metadata = {"filename": "a.pdf"}
        doc.page_content = "chunk content"
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[doc])

        tool = make_retrieve_tool(retriever)
        result = await tool.ainvoke({"query": "viral replication"})
        assert "uuid-1" in result
        assert "a.pdf" in result
        assert "chunk content" in result

    @pytest.mark.asyncio
    async def test_tool_handles_empty_results(self):
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[])

        tool = make_retrieve_tool(retriever)
        result = await tool.ainvoke({"query": "nothing"})
        assert isinstance(result, str)


class TestQueryOrRespond:
    @pytest.mark.asyncio
    async def test_uses_default_prompt_when_not_provided(self):
        llm = MagicMock()
        bound_llm = MagicMock()
        bound_llm.ainvoke = AsyncMock(return_value=MagicMock())
        llm.bind_tools.return_value = bound_llm
        retrieve_tool = MagicMock(name="retrieve_tool")
        state = {"messages": [HumanMessage(content="hi")]}

        result = await query_or_respond(
            state,
            config={},
            store=None,
            llm=llm,
            retrieve_tool=retrieve_tool,
            system_prompt=None,
        )
        llm.bind_tools.assert_called_once_with([retrieve_tool])
        assert "messages" in result

        args, _ = bound_llm.ainvoke.call_args
        sent_messages = args[0]
        assert isinstance(sent_messages[0], SystemMessage)
        assert "Virology" in sent_messages[0].content

    @pytest.mark.asyncio
    async def test_uses_provided_system_prompt(self):
        llm = MagicMock()
        bound_llm = MagicMock()
        response = MagicMock()
        bound_llm.ainvoke = AsyncMock(return_value=response)
        llm.bind_tools.return_value = bound_llm

        state = {"messages": [HumanMessage(content="q")]}
        result = await query_or_respond(
            state,
            config={},
            store=None,
            llm=llm,
            retrieve_tool=MagicMock(),
            system_prompt="Custom system prompt",
        )
        args, _ = bound_llm.ainvoke.call_args
        assert args[0][0].content == "Custom system prompt"
        assert result == {"messages": [response]}

    @pytest.mark.asyncio
    async def test_prepends_system_message_before_state_messages(self):
        llm = MagicMock()
        bound_llm = MagicMock()
        bound_llm.ainvoke = AsyncMock(return_value=MagicMock())
        llm.bind_tools.return_value = bound_llm

        state = {"messages": [HumanMessage(content="first")]}
        await query_or_respond(
            state,
            config={"x": 1},
            store=None,
            llm=llm,
            retrieve_tool=MagicMock(),
            system_prompt="SP",
        )
        args, _ = bound_llm.ainvoke.call_args
        sent = args[0]
        assert isinstance(sent[0], SystemMessage)
        assert sent[1].content == "first"
        # config forwarded to llm
        assert args[1] == {"x": 1}
