"""Tests for agents/vir_chatbot/vir_chatbot.py.

Covers Vir_ChatBot (both LLM providers), load_global_vectorstore (missing
and present), and create_graph (including overrides + SQLite pragmas).

``InMemorySaver`` is used in build_graph tests because LangGraph validates
the checkpointer with isinstance — a plain MagicMock would be rejected.
In create_graph tests ``build_graph`` itself is patched to avoid that same
check and keep the focus on the orchestration logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import config as _
from agents.vir_chatbot.vir_chatbot import (
    Vir_ChatBot,
    create_graph,
    load_global_vectorstore,
)


class TestVirChatBotInit:
    def test_uses_gemini_when_provider_is_gemini(self, monkeypatch):
        monkeypatch.setattr(_, "LLM_PROVIDER", "gemini")
        with patch(
            "agents.vir_chatbot.vir_chatbot.ChatGoogleGenerativeAI"
        ) as gem:
            Vir_ChatBot(
                retriever=MagicMock(),
                llm_model="gemini-x",
                temperature=0.1,
                max_retries=2,
                checkpointer=MagicMock(),
            )
        gem.assert_called_once_with(
            model="gemini-x", temperature=0.1, max_retries=2
        )

    def test_uses_anthropic_when_provider_is_anthropic(self, monkeypatch):
        monkeypatch.setattr(_, "LLM_PROVIDER", "anthropic")
        with patch("agents.vir_chatbot.vir_chatbot.ChatAnthropic") as anthro:
            Vir_ChatBot(
                retriever=MagicMock(),
                llm_model="claude-x",
                temperature=0.2,
                max_retries=1,
                checkpointer=MagicMock(),
            )
        anthro.assert_called_once_with(
            model="claude-x", temperature=0.2, max_retries=1
        )

    def test_stores_retriever_checkpointer_system_prompt(self, monkeypatch):
        monkeypatch.setattr(_, "LLM_PROVIDER", "gemini")
        retriever = MagicMock()
        checkpointer = MagicMock()
        bot = Vir_ChatBot(
            retriever=retriever,
            llm_model="x",
            temperature=0.1,
            max_retries=1,
            checkpointer=checkpointer,
            system_prompt="custom",
        )
        assert bot.retriever is retriever
        assert bot.checkpointer is checkpointer
        assert bot.system_prompt == "custom"
        assert bot.graph is None


class TestBuildGraph:
    @pytest.mark.asyncio
    async def test_compiles_graph_with_checkpointer(self, monkeypatch):
        monkeypatch.setattr(_, "LLM_PROVIDER", "gemini")
        bot = Vir_ChatBot(
            retriever=MagicMock(),
            llm_model="x",
            temperature=0.1,
            max_retries=1,
            checkpointer=InMemorySaver(),
        )
        compiled = await bot.build_graph()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt_in_graph(self, monkeypatch):
        monkeypatch.setattr(_, "LLM_PROVIDER", "gemini")
        bot = Vir_ChatBot(
            retriever=MagicMock(),
            llm_model="x",
            temperature=0.0,
            max_retries=1,
            checkpointer=InMemorySaver(),
            system_prompt="Custom prompt for tests",
        )
        compiled = await bot.build_graph()
        assert compiled is not None


class TestLoadGlobalVectorstore:
    @pytest.mark.asyncio
    async def test_returns_none_when_index_missing(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(_, "VECTORSTORE_PATH", str(tmp_path / "vs"))
        result = await load_global_vectorstore()
        assert result is None

    @pytest.mark.asyncio
    async def test_loads_and_returns_retriever_when_index_present(
        self, tmp_path, monkeypatch
    ):
        vs_dir = tmp_path / "vs"
        vs_dir.mkdir()
        (vs_dir / "index.faiss").write_bytes(b"")
        monkeypatch.setattr(_, "VECTORSTORE_PATH", str(vs_dir))
        monkeypatch.setattr(_, "EMBEDDING_MODEL", "fake-embed")
        monkeypatch.setattr(_, "RETRIEVER_LIMIT", 7)

        fake_vs = MagicMock()
        fake_retriever = MagicMock()
        fake_vs.as_retriever.return_value = fake_retriever
        with patch(
            "agents.vir_chatbot.vir_chatbot.FAISS.load_local",
            return_value=fake_vs,
        ):
            retriever = await load_global_vectorstore()

        assert retriever is fake_retriever
        fake_vs.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 7}
        )

    @pytest.mark.asyncio
    async def test_retriever_limit_override(self, tmp_path, monkeypatch):
        vs_dir = tmp_path / "vs"
        vs_dir.mkdir()
        (vs_dir / "index.faiss").write_bytes(b"")
        monkeypatch.setattr(_, "VECTORSTORE_PATH", str(vs_dir))
        monkeypatch.setattr(_, "RETRIEVER_LIMIT", 5)

        fake_vs = MagicMock()
        with patch(
            "agents.vir_chatbot.vir_chatbot.FAISS.load_local",
            return_value=fake_vs,
        ):
            await load_global_vectorstore(retriever_limit=20)
        fake_vs.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 20}
        )


class TestCreateGraph:
    """build_graph is patched in these tests so the focus stays on
    create_graph's own orchestration (SQLite pragmas, overrides, cm return).
    The graph-compilation path is already covered by TestBuildGraph."""

    def _ctx_mgr_with(self, checkpointer):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=checkpointer)
        return ctx

    @pytest.mark.asyncio
    async def test_uses_defaults_when_no_overrides(self, monkeypatch):
        monkeypatch.setattr(_, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(_, "GEMINI_MODEL", "default-gem")
        monkeypatch.setattr(_, "TEMPERATURE", 0.3)
        monkeypatch.setattr(_, "MAX_RETRIES", 4)

        checkpointer = MagicMock()
        checkpointer.conn = MagicMock()
        checkpointer.conn.execute = AsyncMock()
        ctx = self._ctx_mgr_with(checkpointer)

        fake_graph = MagicMock()
        with (
            patch(
                "agents.vir_chatbot.vir_chatbot.AsyncSqliteSaver"
                ".from_conn_string",
                return_value=ctx,
            ),
            patch(
                "agents.vir_chatbot.vir_chatbot.ChatGoogleGenerativeAI"
            ) as gem,
            patch.object(
                Vir_ChatBot,
                "build_graph",
                new_callable=lambda: lambda self: AsyncMock(
                    return_value=fake_graph
                )(),
            ),
        ):
            graph, cm = await create_graph(global_retriever=MagicMock())

        gem.assert_called_once_with(
            model="default-gem", temperature=0.3, max_retries=4
        )
        assert cm is ctx
        assert checkpointer.conn.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_overrides_win_over_defaults(self, monkeypatch):
        monkeypatch.setattr(_, "LLM_PROVIDER", "gemini")
        checkpointer = MagicMock()
        checkpointer.conn = None
        ctx = self._ctx_mgr_with(checkpointer)

        with (
            patch(
                "agents.vir_chatbot.vir_chatbot.AsyncSqliteSaver"
                ".from_conn_string",
                return_value=ctx,
            ),
            patch(
                "agents.vir_chatbot.vir_chatbot.ChatGoogleGenerativeAI"
            ) as gem,
            patch.object(
                Vir_ChatBot,
                "build_graph",
                new_callable=lambda: lambda self: AsyncMock(
                    return_value=MagicMock()
                )(),
            ),
        ):
            await create_graph(
                global_retriever=MagicMock(),
                llm_model="override-model",
                temperature=0.9,
                max_retries=9,
                system_prompt="system-x",
            )
        gem.assert_called_once_with(
            model="override-model", temperature=0.9, max_retries=9
        )

    @pytest.mark.asyncio
    async def test_pragmas_skipped_when_conn_is_none(self, monkeypatch):
        monkeypatch.setattr(_, "LLM_PROVIDER", "gemini")
        checkpointer = MagicMock()
        checkpointer.conn = None
        ctx = self._ctx_mgr_with(checkpointer)

        with (
            patch(
                "agents.vir_chatbot.vir_chatbot.AsyncSqliteSaver"
                ".from_conn_string",
                return_value=ctx,
            ),
            patch("agents.vir_chatbot.vir_chatbot.ChatGoogleGenerativeAI"),
            patch.object(
                Vir_ChatBot,
                "build_graph",
                new_callable=lambda: lambda self: AsyncMock(
                    return_value=MagicMock()
                )(),
            ),
        ):
            graph, _cm = await create_graph(global_retriever=MagicMock())
        assert graph is not None

    @pytest.mark.asyncio
    async def test_pragmas_skipped_when_no_conn_attr(self, monkeypatch):
        """Covers ``hasattr(checkpointer, 'conn')`` returning False."""
        monkeypatch.setattr(_, "LLM_PROVIDER", "gemini")
        checkpointer = MagicMock(spec=[])  # no conn attribute
        ctx = self._ctx_mgr_with(checkpointer)

        with (
            patch(
                "agents.vir_chatbot.vir_chatbot.AsyncSqliteSaver"
                ".from_conn_string",
                return_value=ctx,
            ),
            patch("agents.vir_chatbot.vir_chatbot.ChatGoogleGenerativeAI"),
            patch.object(
                Vir_ChatBot,
                "build_graph",
                new_callable=lambda: lambda self: AsyncMock(
                    return_value=MagicMock()
                )(),
            ),
        ):
            graph, _cm = await create_graph(global_retriever=MagicMock())
        assert graph is not None
