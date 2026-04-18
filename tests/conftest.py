"""Shared pytest configuration and fixtures.

Environment variables and external-client stubs are applied at
module level so they take effect before any project code is imported.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-gemini-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-anthropic-key")
os.environ.setdefault("LLM_PROVIDER", "gemini")

# Stub LangChain provider packages to avoid deep sdk import chains.
# These are imported at module level; we never need the real classes in
# unit tests (they are only instantiated, not type-checked).
sys.modules["langchain_google_genai"] = MagicMock()
sys.modules["langchain_anthropic"] = MagicMock()


@pytest.fixture()
def tmp_config_path(tmp_path, monkeypatch):
    """Point RUNTIME_CONFIG_PATH at a temp file so tests are isolated."""
    import config as _

    path = str(tmp_path / "runtime_config.json")
    monkeypatch.setattr(_, "RUNTIME_CONFIG_PATH", path)
    return path
