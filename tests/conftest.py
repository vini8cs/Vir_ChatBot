"""Shared pytest configuration and fixtures.

Environment variables and external-client stubs are applied at
module level so they take effect before any project code is imported.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-test-key")
os.environ.setdefault("GCP_CREDENTIALS", "/dev/null")
os.environ.setdefault("GCP_PROJECT", "fake-test-project")
os.environ.setdefault("GCP_REGION", "us-central1")

# Stub google-genai so config.py's `genai.Client()` never hits the network.
_mock_genai = MagicMock()
sys.modules["google.genai"] = _mock_genai
sys.modules["google.genai.types"] = MagicMock()
if "google" in sys.modules:
    sys.modules["google"].genai = _mock_genai

# Stub LangChain Google packages to avoid deep google-sdk import chains.
# llms/gemini.py imports these at module level; we never need the real
# classes in unit tests (they are only instantiated, not type-checked).
sys.modules["langchain_google_genai"] = MagicMock()
sys.modules["langchain_google_vertexai"] = MagicMock()


@pytest.fixture()
def tmp_config_path(tmp_path, monkeypatch):
    """Point RUNTIME_CONFIG_PATH at a temp file so tests are isolated."""
    import config as _

    path = str(tmp_path / "runtime_config.json")
    monkeypatch.setattr(_, "RUNTIME_CONFIG_PATH", path)
    return path
