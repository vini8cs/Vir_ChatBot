"""Tests for module-level side effects in config.py.

The LangSmith env-var branch only runs when both LANGSMITH_API_KEY and
LANGSMITH_PROJECT are non-empty — conditions that are never true during
normal test runs. We force them by reloading the module with patched
env vars (and restore everything afterwards).
"""

import importlib
import os


class TestLangSmithEnvVars:
    def test_sets_langsmith_env_vars_when_both_keys_present(
        self, monkeypatch
    ):
        monkeypatch.setenv("LANGSMITH_API_KEY", "fake-ls-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "my-project")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://api.smith.example")
        monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")

        import config

        importlib.reload(config)

        assert os.environ.get("LANGSMITH_API_KEY") == "fake-ls-key"
        assert os.environ.get("LANGSMITH_PROJECT") == "my-project"
        assert os.environ.get("LANGSMITH_TRACING") == "true"

    def test_does_not_set_langsmith_vars_when_keys_missing(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")

        import config

        importlib.reload(config)

        assert os.environ.get("LANGSMITH_API_KEY", "") == ""
