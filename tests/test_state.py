"""Tests for config persistence helpers in backend/state.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.models import RuntimeConfig
from backend.state import _load_persisted_config, _save_persisted_config


class TestSavePersistedConfig:
    def test_creates_json_file(self, tmp_config_path):
        _save_persisted_config(RuntimeConfig())
        assert Path(tmp_config_path).exists()

    def test_written_data_is_valid_json(self, tmp_config_path):
        cfg = RuntimeConfig(temperature=0.42)
        _save_persisted_config(cfg)
        with open(tmp_config_path) as f:
            data = json.load(f)
        assert data["temperature"] == pytest.approx(0.42)

    def test_all_fields_are_persisted(self, tmp_config_path):
        cfg = RuntimeConfig(max_retries=7, summarize=True)
        _save_persisted_config(cfg)
        with open(tmp_config_path) as f:
            data = json.load(f)
        assert data["max_retries"] == 7
        assert data["summarize"] is True

    def test_creates_parent_directories(self, tmp_path, monkeypatch):
        import config as _

        deep = str(tmp_path / "nested" / "dir" / "config.json")
        monkeypatch.setattr(_, "RUNTIME_CONFIG_PATH", deep)
        _save_persisted_config(RuntimeConfig())
        assert Path(deep).exists()

    def test_overwrites_existing_file(self, tmp_config_path):
        _save_persisted_config(RuntimeConfig(temperature=0.1))
        _save_persisted_config(RuntimeConfig(temperature=0.9))
        with open(tmp_config_path) as f:
            data = json.load(f)
        assert data["temperature"] == pytest.approx(0.9)

    def test_logs_warning_when_write_raises(
        self, tmp_config_path, monkeypatch
    ):
        """Covers the except branch in _save_persisted_config."""
        import backend.state as state
        import config as _

        monkeypatch.setattr(_, "RUNTIME_CONFIG_PATH", tmp_config_path)
        with patch("builtins.open", side_effect=OSError("disk full")):
            state._save_persisted_config(RuntimeConfig())


class TestLoadPersistedConfig:
    def test_returns_defaults_when_file_does_not_exist(self, tmp_config_path):
        cfg = _load_persisted_config()
        assert isinstance(cfg, RuntimeConfig)
        assert cfg == RuntimeConfig()

    def test_loads_previously_saved_values(self, tmp_config_path):
        original = RuntimeConfig(temperature=0.77, max_retries=10)
        _save_persisted_config(original)
        loaded = _load_persisted_config()
        assert loaded.temperature == pytest.approx(0.77)
        assert loaded.max_retries == 10

    def test_returns_defaults_on_corrupt_json(self, tmp_config_path):
        Path(tmp_config_path).write_text("not-valid{{{json")
        cfg = _load_persisted_config()
        assert isinstance(cfg, RuntimeConfig)
        assert cfg == RuntimeConfig()

    def test_roundtrip_preserves_all_fields(self, tmp_config_path):
        original = RuntimeConfig(
            temperature=0.5,
            max_retries=2,
            summarize=True,
            retriever_limit=10,
        )
        _save_persisted_config(original)
        loaded = _load_persisted_config()
        assert loaded == original
