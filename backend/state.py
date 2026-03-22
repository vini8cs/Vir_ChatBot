import json
import logging
import os
from pathlib import Path

import config as _
from backend.models import RuntimeConfig

global_resources: dict = {}
runtime_config: RuntimeConfig = RuntimeConfig()


def _load_persisted_config() -> RuntimeConfig:
    """Load runtime config from JSON file, falling back to defaults."""
    config_path = _.RUNTIME_CONFIG_PATH
    if not os.path.exists(config_path):
        return RuntimeConfig()
    try:
        with open(config_path) as f:
            data = json.load(f)
        return RuntimeConfig(**data)
    except Exception as e:
        logging.warning(
            f"Could not load persisted config, using defaults: {e}"
        )
        return RuntimeConfig()


def _save_persisted_config(config: RuntimeConfig) -> None:
    """Persist runtime config to JSON file."""
    config_path = _.RUNTIME_CONFIG_PATH
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
    except Exception as e:
        logging.warning(f"Could not persist config: {e}")
