"""
Config loader — reads pipeline_config.yaml and merges environment overrides.

Provides a singleton get_config() that returns the full config dict.
Provides safe defaults if the YAML file is missing.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Location of the config file relative to repo root
_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pipeline_config.yaml"

# Minimal defaults in case the YAML file is missing
_DEFAULTS: dict[str, Any] = {
    "ocr": {
        "languages": "hin+eng",
        "oem": 3,
        "psm": 6,
        "confidence_threshold": 30,
        "dpi": 300,
    },
    "preprocessing": {
        "enabled": True,
        "grayscale": True,
        "dpi_normalize": True,
        "noise_removal": True,
        "noise_kernel": 3,
        "deskew": True,
        "deskew_max_angle": 15.0,
        "adaptive_threshold": True,
        "threshold_block_size": 15,
        "threshold_c": 8,
    },
    "ner": {
        "use_finetuned": False,
        "finetuned_model_path": "models/ner",
        "fallback_model": "Davlan/xlm-roberta-base-ner-hrl",
        "confidence_threshold": 0.70,
        "max_chunk_length": 1000,
        "max_token_length": 256,
    },
    "classifier": {
        "use_finetuned": False,
        "finetuned_model_path": "models/classifier",
        "fallback_model": "facebook/bart-large-mnli",
        "max_token_length": 512,
        "categories": [
            "invoice", "receipt", "id_card", "legal_document",
            "medical_report", "bank_statement", "govt_form", "letter",
        ],
    },
    "pipeline": {
        "timeout_seconds": 120,
        "max_pages": 20,
        "max_file_size_mb": 50,
        "parallel_ner_classifier": True,
    },
    "training": {
        "ner": {
            "base_model": "xlm-roberta-base",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 10,
            "max_length": 256,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "fp16": True,
            "early_stopping_patience": 3,
            "metric_for_best_model": "overall_f1",
            "save_total_limit": 3,
            "output_dir": "models/ner",
            "logging_steps": 50,
            "eval_steps": 200,
            "save_steps": 200,
        },
        "classifier": {
            "base_model": "xlm-roberta-base",
            "learning_rate": 3e-5,
            "batch_size": 16,
            "epochs": 8,
            "max_length": 512,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "fp16": True,
            "early_stopping_patience": 3,
            "metric_for_best_model": "f1_macro",
            "save_total_limit": 3,
            "output_dir": "models/classifier",
            "logging_steps": 50,
            "eval_steps": 100,
            "save_steps": 100,
        },
    },
}


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and return the merged pipeline configuration.

    Priority (highest → lowest):
      1. Environment variable overrides (TESSERACT_CMD, POPPLER_PATH, etc.)
      2. pipeline_config.yaml
      3. Built-in defaults (_DEFAULTS)
    """
    config = _deep_merge({}, _DEFAULTS)

    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, yaml_config)
            logger.debug("Loaded config from %s", _CONFIG_PATH)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — using defaults", _CONFIG_PATH, exc)
    else:
        logger.warning("Config file not found at %s — using defaults", _CONFIG_PATH)

    # Apply environment overrides
    if os.getenv("NER_MODEL_PATH"):
        config["ner"]["finetuned_model_path"] = os.getenv("NER_MODEL_PATH")
    if os.getenv("CLASSIFIER_MODEL_PATH"):
        config["classifier"]["finetuned_model_path"] = os.getenv("CLASSIFIER_MODEL_PATH")

    return config


def reload_config() -> dict:
    """Force a config reload (clears the lru_cache)."""
    get_config.cache_clear()
    return get_config()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
