"""
Common pytest fixtures and configuration.
"""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest


@pytest.fixture(autouse=True)
def mock_filesystem_operations():
    """
    Mock filesystem operations to prevent actual file system modifications.
    """
    # Mock Path's home method
    with patch("pathlib.Path.home", return_value=Path("/mock/home")):
        # Mock directory creation
        with patch("pathlib.Path.mkdir"):
            # Mock file existence check (default to True so we don't try to create files)
            with patch("pathlib.Path.exists", return_value=True):
                # Mock file opening operations with a mock that works for both read and write
                mock_json_content = json.dumps({"models": {}})
                with patch("builtins.open", mock_open(read_data=mock_json_content)):
                    yield


@pytest.fixture(autouse=True)
def disable_lru_cache():
    """
    Clear LRU caches between tests to prevent test interference.
    """
    from llm_registry.utils import load_user_models

    load_user_models.cache_clear()
    yield
    load_user_models.cache_clear()
