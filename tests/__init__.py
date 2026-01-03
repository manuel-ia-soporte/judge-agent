# tests/__init__.py
"""
Finance Judge System Test Suite
"""
import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_CONFIG = {
    "test_timeout": 30,
    "max_concurrent_tests": 10,
    "log_level": "INFO",
    "use_real_sec": False,  # Use mocked SEC client in tests
}

# Test categories
TEST_CATEGORIES = {
    "unit": "Unit tests",
    "integration": "Integration tests",
    "performance": "Performance tests",
    "contract": "Contract tests",
    "security": "Security tests",
    "e2e": "End-to-end tests"
}

def pytest_configure(config):
    """Configure pytest"""
    config.option.asyncio_mode = "auto"
    config.option.timeout = TEST_CONFIG["test_timeout"]

def get_test_data_path() -> Path:
    """Get path to test data directory"""
    test_dir = Path(__file__).parent
    data_dir = test_dir / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_fixture_path(fixture_name: str) -> Path:
    """Get path to test fixture"""
    return get_test_data_path() / fixture_name

# Create test data directory if it doesn't exist
test_data_dir = get_test_data_path()
(test_data_dir / "sec_filings").mkdir(exist_ok=True, parents=True)
(test_data_dir / "evaluations").mkdir(exist_ok=True, parents=True)
(test_data_dir / "agents").mkdir(exist_ok=True, parents=True)