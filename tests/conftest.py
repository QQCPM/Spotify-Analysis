"""
Pytest configuration for research validation tests
"""

import pytest
import sys
from pathlib import Path

# Ensure src directory is in Python path for imports
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

@pytest.fixture(scope="session")
def research_mode():
    """Mark that we're in research testing mode"""
    return True

# Configure pytest for research use
pytest_plugins = []