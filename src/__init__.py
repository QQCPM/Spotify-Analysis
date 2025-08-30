"""
Cross-Cultural Music Recommendation Research System

A research-quality music recommendation system analyzing cross-cultural 
preferences (Vietnamese + Western music) with temporal dynamics.

This package provides the core functionality for:
- Data collection from Spotify API
- Cross-cultural music analysis  
- Temporal preference modeling
- Matrix factorization and neural recommenders
- Statistical validation and research analysis

Author: Research Team
Version: 0.1.0
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Core module imports for easy access
from . import data_processing
from . import models 
from . import analysis
from . import evaluation

__all__ = [
    "data_processing",
    "models",
    "analysis", 
    "evaluation",
    "__version__"
]