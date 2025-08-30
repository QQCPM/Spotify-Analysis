"""Data Processing Module

Contains classes for:
- Spotify data collection and API management
- Music data preprocessing and cleaning
- Feature engineering (temporal, cultural, audio)
- User-item matrix construction
"""

from .spotify_collector import SpotifyDataCollector
from .feature_engineer import FeatureEngineer
from .cultural_categorizer import CulturalCategorizer
from .extended_spotify_processor import ExtendedSpotifyProcessor

__all__ = [
    "SpotifyDataCollector",
    "FeatureEngineer",
    "CulturalCategorizer",
    "ExtendedSpotifyProcessor"
]