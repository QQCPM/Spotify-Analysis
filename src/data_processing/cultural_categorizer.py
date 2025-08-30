"""
Cultural Categorization Module

Categorizes music tracks as Vietnamese, Western, or Bridge songs based on 
artist information, linguistic analysis, and market data.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


@dataclass 
class CulturalCategories:
    """Data class for cultural classification results"""
    vietnamese: float
    western: float
    bridge: float
    dominant_culture: str
    confidence: float


class CulturalCategorizer:
    """
    Categorizes music tracks into Vietnamese, Western, or Bridge categories.
    
    Uses multiple signals:
    - Artist names (linguistic patterns)
    - Genres (cultural markers)
    - Market availability
    - Language detection in metadata
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        
        # Vietnamese language patterns
        self.vietnamese_patterns = self._compile_vietnamese_patterns()
        
        # Vietnamese music genres
        self.vietnamese_genres = {
            'vietnamese pop', 'vpop', 'v-pop', 'vietnamese rock', 'vietnamese folk',
            'vietnamese traditional', 'ca tru', 'quan ho', 'cai luong', 'cheo',
            'vietnamese indie', 'vietnamese r&b', 'vietnamese rap', 'vietnamese hip hop'
        }
        
        # Western genres 
        self.western_genres = {
            'pop', 'rock', 'hip hop', 'rap', 'country', 'folk', 'blues', 'jazz',
            'classical', 'electronic', 'dance', 'house', 'techno', 'indie',
            'alternative', 'punk', 'metal', 'r&b', 'soul', 'reggae'
        }
        
        # Bridge genres (cross-cultural appeal)
        self.bridge_genres = {
            'world music', 'fusion', 'world fusion', 'asian pop', 'k-pop',
            'mandopop', 'cantopop', 'international', 'crossover'
        }
        
    def _default_config(self) -> Dict:
        """Default configuration for cultural categorization"""
        return {
            'vietnamese_threshold': 0.7,
            'western_threshold': 0.7,
            'bridge_threshold': 0.5,
            'confidence_threshold': 0.6,
            'market_weight': 0.3,
            'genre_weight': 0.4,
            'linguistic_weight': 0.3
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for categorization activities"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def _compile_vietnamese_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for Vietnamese name detection"""
        # Vietnamese names often have specific patterns
        patterns = [
            # Vietnamese diacritics
            re.compile(r'[àáạảãâầấậẩẫăằắặẳẵ]', re.IGNORECASE),
            re.compile(r'[èéẹẻẽêềếệểễ]', re.IGNORECASE),
            re.compile(r'[ìíịỉĩ]', re.IGNORECASE),
            re.compile(r'[òóọỏõôồốộổỗơờớợởỡ]', re.IGNORECASE),
            re.compile(r'[ùúụủũưừứựửữ]', re.IGNORECASE),
            re.compile(r'[ỳýỵỷỹ]', re.IGNORECASE),
            re.compile(r'đ', re.IGNORECASE),
            
            # Common Vietnamese name patterns
            re.compile(r'\b(nguyen|tran|le|pham|hoang|phan|vu|dang|bui|do|ho|ngo|duong|ly)\b', re.IGNORECASE),
            re.compile(r'\b(son tung|my tam|dam vinh hung|ho ngoc ha|noo phuoc thinh)\b', re.IGNORECASE),
        ]
        return patterns
        
    def detect_vietnamese_linguistic_features(self, text: str) -> float:
        """
        Detect Vietnamese linguistic features in text.
        
        Args:
            text: Text to analyze (artist name, song title, etc.)
            
        Returns:
            Score between 0 and 1 indicating Vietnamese likelihood
        """
        if not text or not isinstance(text, str):
            return 0.0
            
        text_lower = text.lower()
        matches = 0
        
        for pattern in self.vietnamese_patterns:
            if pattern.search(text_lower):
                matches += 1
                
        # Normalize by number of patterns
        return min(matches / len(self.vietnamese_patterns), 1.0)
        
    def analyze_genres(self, genres: List[str]) -> Tuple[float, float, float]:
        """
        Analyze genres to determine cultural classification.
        
        Args:
            genres: List of genre strings
            
        Returns:
            Tuple of (vietnamese_score, western_score, bridge_score)
        """
        if not genres:
            return 0.0, 0.0, 0.0
            
        genres_lower = [g.lower() for g in genres]
        
        vietnamese_matches = sum(1 for g in genres_lower 
                               if any(vg in g for vg in self.vietnamese_genres))
        
        western_matches = sum(1 for g in genres_lower 
                            if any(wg in g for wg in self.western_genres))
        
        bridge_matches = sum(1 for g in genres_lower 
                           if any(bg in g for bg in self.bridge_genres))
        
        total = len(genres)
        
        return (
            vietnamese_matches / total,
            western_matches / total,
            bridge_matches / total
        )
        
    def analyze_market_availability(self, markets: List[str]) -> Tuple[float, float]:
        """
        Analyze market availability to infer cultural origin.
        
        Args:
            markets: List of market codes where track is available
            
        Returns:
            Tuple of (vietnamese_score, western_score)
        """
        if not markets:
            return 0.0, 0.0
            
        markets_set = set(markets)
        
        # Vietnamese market indicators
        vietnamese_markets = {'VN'}  # Vietnam
        asian_markets = {'VN', 'SG', 'MY', 'TH', 'ID', 'PH'}  # Southeast Asian markets
        
        # Western market indicators
        western_markets = {'US', 'GB', 'CA', 'AU', 'DE', 'FR', 'IT', 'ES', 'NL', 'SE'}
        
        vietnamese_score = 0.0
        if 'VN' in markets_set:
            vietnamese_score += 0.8
        if len(markets_set.intersection(asian_markets)) >= 3:
            vietnamese_score += 0.3
            
        western_score = len(markets_set.intersection(western_markets)) / len(western_markets)
        
        return vietnamese_score, western_score
        
    def categorize_track(
        self,
        artist_name: str,
        track_name: str,
        genres: List[str],
        markets: List[str]
    ) -> CulturalCategories:
        """
        Categorize a single track into cultural categories.
        
        Args:
            artist_name: Name of the artist
            track_name: Name of the track
            genres: List of genres
            markets: List of available markets
            
        Returns:
            CulturalCategories with classification results
        """
        # Linguistic analysis
        artist_vietnamese = self.detect_vietnamese_linguistic_features(artist_name)
        track_vietnamese = self.detect_vietnamese_linguistic_features(track_name)
        linguistic_score = (artist_vietnamese + track_vietnamese) / 2
        
        # Genre analysis
        genre_vietnamese, genre_western, genre_bridge = self.analyze_genres(genres)
        
        # Market analysis
        market_vietnamese, market_western = self.analyze_market_availability(markets)
        
        # Weighted combination
        vietnamese_score = (
            self.config['linguistic_weight'] * linguistic_score +
            self.config['genre_weight'] * genre_vietnamese +
            self.config['market_weight'] * market_vietnamese
        )
        
        western_score = (
            self.config['genre_weight'] * genre_western +
            self.config['market_weight'] * market_western
        )
        
        # Bridge score considers cross-cultural elements
        bridge_score = (
            self.config['genre_weight'] * genre_bridge +
            0.5 * min(vietnamese_score, western_score)  # Balanced cultural elements
        )
        
        # Normalize scores
        total_score = vietnamese_score + western_score + bridge_score
        if total_score > 0:
            vietnamese_score /= total_score
            western_score /= total_score
            bridge_score /= total_score
            
        # Determine dominant culture
        scores = {
            'vietnamese': vietnamese_score,
            'western': western_score, 
            'bridge': bridge_score
        }
        
        dominant_culture = max(scores, key=scores.get)
        confidence = scores[dominant_culture]
        
        return CulturalCategories(
            vietnamese=vietnamese_score,
            western=western_score,
            bridge=bridge_score,
            dominant_culture=dominant_culture,
            confidence=confidence
        )
        
    def categorize_dataset(
        self,
        listening_data: pd.DataFrame,
        artist_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Categorize an entire dataset of tracks.
        
        Args:
            listening_data: DataFrame with track and artist information
            artist_data: DataFrame with detailed artist information
            
        Returns:
            DataFrame with cultural categorization columns added
        """
        self.logger.info(f"Categorizing {len(listening_data)} tracks")
        
        # Merge with artist data to get genres
        merged_data = listening_data.merge(
            artist_data[['artist_id', 'genres']], 
            on='artist_id', 
            how='left'
        )
        
        # Initialize result columns
        results = []
        
        for _, row in merged_data.iterrows():
            # Handle missing data
            genres = row.get('genres', [])
            if isinstance(genres, str):
                genres = eval(genres) if genres.startswith('[') else [genres]
            elif pd.isna(genres):
                genres = []
                
            markets = row.get('markets', [])
            if isinstance(markets, str):
                markets = eval(markets) if markets.startswith('[') else [markets]
            elif pd.isna(markets):
                markets = []
                
            # Categorize track
            category = self.categorize_track(
                artist_name=row.get('artist_name', ''),
                track_name=row.get('track_name', ''),
                genres=genres,
                markets=markets
            )
            
            results.append({
                'vietnamese_score': category.vietnamese,
                'western_score': category.western,
                'bridge_score': category.bridge,
                'dominant_culture': category.dominant_culture,
                'cultural_confidence': category.confidence
            })
            
        # Add results to original dataframe
        result_df = listening_data.copy()
        for i, result in enumerate(results):
            for key, value in result.items():
                result_df.loc[i, key] = value
                
        # Add categorical labels
        result_df['is_vietnamese'] = (
            result_df['dominant_culture'] == 'vietnamese'
        ) & (result_df['cultural_confidence'] >= self.config['confidence_threshold'])
        
        result_df['is_western'] = (
            result_df['dominant_culture'] == 'western'
        ) & (result_df['cultural_confidence'] >= self.config['confidence_threshold'])
        
        result_df['is_bridge'] = (
            result_df['dominant_culture'] == 'bridge'
        ) & (result_df['cultural_confidence'] >= self.config['confidence_threshold'])
        
        self.logger.info(f"Categorization complete: "
                        f"{result_df['is_vietnamese'].sum()} Vietnamese, "
                        f"{result_df['is_western'].sum()} Western, "
                        f"{result_df['is_bridge'].sum()} Bridge songs")
        
        return result_df
        
    def generate_cultural_statistics(self, categorized_data: pd.DataFrame) -> Dict:
        """Generate comprehensive cultural statistics"""
        stats = {
            'total_tracks': len(categorized_data),
            'vietnamese_tracks': categorized_data['is_vietnamese'].sum(),
            'western_tracks': categorized_data['is_western'].sum(),
            'bridge_tracks': categorized_data['is_bridge'].sum(),
            'uncategorized_tracks': len(categorized_data) - (
                categorized_data['is_vietnamese'].sum() +
                categorized_data['is_western'].sum() +
                categorized_data['is_bridge'].sum()
            )
        }
        
        # Percentages
        total = stats['total_tracks']
        if total > 0:
            stats['vietnamese_percentage'] = stats['vietnamese_tracks'] / total * 100
            stats['western_percentage'] = stats['western_tracks'] / total * 100
            stats['bridge_percentage'] = stats['bridge_tracks'] / total * 100
            
        # Confidence statistics
        stats['avg_confidence'] = categorized_data['cultural_confidence'].mean()
        stats['confidence_std'] = categorized_data['cultural_confidence'].std()
        
        return stats


def categorize_cultural_data(
    listening_data: pd.DataFrame,
    artist_data: pd.DataFrame,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    High-level function to categorize cultural data.
    
    Args:
        listening_data: DataFrame with listening history
        artist_data: DataFrame with artist information
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (categorized_data, statistics)
    """
    categorizer = CulturalCategorizer(config)
    
    categorized_data = categorizer.categorize_dataset(listening_data, artist_data)
    statistics = categorizer.generate_cultural_statistics(categorized_data)
    
    return categorized_data, statistics