"""
Spotify Data Collection Module

Handles authentication, data collection, and preprocessing from Spotify Web API.
Implements rate limiting, error handling, and data validation.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load environment variables once at module level
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional, fallback to system env vars

@dataclass
class SpotifyConfig:
    """Configuration for Spotify API client - loads from environment by default"""
    client_id: str = field(default_factory=lambda: os.getenv('SPOTIFY_CLIENT_ID', ''))
    client_secret: str = field(default_factory=lambda: os.getenv('SPOTIFY_CLIENT_SECRET', ''))
    redirect_uri: str = field(default_factory=lambda: os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8080/callback'))
    scope: str = "user-read-recently-played user-read-playback-state user-top-read"
    cache_path: str = ".spotify_cache"
    rate_limit_delay: float = 0.6  # Spotify allows 100 requests/minute
    
    def __post_init__(self):
        """Validate that required credentials are available"""
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify credentials not found. Please:\n"
                "1. Copy .env.example to .env\n"
                "2. Fill in your SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET\n"
                "3. Get credentials at: https://developer.spotify.com/dashboard/"
            )

class SpotifyDataCollector:
    """
    Collects and preprocesses music listening data from Spotify Web API.
    
    Handles rate limiting, data validation, and cultural categorization.
    Designed for cross-cultural music recommendation research.
    """
    
    def __init__(self, config: SpotifyConfig):
        self.config = config
        self.client = self._initialize_client()
        self.logger = self._setup_logging()
        
    def _initialize_client(self) -> spotipy.Spotify:
        """Initialize Spotify API client with OAuth authentication"""
        auth_manager = SpotifyOAuth(
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            redirect_uri=self.config.redirect_uri,
            scope=self.config.scope,
            cache_path=self.config.cache_path
        )
        return spotipy.Spotify(auth_manager=auth_manager)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for data collection activities"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _rate_limit_wait(self):
        """Implement rate limiting to respect Spotify API limits"""
        time.sleep(self.config.rate_limit_delay)
    
    def collect_user_listening_history(
        self, 
        limit: int = 50,
        max_requests: int = 100
    ) -> pd.DataFrame:
        """
        Collect user's recent listening history from Spotify.
        
        Args:
            limit: Number of tracks per request (max 50)
            max_requests: Maximum number of API requests
            
        Returns:
            DataFrame with listening history including timestamps and track metadata
        """
        self.logger.info(f"Starting data collection (limit={limit}, max_requests={max_requests})")
        
        all_tracks = []
        before = None
        
        for request_num in tqdm(range(max_requests), desc="Collecting listening data"):
            try:
                self._rate_limit_wait()
                
                # Get recently played tracks
                results = self.client.current_user_recently_played(
                    limit=limit,
                    before=before
                )
                
                if not results['items']:
                    self.logger.info("No more tracks available")
                    break
                
                # Process each track
                for item in results['items']:
                    track_data = self._extract_track_features(item)
                    all_tracks.append(track_data)
                
                # Update cursor for pagination
                before = results['cursors'].get('before') if results.get('cursors') else None
                
                self.logger.info(f"Request {request_num + 1}: Collected {len(results['items'])} tracks")
                
            except Exception as e:
                self.logger.error(f"Error in request {request_num + 1}: {str(e)}")
                continue
        
        df = pd.DataFrame(all_tracks)
        self.logger.info(f"Collection complete: {len(df)} total tracks")
        
        return df
    
    def _extract_track_features(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from a Spotify track item"""
        track = item['track']
        
        return {
            'played_at': item['played_at'],
            'track_id': track['id'],
            'track_name': track['name'],
            'artist_id': track['artists'][0]['id'],
            'artist_name': track['artists'][0]['name'],
            'album_id': track['album']['id'],
            'album_name': track['album']['name'],
            'duration_ms': track['duration_ms'],
            'popularity': track['popularity'],
            'explicit': track['explicit'],
            'preview_url': track.get('preview_url'),
            'release_date': track['album'].get('release_date'),
            'markets': track.get('available_markets', []),
            'external_urls': track.get('external_urls', {})
        }
    
    def collect_audio_features(self, track_ids: List[str]) -> pd.DataFrame:
        """
        Collect audio features for a list of tracks.
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            DataFrame with audio features for each track
        """
        self.logger.info(f"Collecting audio features for {len(track_ids)} tracks")
        
        # Spotify allows up to 100 IDs per request
        batch_size = 100
        all_features = []
        
        for i in tqdm(range(0, len(track_ids), batch_size), desc="Collecting audio features"):
            batch = track_ids[i:i + batch_size]
            
            try:
                self._rate_limit_wait()
                features = self.client.audio_features(batch)
                
                # Filter out None results (tracks without features)
                valid_features = [f for f in features if f is not None]
                all_features.extend(valid_features)
                
            except Exception as e:
                self.logger.error(f"Error collecting audio features for batch {i//batch_size + 1}: {str(e)}")
                continue
        
        df = pd.DataFrame(all_features)
        self.logger.info(f"Audio features collected for {len(df)} tracks")
        
        return df
    
    def collect_artist_information(self, artist_ids: List[str]) -> pd.DataFrame:
        """
        Collect detailed artist information including genres and popularity.
        
        Args:
            artist_ids: List of Spotify artist IDs
            
        Returns:
            DataFrame with artist metadata
        """
        self.logger.info(f"Collecting artist information for {len(artist_ids)} artists")
        
        # Remove duplicates
        unique_artist_ids = list(set(artist_ids))
        batch_size = 50  # Spotify allows up to 50 artist IDs per request
        all_artists = []
        
        for i in tqdm(range(0, len(unique_artist_ids), batch_size), desc="Collecting artist data"):
            batch = unique_artist_ids[i:i + batch_size]
            
            try:
                self._rate_limit_wait()
                artists = self.client.artists(batch)
                
                for artist in artists['artists']:
                    artist_data = {
                        'artist_id': artist['id'],
                        'artist_name': artist['name'],
                        'genres': artist['genres'],
                        'popularity': artist['popularity'],
                        'followers': artist['followers']['total'],
                        'external_urls': artist.get('external_urls', {})
                    }
                    all_artists.append(artist_data)
                    
            except Exception as e:
                self.logger.error(f"Error collecting artist info for batch {i//batch_size + 1}: {str(e)}")
                continue
        
        df = pd.DataFrame(all_artists)
        self.logger.info(f"Artist information collected for {len(df)} artists")
        
        return df
    
    def save_data(
        self, 
        listening_history: pd.DataFrame, 
        audio_features: pd.DataFrame, 
        artist_info: pd.DataFrame,
        output_dir: str = "data/raw"
    ) -> None:
        """Save collected data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        listening_history.to_csv(
            os.path.join(output_dir, f"listening_history_{timestamp}.csv"),
            index=False
        )
        
        audio_features.to_csv(
            os.path.join(output_dir, f"audio_features_{timestamp}.csv"),
            index=False
        )
        
        artist_info.to_csv(
            os.path.join(output_dir, f"artist_info_{timestamp}.csv"),
            index=False
        )
        
        self.logger.info(f"Data saved to {output_dir}")


def collect_spotify_data(
    client_id: str,
    client_secret: str,
    max_tracks: int = 5000,
    output_dir: str = "data/raw"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    High-level function to collect comprehensive Spotify data.
    
    Args:
        client_id: Spotify API client ID
        client_secret: Spotify API client secret
        max_tracks: Maximum number of tracks to collect
        output_dir: Directory to save collected data
        
    Returns:
        Tuple of (listening_history, audio_features, artist_info) DataFrames
    """
    config = SpotifyConfig(
        client_id=client_id,
        client_secret=client_secret
    )
    
    collector = SpotifyDataCollector(config)
    
    # Calculate number of requests needed
    tracks_per_request = 50
    max_requests = min(100, (max_tracks + tracks_per_request - 1) // tracks_per_request)
    
    # Collect listening history
    listening_history = collector.collect_user_listening_history(
        limit=tracks_per_request,
        max_requests=max_requests
    )
    
    if listening_history.empty:
        raise ValueError("No listening history collected. Check Spotify API credentials.")
    
    # Collect audio features
    track_ids = listening_history['track_id'].unique().tolist()
    audio_features = collector.collect_audio_features(track_ids)
    
    # Collect artist information
    artist_ids = listening_history['artist_id'].unique().tolist()
    artist_info = collector.collect_artist_information(artist_ids)
    
    # Save data
    collector.save_data(listening_history, audio_features, artist_info, output_dir)
    
    return listening_history, audio_features, artist_info


if __name__ == "__main__":
    # Example usage - now uses secure environment loading by default
    try:
        config = SpotifyConfig()  # Automatically loads from .env
        collector = SpotifyDataCollector(config)
        
        # Collect sample data for research
        listening_history, audio_features, artist_info = collect_spotify_data(
            client_id=config.client_id,
            client_secret=config.client_secret,
            max_tracks=1000
        )
        
        print(f"✅ Collected {len(listening_history)} listening events")
        print(f"✅ Audio features for {len(audio_features)} tracks") 
        print(f"✅ Information for {len(artist_info)} artists")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
    except Exception as e:
        print(f"❌ Collection Error: {e}")
        print("Check your Spotify API credentials and internet connection.")