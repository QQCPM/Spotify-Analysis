"""
Extended Spotify Data Processor

Processes Spotify Extended Streaming History JSON files for cross-cultural music research.
Handles the rich temporal, geographical, and behavioral data from Spotify's extended export.
"""

import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
import pytz

warnings.filterwarnings('ignore')


@dataclass
class StreamingRecord:
    """Individual streaming record from Spotify Extended History"""
    timestamp: datetime
    platform: str
    ms_played: int
    conn_country: str
    ip_addr: str
    track_name: Optional[str]
    artist_name: Optional[str]
    album_name: Optional[str]
    spotify_track_uri: Optional[str]
    episode_name: Optional[str]
    episode_show_name: Optional[str]
    spotify_episode_uri: Optional[str]
    audio_features: Optional[Dict[str, float]]
    reason_start: Optional[str]
    reason_end: Optional[str]
    shuffle: Optional[bool]
    skipped: Optional[bool]
    offline: Optional[bool]
    offline_timestamp: Optional[int]
    incognito_mode: Optional[bool]


@dataclass
class ProcessingStats:
    """Statistics from data processing"""
    total_records: int
    audio_records: int
    video_records: int
    unique_tracks: int
    unique_artists: int
    date_range: Tuple[datetime, datetime]
    countries: List[str]
    platforms: List[str]
    total_listening_time_hours: float
    avg_track_completion_rate: float


class ExtendedSpotifyProcessor:
    """
    Processes Spotify Extended Streaming History for research analysis.
    
    Handles rich temporal data including:
    - Complete listening history with exact timestamps
    - Platform and location information
    - Detailed playback behavior (skips, shuffles, offline)
    - Audio vs video content separation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        self.processing_stats = None
        
    def _default_config(self) -> Dict:
        """Default configuration for extended Spotify processing"""
        return {
            'data_quality': {
                'min_ms_played': 30000,  # Minimum 30 seconds for valid listen
                'max_ms_played': 1800000,  # Maximum 30 minutes (catch data errors)
                'exclude_podcasts': True,  # Focus on music only
                'exclude_very_short_tracks': True  # Exclude <30 second tracks
            },
            'temporal_processing': {
                'timezone': 'UTC',  # Convert all times to UTC
                'session_gap_minutes': 30,  # Gap to define listening sessions
                'day_boundary_hour': 4  # 4 AM as start of new "listening day"
            },
            'cultural_detection': {
                'vietnam_ip_prefixes': ['14.', '27.', '42.', '43.', '45.', '49.', '58.', '59.', '60.', '61.', '62.', '103.', '113.', '115.', '116.', '117.', '118.', '119.', '120.', '121.', '123.', '124.', '125.', '171.', '210.', '222.'],
                'vietnam_country_codes': ['VN', 'VIETNAM'],
                'western_country_codes': ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'IT', 'ES', 'NL', 'SE', 'NO', 'DK']
            },
            'feature_extraction': {
                'calculate_completion_rates': True,
                'detect_skip_patterns': True,  
                'identify_repeated_listens': True,
                'track_platform_preferences': True,
                'analyze_listening_contexts': True
            }
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for extended Spotify processing"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def process_extended_history(
        self, 
        data_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[pd.DataFrame, ProcessingStats]:
        """
        Process complete Spotify Extended Streaming History.
        
        Args:
            data_path: Path to directory containing JSON files
            output_path: Optional path to save processed data
            
        Returns:
            Tuple of (processed_dataframe, processing_statistics)
        """
        self.logger.info("Starting Extended Spotify History processing")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # Find all streaming history files
        json_files = list(data_path.glob("Streaming_History_Audio_*.json"))
        if not json_files:
            raise ValueError(f"No Spotify streaming history files found in {data_path}")
        
        self.logger.info(f"Found {len(json_files)} streaming history files")
        
        # Process each file
        all_records = []
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            records = self._process_json_file(json_file)
            all_records.extend(records)
            self.logger.info(f"Processed {json_file.name}: {len(records)} records")
        
        # Convert to DataFrame
        df = self._create_dataframe(all_records)
        
        # Apply data quality filters
        df = self._apply_quality_filters(df)
        
        # Generate processing statistics
        self.processing_stats = self._generate_processing_stats(df)
        
        # Save processed data if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            self.logger.info(f"Saved processed data to {output_path}")
            
            # Save processing statistics
            stats_path = output_path.parent / f"{output_path.stem}_stats.json"
            self._save_processing_stats(stats_path)
        
        self.logger.info(f"Processing complete: {len(df)} valid records from {len(all_records)} total")
        
        return df, self.processing_stats

    def _process_json_file(self, json_file: Path) -> List[StreamingRecord]:
        """Process individual JSON streaming history file"""
        
        records = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                record = self._parse_streaming_record(item)
                if record:
                    records.append(record)
                    
        except Exception as e:
            self.logger.error(f"Error processing {json_file}: {str(e)}")
            
        return records

    def _parse_streaming_record(self, item: Dict[str, Any]) -> Optional[StreamingRecord]:
        """Parse individual streaming record from JSON"""
        
        try:
            # Parse timestamp
            timestamp = datetime.fromisoformat(item['ts'].replace('Z', '+00:00'))
            
            # Extract basic info
            ms_played = item.get('ms_played', 0)
            platform = item.get('platform', 'Unknown')
            conn_country = item.get('conn_country', 'Unknown')
            ip_addr = item.get('ip_addr', '')
            
            # Track information
            track_name = item.get('master_metadata_track_name')
            artist_name = item.get('master_metadata_album_artist_name')
            album_name = item.get('master_metadata_album_album_name')
            spotify_track_uri = item.get('spotify_track_uri')
            
            # Episode information (podcasts)
            episode_name = item.get('episode_name')
            episode_show_name = item.get('episode_show_name')
            spotify_episode_uri = item.get('spotify_episode_uri')
            
            # Playback behavior
            reason_start = item.get('reason_start')
            reason_end = item.get('reason_end')
            shuffle = item.get('shuffle')
            skipped = item.get('skipped')
            offline = item.get('offline')
            offline_timestamp = item.get('offline_timestamp')
            incognito_mode = item.get('incognito_mode')
            
            # Audio features (if available in extended data)
            audio_features = None
            if 'audio_features' in item:
                audio_features = item['audio_features']
            
            return StreamingRecord(
                timestamp=timestamp,
                platform=platform,
                ms_played=ms_played,
                conn_country=conn_country,
                ip_addr=ip_addr,
                track_name=track_name,
                artist_name=artist_name,
                album_name=album_name,
                spotify_track_uri=spotify_track_uri,
                episode_name=episode_name,
                episode_show_name=episode_show_name,
                spotify_episode_uri=spotify_episode_uri,
                audio_features=audio_features,
                reason_start=reason_start,
                reason_end=reason_end,
                shuffle=shuffle,
                skipped=skipped,
                offline=offline,
                offline_timestamp=offline_timestamp,
                incognito_mode=incognito_mode
            )
            
        except Exception as e:
            self.logger.warning(f"Error parsing record: {str(e)}")
            return None

    def _create_dataframe(self, records: List[StreamingRecord]) -> pd.DataFrame:
        """Convert streaming records to pandas DataFrame"""
        
        data = []
        
        for record in records:
            row = {
                'played_at': record.timestamp,
                'platform': record.platform,
                'ms_played': record.ms_played,
                'conn_country': record.conn_country,
                'ip_addr': record.ip_addr,
                'track_name': record.track_name,
                'artist_name': record.artist_name,
                'album_name': record.album_name,
                'spotify_track_uri': record.spotify_track_uri,
                'episode_name': record.episode_name,
                'episode_show_name': record.episode_show_name,
                'spotify_episode_uri': record.spotify_episode_uri,
                'reason_start': record.reason_start,
                'reason_end': record.reason_end,
                'shuffle': record.shuffle,
                'skipped': record.skipped,
                'offline': record.offline,
                'offline_timestamp': record.offline_timestamp,
                'incognito_mode': record.incognito_mode
            }
            
            # Add audio features if available
            if record.audio_features:
                for key, value in record.audio_features.items():
                    row[f'audio_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Basic data type corrections
        if len(df) > 0:
            df['played_at'] = pd.to_datetime(df['played_at'])
            df['ms_played'] = pd.to_numeric(df['ms_played'], errors='coerce')
            
            # Create derived columns
            df = self._create_derived_columns(df)
        
        return df

    def _create_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived columns for analysis"""
        
        # Listening duration in minutes
        df['minutes_played'] = df['ms_played'] / (1000 * 60)
        
        # Date components
        df['date'] = df['played_at'].dt.date
        df['hour'] = df['played_at'].dt.hour
        df['day_of_week'] = df['played_at'].dt.dayofweek
        df['month'] = df['played_at'].dt.month
        df['year'] = df['played_at'].dt.year
        
        # Content type
        df['is_music'] = df['track_name'].notna()
        df['is_podcast'] = df['episode_name'].notna()
        
        # Skip detection
        df['likely_skipped'] = (df['ms_played'] < 30000) | (df['skipped'] == True)
        
        # Platform categorization
        df['platform_type'] = df['platform'].apply(self._categorize_platform)
        
        # Location-based cultural context
        df['listening_location'] = df.apply(self._categorize_location, axis=1)
        
        # Listening context
        df['listening_context'] = df['reason_start'].apply(self._categorize_listening_context)
        
        # Track identification
        df['track_id'] = df['spotify_track_uri'].apply(self._extract_track_id)
        df['artist_id'] = df['track_name'] + '_' + df['artist_name']  # Temporary until we get real IDs
        
        return df

    def _categorize_platform(self, platform: str) -> str:
        """Categorize platform into broader types"""
        
        if pd.isna(platform):
            return 'Unknown'
        
        platform_lower = platform.lower()
        
        if 'ios' in platform_lower or 'iphone' in platform_lower:
            return 'iOS'
        elif 'android' in platform_lower:
            return 'Android'
        elif 'windows' in platform_lower:
            return 'Windows'
        elif 'macos' in platform_lower or 'mac' in platform_lower:
            return 'macOS'
        elif 'web' in platform_lower or 'browser' in platform_lower:
            return 'Web'
        else:
            return 'Other'

    def _categorize_location(self, row) -> str:
        """Categorize listening location based on IP and country"""
        
        country = row['conn_country']
        ip_addr = row['ip_addr']
        
        if country in self.config['cultural_detection']['vietnam_country_codes']:
            return 'Vietnam'
        elif country in self.config['cultural_detection']['western_country_codes']:
            return 'Western'
        elif pd.notna(ip_addr):
            # Check Vietnamese IP prefixes
            for prefix in self.config['cultural_detection']['vietnam_ip_prefixes']:
                if ip_addr.startswith(prefix):
                    return 'Vietnam'
            return 'Other'
        else:
            return 'Unknown'

    def _categorize_listening_context(self, reason_start: str) -> str:
        """Categorize listening context from reason_start"""
        
        if pd.isna(reason_start):
            return 'Unknown'
        
        reason_lower = reason_start.lower()
        
        if 'playlist' in reason_lower:
            return 'Playlist'
        elif 'album' in reason_lower:
            return 'Album'
        elif 'artist' in reason_lower:
            return 'Artist'
        elif 'search' in reason_lower:
            return 'Search'
        elif 'radio' in reason_lower:
            return 'Radio'
        elif 'recommendation' in reason_lower:
            return 'Recommendation'
        elif 'shuffle' in reason_lower:
            return 'Shuffle'
        else:
            return 'Other'

    def _extract_track_id(self, spotify_uri: str) -> Optional[str]:
        """Extract track ID from Spotify URI"""
        
        if pd.isna(spotify_uri):
            return None
        
        try:
            # Format: spotify:track:TRACK_ID
            return spotify_uri.split(':')[-1]
        except:
            return None

    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters"""
        
        if len(df) == 0:
            return df
        
        initial_size = len(df)
        
        # Filter 1: Minimum listening time
        min_ms = self.config['data_quality']['min_ms_played']
        df = df[df['ms_played'] >= min_ms]
        self.logger.info(f"After minimum listening time filter: {len(df)} records (removed {initial_size - len(df)})")
        
        # Filter 2: Maximum listening time (catch data errors)
        max_ms = self.config['data_quality']['max_ms_played']
        df = df[df['ms_played'] <= max_ms]
        self.logger.info(f"After maximum listening time filter: {len(df)} records")
        
        # Filter 3: Music only (exclude podcasts if configured)
        if self.config['data_quality']['exclude_podcasts']:
            df = df[df['is_music'] == True]
            self.logger.info(f"After music-only filter: {len(df)} records")
        
        # Filter 4: Valid track information
        df = df[df['track_name'].notna() & df['artist_name'].notna()]
        self.logger.info(f"After valid track info filter: {len(df)} records")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df

    def _generate_processing_stats(self, df: pd.DataFrame) -> ProcessingStats:
        """Generate comprehensive processing statistics"""
        
        if len(df) == 0:
            return ProcessingStats(
                total_records=0, audio_records=0, video_records=0,
                unique_tracks=0, unique_artists=0,
                date_range=(datetime.now(), datetime.now()),
                countries=[], platforms=[],
                total_listening_time_hours=0.0,
                avg_track_completion_rate=0.0
            )
        
        # Basic counts
        total_records = len(df)
        audio_records = df['is_music'].sum()
        video_records = df['is_podcast'].sum()
        unique_tracks = df['track_id'].nunique()
        unique_artists = df['artist_name'].nunique()
        
        # Date range
        min_date = df['played_at'].min()
        max_date = df['played_at'].max()
        date_range = (min_date, max_date)
        
        # Geographic and platform diversity
        countries = sorted(df['conn_country'].unique().tolist())
        platforms = sorted(df['platform_type'].unique().tolist())
        
        # Listening time
        total_listening_time_hours = df['ms_played'].sum() / (1000 * 60 * 60)
        
        # Track completion rate (rough estimate)
        # Assume average track length is 3.5 minutes
        avg_track_length_ms = 3.5 * 60 * 1000
        completion_rates = df['ms_played'] / avg_track_length_ms
        completion_rates = np.clip(completion_rates, 0, 1)  # Cap at 100%
        avg_track_completion_rate = completion_rates.mean()
        
        return ProcessingStats(
            total_records=total_records,
            audio_records=audio_records,
            video_records=video_records,
            unique_tracks=unique_tracks,
            unique_artists=unique_artists,
            date_range=date_range,
            countries=countries,
            platforms=platforms,
            total_listening_time_hours=total_listening_time_hours,
            avg_track_completion_rate=avg_track_completion_rate
        )

    def _save_processing_stats(self, stats_path: Path):
        """Save processing statistics to JSON"""
        
        if self.processing_stats is None:
            return
        
        stats_dict = {
            'total_records': self.processing_stats.total_records,
            'audio_records': self.processing_stats.audio_records,
            'video_records': self.processing_stats.video_records,
            'unique_tracks': self.processing_stats.unique_tracks,
            'unique_artists': self.processing_stats.unique_artists,
            'date_range': [
                self.processing_stats.date_range[0].isoformat(),
                self.processing_stats.date_range[1].isoformat()
            ],
            'countries': self.processing_stats.countries,
            'platforms': self.processing_stats.platforms,
            'total_listening_time_hours': self.processing_stats.total_listening_time_hours,
            'avg_track_completion_rate': self.processing_stats.avg_track_completion_rate,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

    def create_listening_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create listening sessions from streaming data"""
        
        if len(df) == 0:
            return df
        
        self.logger.info("Creating listening sessions")
        
        # Sort by timestamp
        df = df.sort_values('played_at').reset_index(drop=True)
        
        # Calculate time gaps between consecutive plays
        df['time_gap_minutes'] = df['played_at'].diff().dt.total_seconds() / 60
        
        # Identify session breaks
        session_gap_minutes = self.config['temporal_processing']['session_gap_minutes']
        df['session_break'] = (df['time_gap_minutes'] > session_gap_minutes) | df['time_gap_minutes'].isna()
        
        # Assign session IDs
        df['session_id'] = df['session_break'].cumsum()
        
        # Calculate session statistics
        session_stats = df.groupby('session_id').agg({
            'played_at': ['min', 'max', 'count'],
            'track_id': 'nunique',
            'artist_name': 'nunique',
            'ms_played': 'sum',
            'likely_skipped': 'sum',
            'listening_location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'platform_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        # Flatten column names
        session_stats.columns = [
            'session_start', 'session_end', 'session_length',
            'unique_tracks', 'unique_artists', 'total_ms_played',
            'tracks_skipped', 'primary_location', 'primary_platform'
        ]
        
        # Calculate session duration
        session_stats['session_duration_minutes'] = (
            (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds() / 60
        ).round(1)
        
        # Skip rate
        session_stats['skip_rate'] = (
            session_stats['tracks_skipped'] / session_stats['session_length']
        ).fillna(0).round(3)
        
        # Merge session stats back to main dataframe
        df = df.merge(session_stats, on='session_id', how='left', suffixes=('', '_session'))
        
        # Position in session
        df['track_position_in_session'] = df.groupby('session_id').cumcount() + 1
        
        self.logger.info(f"Created {df['session_id'].nunique()} listening sessions")
        
        return df

    def extract_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract temporal listening patterns"""
        
        if len(df) == 0:
            return {}
        
        self.logger.info("Extracting temporal patterns")
        
        patterns = {}
        
        # Daily patterns
        hourly_activity = df.groupby('hour')['ms_played'].agg(['count', 'sum']).reset_index()
        hourly_activity['avg_track_length'] = hourly_activity['sum'] / hourly_activity['count'] / (1000 * 60)
        patterns['hourly_activity'] = hourly_activity.to_dict('records')
        
        # Weekly patterns
        daily_activity = df.groupby('day_of_week')['ms_played'].agg(['count', 'sum']).reset_index()
        daily_activity['day_name'] = daily_activity['day_of_week'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        patterns['daily_activity'] = daily_activity.to_dict('records')
        
        # Monthly patterns
        monthly_activity = df.groupby('month')['ms_played'].agg(['count', 'sum']).reset_index()
        patterns['monthly_activity'] = monthly_activity.to_dict('records')
        
        # Listening context patterns
        context_patterns = df.groupby('listening_context')['ms_played'].agg(['count', 'sum']).reset_index()
        context_patterns['avg_session_length'] = context_patterns['sum'] / context_patterns['count'] / (1000 * 60)
        patterns['context_patterns'] = context_patterns.to_dict('records')
        
        # Location patterns
        location_patterns = df.groupby('listening_location')['ms_played'].agg(['count', 'sum']).reset_index()
        patterns['location_patterns'] = location_patterns.to_dict('records')
        
        # Platform preferences
        platform_patterns = df.groupby('platform_type')['ms_played'].agg(['count', 'sum']).reset_index()
        patterns['platform_patterns'] = platform_patterns.to_dict('records')
        
        return patterns

    def detect_cultural_preferences(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect cultural preferences in listening data"""
        
        if len(df) == 0:
            return {}
        
        self.logger.info("Detecting cultural preferences")
        
        cultural_analysis = {}
        
        # Location-based listening analysis
        location_analysis = df.groupby(['listening_location', 'date']).agg({
            'ms_played': 'sum',
            'track_id': 'nunique',
            'artist_name': 'nunique'
        }).reset_index()
        
        cultural_analysis['location_timeline'] = location_analysis.to_dict('records')
        
        # Artist nationality analysis (based on names - preliminary)
        artist_cultural_hints = self._detect_artist_cultural_hints(df)
        cultural_analysis['artist_cultural_distribution'] = artist_cultural_hints
        
        # Temporal cultural patterns
        cultural_timeline = df.groupby(['date', 'listening_location']).size().unstack(fill_value=0)
        cultural_analysis['daily_cultural_balance'] = cultural_timeline.to_dict('index')
        
        return cultural_analysis

    def _detect_artist_cultural_hints(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect cultural hints from artist names (preliminary analysis)"""
        
        # This is a basic implementation - would be enhanced with proper cultural classification
        artists = df['artist_name'].value_counts().head(50)
        
        # Basic Vietnamese name detection patterns
        vietnamese_patterns = [
            'Sơn Tùng', 'Hoàng Thùy Linh', 'Đức Phúc', 'Bích Phương', 'Chi Pu',
            'Noo Phước Thịnh', 'Hương Tràm', 'Mỹ Tâm', 'Đàm Vĩnh Hưng'
        ]
        
        cultural_hints = {
            'likely_vietnamese_artists': [],
            'likely_western_artists': [],
            'unknown_artists': []
        }
        
        for artist, count in artists.items():
            if any(pattern in artist for pattern in vietnamese_patterns):
                cultural_hints['likely_vietnamese_artists'].append({'artist': artist, 'play_count': count})
            elif any(char in artist for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                cultural_hints['likely_vietnamese_artists'].append({'artist': artist, 'play_count': count})
            else:
                # Simple heuristic: if it's mostly ASCII, likely Western
                if all(ord(char) < 128 for char in artist):
                    cultural_hints['likely_western_artists'].append({'artist': artist, 'play_count': count})
                else:
                    cultural_hints['unknown_artists'].append({'artist': artist, 'play_count': count})
        
        return cultural_hints


# High-level processing functions
def process_spotify_extended_data(
    data_path: str,
    output_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, ProcessingStats]:
    """
    Process Spotify Extended Streaming History data.
    
    This is the main entry point for processing Spotify extended data.
    """
    
    processor = ExtendedSpotifyProcessor(config)
    
    # Process the raw data
    df, stats = processor.process_extended_history(data_path, output_path)
    
    # Create listening sessions
    df = processor.create_listening_sessions(df)
    
    # Extract temporal patterns
    temporal_patterns = processor.extract_temporal_patterns(df)
    
    # Detect cultural preferences
    cultural_analysis = processor.detect_cultural_preferences(df)
    
    # Add analysis results to DataFrame metadata
    df.attrs['temporal_patterns'] = temporal_patterns
    df.attrs['cultural_analysis'] = cultural_analysis
    df.attrs['processing_stats'] = stats
    
    return df, stats