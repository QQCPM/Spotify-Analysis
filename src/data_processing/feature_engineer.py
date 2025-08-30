"""
Feature Engineering Module

Creates comprehensive features for cross-cultural music recommendation.
Includes temporal, audio, cultural, and behavioral features.
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
import logging

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering for cross-cultural music recommendation.
    
    Creates features for:
    - Temporal patterns (time of day, seasonality, listening sessions)
    - Audio characteristics (energy, valence, danceability)
    - Cultural markers (language, market, genre diversity)
    - Behavioral patterns (skip rate, repeat plays, exploration)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        self.scalers = {}
        self.encoders = {}
        self.latent_models = {}
        self.interaction_matrices = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for feature engineering"""
        return {
            'temporal_window_hours': 4,  # Session definition
            'audio_features': [
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms'
            ],
            'cultural_features': [
                'vietnamese_score', 'western_score', 'bridge_score',
                'dominant_culture', 'cultural_confidence'
            ],
            'normalize_features': True,
            'pca_components': 10,
            'clustering_components': 5,
            'latent_factors': {
                'n_factors': 50,
                'svd_components': 20,
                'nmf_components': 15,
                'min_interactions': 5,
                'regularization': 0.01,
                'max_iter': 1000,
                'random_state': 42
            }
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for feature engineering"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive temporal features from listening timestamps.
        
        Args:
            data: DataFrame with 'played_at' timestamp column
            
        Returns:
            DataFrame with temporal features added
        """
        self.logger.info("Creating temporal features")
        
        df = data.copy()
        
        # Convert timestamp
        df['played_at'] = pd.to_datetime(df['played_at'])
        
        # Basic time features
        df['hour'] = df['played_at'].dt.hour
        df['day_of_week'] = df['played_at'].dt.dayofweek
        df['month'] = df['played_at'].dt.month
        df['quarter'] = df['played_at'].dt.quarter
        df['year'] = df['played_at'].dt.year
        
        # Time of day categories
        df['time_of_day'] = pd.cut(
            df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Season (Northern Hemisphere)
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
        
    def create_session_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create listening session features based on temporal patterns.
        
        Args:
            data: DataFrame with temporal features
            
        Returns:
            DataFrame with session features added
        """
        self.logger.info("Creating session features")
        
        df = data.copy()
        df = df.sort_values('played_at').reset_index(drop=True)
        
        # Calculate time gaps between consecutive plays
        df['time_gap'] = df['played_at'].diff().dt.total_seconds() / 3600  # hours
        
        # Define sessions based on time gaps
        session_threshold = self.config['temporal_window_hours']
        df['session_break'] = (df['time_gap'] > session_threshold) | (df['time_gap'].isna())
        df['session_id'] = df['session_break'].cumsum()
        
        # Session-level statistics
        session_stats = df.groupby('session_id').agg({
            'played_at': ['min', 'max', 'count'],
            'track_id': 'nunique',
            'artist_id': 'nunique',
            'duration_ms': ['mean', 'sum']
        }).round(3)
        
        session_stats.columns = [
            'session_start', 'session_end', 'session_length',
            'unique_tracks', 'unique_artists', 'avg_duration', 'total_duration'
        ]
        
        # Session duration in minutes
        session_stats['session_duration_mins'] = (
            (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds() / 60
        )
        
        # Merge session features back
        df = df.merge(session_stats, on='session_id', how='left')
        
        # Position in session
        df['position_in_session'] = df.groupby('session_id').cumcount() + 1
        df['position_normalized'] = df['position_in_session'] / df['session_length']
        
        return df
        
    def create_audio_features(self, data: pd.DataFrame, audio_features: pd.DataFrame) -> pd.DataFrame:
        """
        Process and enhance audio features from Spotify API.
        
        Args:
            data: Main listening data
            audio_features: Audio features from Spotify
            
        Returns:
            DataFrame with processed audio features
        """
        self.logger.info("Processing audio features")
        
        # Merge audio features
        df = data.merge(audio_features, on='track_id', how='left', suffixes=('', '_audio'))
        
        # Handle missing audio features
        audio_cols = self.config['audio_features']
        missing_audio = df[audio_cols].isna().sum()
        
        if missing_audio.any():
            self.logger.warning(f"Missing audio features: {missing_audio[missing_audio > 0].to_dict()}")
            
        # Fill missing values with median
        for col in audio_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                
        # Create derived audio features
        if all(col in df.columns for col in ['energy', 'valence']):
            df['energy_valence'] = df['energy'] * df['valence']  # Energetic positivity
            
        if all(col in df.columns for col in ['danceability', 'energy']):
            df['dance_energy'] = df['danceability'] * df['energy']  # Dance potential
            
        if 'tempo' in df.columns:
            df['tempo_category'] = pd.cut(
                df['tempo'],
                bins=[0, 90, 120, 140, 200],
                labels=['slow', 'moderate', 'fast', 'very_fast'],
                include_lowest=True
            )
            
        # Audio feature clusters
        if len([col for col in audio_cols if col in df.columns]) >= 5:
            audio_data = df[[col for col in audio_cols if col in df.columns]].fillna(0)
            
            # Standardize for clustering
            scaler = StandardScaler()
            audio_scaled = scaler.fit_transform(audio_data)
            
            # K-means clustering
            kmeans = KMeans(
                n_clusters=self.config['clustering_components'], 
                random_state=42
            )
            df['audio_cluster'] = kmeans.fit_predict(audio_scaled)
            
            # Store scaler for future use
            self.scalers['audio'] = scaler
            
        return df
        
    def create_cultural_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced cultural features beyond basic categorization.
        
        Args:
            data: DataFrame with cultural categorization
            
        Returns:
            DataFrame with enhanced cultural features
        """
        self.logger.info("Creating enhanced cultural features")
        
        df = data.copy()
        
        # Cultural diversity per session
        cultural_diversity = df.groupby('session_id').agg({
            'vietnamese_score': 'mean',
            'western_score': 'mean',
            'bridge_score': 'mean',
            'dominant_culture': lambda x: len(x.unique())
        })
        
        cultural_diversity.columns = [
            'session_vietnamese_score', 'session_western_score',
            'session_bridge_score', 'session_cultural_diversity'
        ]
        
        df = df.merge(cultural_diversity, on='session_id', how='left')
        
        # Cultural transition indicators
        df['cultural_transition'] = (
            df['dominant_culture'] != df['dominant_culture'].shift(1)
        ).astype(int)
        
        # Rolling cultural preferences (last 10 songs)
        window = 10
        df['rolling_vietnamese'] = df['vietnamese_score'].rolling(window=window).mean()
        df['rolling_western'] = df['western_score'].rolling(window=window).mean()
        df['rolling_bridge'] = df['bridge_score'].rolling(window=window).mean()
        
        return df
        
    def create_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features from listening patterns.
        
        Args:
            data: DataFrame with listening data
            
        Returns:
            DataFrame with behavioral features
        """
        self.logger.info("Creating behavioral features")
        
        df = data.copy()
        
        # Artist and track repetition patterns
        df['track_play_count'] = df.groupby('track_id')['track_id'].transform('count')
        df['artist_play_count'] = df.groupby('artist_id')['artist_id'].transform('count')
        
        # Exploration indicators
        df['is_new_track'] = df['track_play_count'] == 1
        df['is_new_artist'] = df['artist_play_count'] == 1
        
        # User-level statistics (if multiple users)
        if 'user_id' in df.columns:
            user_stats = df.groupby('user_id').agg({
                'track_id': 'nunique',
                'artist_id': 'nunique',
                'played_at': 'count'
            })
            user_stats.columns = ['unique_tracks_user', 'unique_artists_user', 'total_plays_user']
            df = df.merge(user_stats, on='user_id', how='left')
            
        # Time since last play of same track/artist
        df = df.sort_values(['track_id', 'played_at'])
        df['time_since_track'] = df.groupby('track_id')['played_at'].diff().dt.total_seconds() / 3600
        
        df = df.sort_values(['artist_id', 'played_at'])  
        df['time_since_artist'] = df.groupby('artist_id')['played_at'].diff().dt.total_seconds() / 3600
        
        # Popularity vs personal preference
        if 'popularity' in df.columns:
            user_avg_popularity = df.groupby('user_id')['popularity'].mean() if 'user_id' in df.columns else df['popularity'].mean()
            df['popularity_deviation'] = df['popularity'] - user_avg_popularity
            
        return df
        
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different feature types.
        
        Args:
            data: DataFrame with all base features
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info("Creating interaction features")
        
        df = data.copy()
        
        # Time-Cultural interactions
        if all(col in df.columns for col in ['hour', 'vietnamese_score', 'western_score']):
            df['morning_vietnamese'] = (df['hour'].between(6, 12)) * df['vietnamese_score']
            df['evening_western'] = (df['hour'].between(18, 24)) * df['western_score']
            
        # Audio-Cultural interactions
        if all(col in df.columns for col in ['energy', 'vietnamese_score', 'western_score']):
            df['high_energy_vietnamese'] = (df['energy'] > 0.7) * df['vietnamese_score']
            df['low_energy_western'] = (df['energy'] < 0.3) * df['western_score']
            
        # Behavioral-Cultural interactions
        if all(col in df.columns for col in ['session_cultural_diversity', 'cultural_confidence']):
            df['diverse_confident'] = df['session_cultural_diversity'] * df['cultural_confidence']
            
        return df
        
    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features for machine learning models.
        
        Args:
            data: DataFrame with features to normalize
            
        Returns:
            DataFrame with normalized features
        """
        self.logger.info("Normalizing features")
        
        df = data.copy()
        
        # Identify numerical columns to normalize
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude certain columns from normalization
        exclude_cols = [
            'track_id', 'artist_id', 'session_id', 'user_id', 'position_in_session',
            'session_length', 'track_play_count', 'artist_play_count'
        ]
        
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if self.config['normalize_features'] and numerical_cols:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['features'] = scaler
            
        return df
        
    def create_all_features(
        self,
        listening_data: pd.DataFrame,
        audio_features: pd.DataFrame,
        cultural_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create all features using the complete pipeline.
        
        Args:
            listening_data: Base listening history data
            audio_features: Audio features from Spotify
            cultural_data: Optional pre-computed cultural features
            
        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("Starting comprehensive feature engineering")
        
        df = listening_data.copy()
        
        # Step 1: Temporal features
        df = self.create_temporal_features(df)
        
        # Step 2: Session features
        df = self.create_session_features(df)
        
        # Step 3: Audio features
        df = self.create_audio_features(df, audio_features)
        
        # Step 4: Cultural features (if not pre-computed)
        if cultural_data is not None:
            # Merge pre-computed cultural features
            cultural_cols = [col for col in cultural_data.columns if col not in df.columns or col == 'track_id']
            df = df.merge(cultural_data[cultural_cols], on='track_id', how='left')
            
        df = self.create_cultural_features(df)
        
        # Step 5: Behavioral features
        df = self.create_behavioral_features(df)
        
        # Step 6: Interaction features
        df = self.create_interaction_features(df)
        
        # Step 7: Normalization (optional)
        if self.config['normalize_features']:
            df = self.normalize_features(df)
            
        self.logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
        
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return grouped feature names for analysis"""
        return {
            'temporal': [
                'hour', 'day_of_week', 'month', 'is_weekend', 'season',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
            ],
            'session': [
                'session_length', 'unique_tracks', 'unique_artists',
                'session_duration_mins', 'position_in_session', 'position_normalized'
            ],
            'audio': [
                'danceability', 'energy', 'valence', 'tempo', 'acousticness',
                'instrumentalness', 'liveness', 'speechiness', 'audio_cluster'
            ],
            'cultural': [
                'vietnamese_score', 'western_score', 'bridge_score',
                'cultural_confidence', 'session_cultural_diversity', 'cultural_transition'
            ],
            'behavioral': [
                'track_play_count', 'artist_play_count', 'is_new_track',
                'is_new_artist', 'time_since_track', 'time_since_artist'
            ],
            'latent': [
                'svd_factors', 'nmf_factors', 'latent_clusters', 'preference_stability'
            ]
        }

    def construct_interaction_matrix(
        self, 
        data: pd.DataFrame,
        value_col: str = 'implicit_rating'
    ) -> Tuple[sparse.csr_matrix, Dict[str, int], Dict[str, int]]:
        """
        Construct user-item interaction matrix from listening data.
        
        Args:
            data: DataFrame with user_id, track_id, and interaction data
            value_col: Column to use for interaction values
            
        Returns:
            Tuple of (interaction_matrix, user_mapping, item_mapping)
        """
        self.logger.info("Constructing user-item interaction matrix")
        
        # Create implicit ratings if not present
        if value_col not in data.columns:
            if 'track_play_count' in data.columns:
                data = data.copy()
                # Convert play count to implicit rating (log-scaled)
                data[value_col] = np.log1p(data['track_play_count'])
            else:
                data = data.copy()
                # Binary interaction (1 for any play)
                data[value_col] = 1.0
        
        # Filter by minimum interactions
        min_interactions = self.config['latent_factors']['min_interactions']
        
        # Filter users with minimum interactions
        user_counts = data['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        data_filtered = data[data['user_id'].isin(valid_users)]
        
        # Filter items with minimum interactions  
        item_counts = data_filtered['track_id'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        data_filtered = data_filtered[data_filtered['track_id'].isin(valid_items)]
        
        self.logger.info(f"Filtered to {len(valid_users)} users and {len(valid_items)} items")
        
        # Create mappings
        unique_users = sorted(data_filtered['user_id'].unique())
        unique_items = sorted(data_filtered['track_id'].unique())
        
        user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        
        # Create matrix
        n_users, n_items = len(unique_users), len(unique_items)
        
        user_indices = [user_mapping[user] for user in data_filtered['user_id']]
        item_indices = [item_mapping[item] for item in data_filtered['track_id']]
        values = data_filtered[value_col].values
        
        interaction_matrix = sparse.csr_matrix(
            (values, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )
        
        self.logger.info(f"Interaction matrix shape: {interaction_matrix.shape}, "
                        f"Sparsity: {1 - interaction_matrix.nnz / (n_users * n_items):.4f}")
        
        # Store for later use
        self.interaction_matrices['user_item'] = interaction_matrix
        self.interaction_matrices['user_mapping'] = user_mapping
        self.interaction_matrices['item_mapping'] = item_mapping
        
        return interaction_matrix, user_mapping, item_mapping

    def compute_svd_factors(
        self,
        interaction_matrix: sparse.csr_matrix,
        n_components: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SVD-based latent factors for user-item interactions.
        
        Args:
            interaction_matrix: User-item interaction matrix
            n_components: Number of latent factors (default from config)
            
        Returns:
            Tuple of (user_factors, item_factors, singular_values)
        """
        self.logger.info("Computing SVD latent factors")
        
        n_components = n_components or self.config['latent_factors']['svd_components']
        
        # Use TruncatedSVD for sparse matrices
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=self.config['latent_factors']['random_state']
        )
        
        # Fit and transform
        user_factors = svd.fit_transform(interaction_matrix)
        item_factors = svd.components_.T
        singular_values = svd.singular_values_
        
        # Store model
        self.latent_models['svd'] = svd
        
        # Explained variance
        explained_variance_ratio = svd.explained_variance_ratio_
        total_variance = explained_variance_ratio.sum()
        
        self.logger.info(f"SVD factors computed: {n_components} components, "
                        f"explained variance: {total_variance:.4f}")
        
        return user_factors, item_factors, singular_values

    def compute_nmf_factors(
        self,
        interaction_matrix: sparse.csr_matrix,
        n_components: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute NMF-based latent factors (non-negative).
        
        Args:
            interaction_matrix: User-item interaction matrix
            n_components: Number of latent factors (default from config)
            
        Returns:
            Tuple of (user_factors, item_factors)
        """
        self.logger.info("Computing NMF latent factors")
        
        n_components = n_components or self.config['latent_factors']['nmf_components']
        
        # NMF requires non-negative values
        interaction_matrix_pos = interaction_matrix.copy()
        interaction_matrix_pos.data = np.maximum(interaction_matrix_pos.data, 0)
        
        nmf = NMF(
            n_components=n_components,
            init='random',
            max_iter=self.config['latent_factors']['max_iter'],
            random_state=self.config['latent_factors']['random_state'],
            alpha_W=self.config['latent_factors']['regularization'],
            alpha_H=self.config['latent_factors']['regularization']
        )
        
        # Fit and transform
        user_factors = nmf.fit_transform(interaction_matrix_pos)
        item_factors = nmf.components_.T
        
        # Store model
        self.latent_models['nmf'] = nmf
        
        # Reconstruction error
        reconstruction_error = nmf.reconstruction_err_
        
        self.logger.info(f"NMF factors computed: {n_components} components, "
                        f"reconstruction error: {reconstruction_error:.4f}")
        
        return user_factors, item_factors

    def interpret_latent_factors(
        self,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        audio_features: pd.DataFrame,
        cultural_features: pd.DataFrame,
        item_mapping: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Interpret latent factors by correlating with audio and cultural features.
        
        Args:
            user_factors: User latent factor matrix
            item_factors: Item latent factor matrix  
            audio_features: Audio features DataFrame
            cultural_features: Cultural features DataFrame
            item_mapping: Track ID to matrix index mapping
            
        Returns:
            Dictionary with factor interpretations
        """
        self.logger.info("Interpreting latent factors")
        
        # Create reverse mapping
        idx_to_item = {idx: item for item, idx in item_mapping.items()}
        
        # Align audio features with item factors
        audio_aligned = []
        cultural_aligned = []
        item_factors_aligned = []
        
        for idx in range(len(item_factors)):
            track_id = idx_to_item[idx]
            
            # Get audio features for this track
            audio_row = audio_features[audio_features['track_id'] == track_id]
            cultural_row = cultural_features[cultural_features['track_id'] == track_id]
            
            if not audio_row.empty and not cultural_row.empty:
                audio_cols = [col for col in self.config['audio_features'] if col in audio_row.columns]
                cultural_cols = [col for col in self.config['cultural_features'] if col in cultural_row.columns]
                
                if audio_cols and cultural_cols:
                    audio_aligned.append(audio_row[audio_cols].iloc[0].values)
                    cultural_aligned.append(cultural_row[cultural_cols].iloc[0].values)
                    item_factors_aligned.append(item_factors[idx])
        
        if not audio_aligned:
            self.logger.warning("No aligned features found for interpretation")
            return {}
        
        audio_matrix = np.array(audio_aligned)
        cultural_matrix = np.array(cultural_aligned)
        factors_matrix = np.array(item_factors_aligned)
        
        # Compute correlations between factors and features
        n_factors = factors_matrix.shape[1]
        interpretations = {}
        
        for factor_idx in range(n_factors):
            factor_values = factors_matrix[:, factor_idx]
            
            # Audio correlations
            audio_correlations = {}
            for feat_idx, feat_name in enumerate(self.config['audio_features']):
                if feat_idx < audio_matrix.shape[1]:
                    correlation, p_value = pearsonr(factor_values, audio_matrix[:, feat_idx])
                    if abs(correlation) > 0.3 and p_value < 0.05:
                        audio_correlations[feat_name] = {
                            'correlation': correlation,
                            'p_value': p_value
                        }
            
            # Cultural correlations
            cultural_correlations = {}
            for feat_idx, feat_name in enumerate(self.config['cultural_features']):
                if feat_idx < cultural_matrix.shape[1] and feat_name != 'dominant_culture':
                    correlation, p_value = pearsonr(factor_values, cultural_matrix[:, feat_idx])
                    if abs(correlation) > 0.3 and p_value < 0.05:
                        cultural_correlations[feat_name] = {
                            'correlation': correlation,
                            'p_value': p_value
                        }
            
            # Generate interpretation
            interpretation = self._generate_factor_interpretation(
                audio_correlations, cultural_correlations
            )
            
            interpretations[f'factor_{factor_idx}'] = {
                'audio_correlations': audio_correlations,
                'cultural_correlations': cultural_correlations,
                'interpretation': interpretation,
                'factor_strength': np.std(factor_values)
            }
        
        return interpretations

    def _generate_factor_interpretation(
        self,
        audio_correlations: Dict,
        cultural_correlations: Dict
    ) -> str:
        """Generate human-readable interpretation of a latent factor."""
        
        # Identify strongest correlations
        all_correlations = {}
        for feat, data in audio_correlations.items():
            all_correlations[feat] = abs(data['correlation'])
        for feat, data in cultural_correlations.items():
            all_correlations[feat] = abs(data['correlation'])
        
        if not all_correlations:
            return "Uninterpretable factor (no significant correlations)"
        
        # Get top correlations
        top_features = sorted(all_correlations.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate description
        descriptions = []
        for feat, strength in top_features:
            if feat in audio_correlations:
                direction = "high" if audio_correlations[feat]['correlation'] > 0 else "low"
                descriptions.append(f"{direction} {feat}")
            elif feat in cultural_correlations:
                direction = "high" if cultural_correlations[feat]['correlation'] > 0 else "low"
                descriptions.append(f"{direction} {feat}")
        
        return f"Factor characterized by {', '.join(descriptions)}"

    def analyze_factor_stability(
        self,
        data: pd.DataFrame,
        time_windows: List[Tuple[datetime, datetime]]
    ) -> Dict[str, Any]:
        """
        Analyze temporal stability of latent factors across time windows.
        
        Args:
            data: Full listening data with timestamps
            time_windows: List of (start, end) datetime tuples
            
        Returns:
            Dictionary with stability analysis results
        """
        self.logger.info(f"Analyzing factor stability across {len(time_windows)} time windows")
        
        window_factors = []
        window_info = []
        
        for i, (start_time, end_time) in enumerate(time_windows):
            # Filter data for this time window
            window_data = data[
                (data['played_at'] >= start_time) & 
                (data['played_at'] < end_time)
            ].copy()
            
            if len(window_data) < self.config['latent_factors']['min_interactions']:
                self.logger.warning(f"Insufficient data in window {i}, skipping")
                continue
            
            # Construct interaction matrix for this window
            try:
                interaction_matrix, user_mapping, item_mapping = self.construct_interaction_matrix(window_data)
                
                # Compute factors
                user_factors, item_factors, _ = self.compute_svd_factors(interaction_matrix)
                
                window_factors.append({
                    'window_idx': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'user_factors': user_factors,
                    'item_factors': item_factors,
                    'n_users': len(user_mapping),
                    'n_items': len(item_mapping)
                })
                
                window_info.append({
                    'window_idx': i,
                    'n_interactions': len(window_data),
                    'n_users': len(user_mapping),
                    'n_items': len(item_mapping)
                })
                
            except Exception as e:
                self.logger.error(f"Error processing window {i}: {str(e)}")
                continue
        
        if len(window_factors) < 2:
            self.logger.warning("Insufficient windows for stability analysis")
            return {}
        
        # Compute stability metrics
        stability_metrics = self._compute_stability_metrics(window_factors)
        
        return {
            'window_info': window_info,
            'stability_metrics': stability_metrics,
            'n_windows': len(window_factors)
        }

    def _compute_stability_metrics(self, window_factors: List[Dict]) -> Dict[str, float]:
        """Compute stability metrics between consecutive time windows."""
        
        correlations = []
        
        for i in range(len(window_factors) - 1):
            factors_1 = window_factors[i]['user_factors']
            factors_2 = window_factors[i + 1]['user_factors']
            
            # Match common dimensions
            min_dim = min(factors_1.shape[1], factors_2.shape[1])
            
            # Compute factor correlations (column-wise)
            window_correlations = []
            for dim in range(min_dim):
                corr, _ = pearsonr(factors_1[:, dim], factors_2[:, dim])
                if not np.isnan(corr):
                    window_correlations.append(abs(corr))
            
            if window_correlations:
                correlations.extend(window_correlations)
        
        if not correlations:
            return {}
        
        return {
            'mean_stability': np.mean(correlations),
            'std_stability': np.std(correlations),
            'min_stability': np.min(correlations),
            'max_stability': np.max(correlations),
            'stability_score': np.mean(correlations)  # Main metric
        }


def engineer_features(
    listening_data: pd.DataFrame,
    audio_features: pd.DataFrame,
    cultural_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    High-level function for comprehensive feature engineering.
    
    Args:
        listening_data: Base listening history
        audio_features: Audio features from Spotify
        cultural_data: Optional cultural categorization
        config: Optional configuration
        
    Returns:
        Tuple of (engineered_features, feature_groups)
    """
    engineer = FeatureEngineer(config)
    
    engineered_data = engineer.create_all_features(
        listening_data, audio_features, cultural_data
    )
    
    feature_groups = engineer.get_feature_importance_groups()
    
    return engineered_data, feature_groups