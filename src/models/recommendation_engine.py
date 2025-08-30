"""
Phase 4: Cross-Cultural Music Recommendation Engine

Leverages Phase 3 discoveries:
- 3 Musical Personalities 
- 81 Preference Change Points
- 655+ Cultural Bridge Songs

Core architecture for personality-aware, temporal, cross-cultural recommendations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Result from recommendation engine"""
    track_id: str
    track_name: str
    artist_name: str
    score: float
    personality_scores: Dict[str, float]
    bridge_score: float
    temporal_weight: float
    cultural_classification: str
    reasoning: str


@dataclass 
class UserProfile:
    """Dynamic user profile based on Phase 3 discoveries"""
    personality_weights: Dict[str, float]  # Weight for each of 3 personalities
    current_preferences: Dict[str, float]  # Audio feature preferences
    cultural_preferences: Dict[str, float]  # Vietnamese/Western/Chinese preferences
    recent_change_points: List[datetime]    # Recent preference shifts
    bridge_song_affinity: float            # Likelihood to explore via bridges
    temporal_context: str                  # Current listening context


class PersonalityRecommender:
    """
    Individual recommender for each discovered musical personality.
    
    Based on Phase 3 findings:
    - Personality 1: Vietnamese-dominant Indie/Alternative (buitruonglinh-led)
    - Personality 2: Culturally diverse Mixed genre (Christina Grimmie-led)  
    - Personality 3: Western-dominant Mixed genre (14 Casper-led)
    """
    
    def __init__(self, personality_id: str, personality_data: Dict):
        self.personality_id = personality_id
        self.personality_data = personality_data
        
        # Handle top_artists as either list or dict
        top_artists_raw = personality_data.get('top_artists', [])
        if isinstance(top_artists_raw, list):
            # Convert list to dict with equal weights
            self.top_artists = {artist: 1.0 for artist in top_artists_raw}
        else:
            self.top_artists = top_artists_raw
            
        self.cultural_profile = personality_data.get('cultural_profile', {})
        self.strength = personality_data.get('strength', 1.0)
        
        # Audio feature preferences for this personality
        self.audio_preferences = self._extract_audio_preferences()
        
    def _extract_audio_preferences(self) -> Dict[str, float]:
        """Extract audio preferences from personality characteristics"""
        
        # Default preferences based on personality interpretation
        if 'vietnamese' in self.personality_data.get('interpretation', '').lower():
            # Vietnamese personality tends to prefer moderate energy, varied valence
            return {
                'energy': 0.5,
                'valence': 0.4,  
                'danceability': 0.6,
                'acousticness': 0.6,
                'speechiness': 0.1
            }
        elif 'western' in self.personality_data.get('interpretation', '').lower():
            # Western personality tends to prefer higher energy, moderate valence
            return {
                'energy': 0.7,
                'valence': 0.6,
                'danceability': 0.7, 
                'acousticness': 0.3,
                'speechiness': 0.1
            }
        else:
            # Mixed/diverse personality - balanced preferences
            return {
                'energy': 0.6,
                'valence': 0.5,
                'danceability': 0.65,
                'acousticness': 0.4,
                'speechiness': 0.15
            }
    
    def calculate_personality_score(self, track_features: Dict[str, float]) -> float:
        """Calculate how well a track matches this personality"""
        
        score = 0.0
        total_weight = 0.0
        
        # Audio feature similarity
        for feature, preferred_value in self.audio_preferences.items():
            if feature in track_features:
                feature_score = 1 - abs(preferred_value - track_features[feature])
                score += feature_score * 0.4  # 40% weight to audio features
                total_weight += 0.4
        
        # Artist affinity (if track artist is in top artists for this personality)
        track_artist = track_features.get('artist_name', '').lower()
        artist_score = 0.0
        for artist, count in self.top_artists.items():
            if artist.lower() in track_artist or track_artist in artist.lower():
                artist_score = min(count / 100.0, 1.0)  # Normalize artist popularity
                break
        
        score += artist_score * 0.4  # 40% weight to artist affinity
        total_weight += 0.4
        
        # Cultural alignment
        track_culture = track_features.get('cultural_classification', 'unknown')
        cultural_weight = 0.0
        if track_culture == 'vietnamese':
            cultural_weight = self.cultural_profile.get('vietnamese_ratio', 0)
        elif track_culture == 'western':
            cultural_weight = self.cultural_profile.get('western_ratio', 0)
        elif track_culture in ['chinese', 'bridge']:
            cultural_weight = 0.5  # Moderate weight for other cultures
            
        score += cultural_weight * 0.2  # 20% weight to cultural alignment
        total_weight += 0.2
        
        return score / max(total_weight, 0.1)


class TemporalWeightingSystem:
    """
    Applies temporal weighting based on 81 discovered preference change points.
    
    Recent preferences get higher weight, with decay based on change point proximity.
    """
    
    def __init__(self, change_points: List[Dict]):
        self.change_points = change_points
        self.change_point_dates = [pd.to_datetime(cp['date']) for cp in change_points]
        
    def calculate_temporal_weight(
        self,
        track_timestamp: datetime,
        current_time: Optional[datetime] = None
    ) -> float:
        """Calculate temporal weight for a track based on change points"""
        
        if current_time is None:
            current_time = pd.Timestamp.now(tz='UTC')
            
        track_time = pd.to_datetime(track_timestamp)
        
        # Ensure both timestamps are timezone-aware for comparison
        if track_time.tz is None:
            track_time = track_time.tz_localize('UTC')
        if hasattr(current_time, 'tz') and current_time.tz is None:
            current_time = current_time.tz_localize('UTC')
            
        # Base temporal decay (more recent = higher weight)
        days_ago = (current_time - track_time).days
        base_weight = np.exp(-days_ago / 180.0)  # 180-day half-life
        
        # Adjust based on proximity to change points
        change_point_adjustment = 1.0
        for cp_date in self.change_point_dates:
            # Ensure timezone compatibility
            if cp_date.tz is None:
                cp_date = cp_date.tz_localize('UTC')
                
            days_from_cp = abs((track_time - cp_date).days)
            if days_from_cp < 30:  # Within 30 days of change point
                # Recent change points reduce weight (preference may have shifted)
                cp_recency = (current_time - cp_date).days
                if cp_recency < 90:  # Recent change point
                    change_point_adjustment *= 0.7  # Reduce weight
                    
        return base_weight * change_point_adjustment
    
    def get_current_preference_context(self, current_time: Optional[datetime] = None) -> str:
        """Determine current preference context based on recent change points"""
        
        if current_time is None:
            current_time = pd.Timestamp.now(tz='UTC')
            
        # Check for recent change points
        recent_changes = []
        for cp in self.change_points:
            cp_date = pd.to_datetime(cp['date'])
            
            # Ensure timezone compatibility
            if cp_date.tz is None:
                cp_date = cp_date.tz_localize('UTC')
            if hasattr(current_time, 'tz') and current_time.tz is None:
                current_time = current_time.tz_localize('UTC')
                
            days_since = (current_time - cp_date).days
            if days_since < 60:  # Within 60 days
                recent_changes.append(cp)
                
        if not recent_changes:
            return "stable"
        
        # Analyze recent changes
        recent_signals = []
        for cp in recent_changes:
            recent_signals.extend(cp.get('signals_affected', []))
            
        if 'vietnamese_score' in recent_signals:
            return "vietnamese_shift"
        elif 'western_score' in recent_signals:
            return "western_shift" 
        elif 'energy' in str(recent_signals):
            return "energy_shift"
        else:
            return "preference_exploration"


class CulturalBridgeEngine:
    """
    Leverages 655+ discovered bridge songs for cross-cultural music discovery.
    
    Identifies optimal transition points between Vietnamese and Western music.
    """
    
    def __init__(self, bridge_songs: List[Dict]):
        self.bridge_songs = bridge_songs
        self.bridge_lookup = {
            f"{song['track_name']}_{song['artist_name']}".lower(): song 
            for song in bridge_songs
        }
        
    def calculate_bridge_score(self, track_features: Dict[str, float]) -> float:
        """Calculate bridge potential for a track"""
        
        track_key = f"{track_features.get('track_name', '')}_{track_features.get('artist_name', '')}".lower()
        
        # Direct bridge song match
        if track_key in self.bridge_lookup:
            bridge_data = self.bridge_lookup[track_key]
            return bridge_data.get('bridge_score', 5.0) / 10.0  # Normalize to 0-1
        
        # Calculate bridge potential based on characteristics
        bridge_score = 0.0
        
        # Audio characteristics that make good bridges
        energy = track_features.get('energy', 0.5)
        valence = track_features.get('valence', 0.5) 
        acousticness = track_features.get('acousticness', 0.5)
        
        # Moderate energy and high valence = good bridge potential
        if 0.4 <= energy <= 0.7:
            bridge_score += 0.3
        if valence > 0.6:
            bridge_score += 0.3
        if acousticness > 0.3:
            bridge_score += 0.2
            
        # Cultural versatility
        cultural = track_features.get('cultural_classification', 'unknown')
        if cultural == 'bridge' or cultural == 'other':
            bridge_score += 0.2
            
        return min(bridge_score, 1.0)
    
    def get_bridge_recommendations(
        self,
        current_culture: str,
        target_culture: str,
        n_recommendations: int = 5
    ) -> List[Dict]:
        """Get bridge songs for transitioning between cultures"""
        
        suitable_bridges = []
        
        for bridge_song in self.bridge_songs[:20]:  # Top 20 bridges
            # Check if bridge connects the cultures
            bridge_reasons = bridge_song.get('bridge_reasons', [])
            
            # Good bridge if it has cross-cultural elements
            if (current_culture == 'vietnamese' and target_culture == 'western') or \
               (current_culture == 'western' and target_culture == 'vietnamese'):
                if 'cross_cultural_artist' in bridge_reasons or \
                   'high_valence' in bridge_reasons:
                    suitable_bridges.append(bridge_song)
                    
        return suitable_bridges[:n_recommendations]


class CrossCulturalRecommendationEngine:
    """
    Main recommendation engine integrating all Phase 3 discoveries.
    
    Combines 3 personality recommenders with temporal weighting and bridge detection.
    """
    
    def __init__(self, phase3_results_path: Optional[str] = None):
        # Use default path if not provided
        if phase3_results_path is None:
            phase3_results_path = "results/phase3"
            
        self.phase3_results_path = Path(phase3_results_path)
        self.personality_recommenders = {}
        self.temporal_weighting = None
        self.bridge_engine = None
        self.scaler = StandardScaler()
        
        try:
            self._load_phase3_results()
            self._initialize_components()
        except Exception as e:
            # For research: fail gracefully with informative message
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ Could not load Phase 3 results: {e}")
            logger.info("💡 Try running: python phase3_deep_analysis.py")
            
            # Initialize with minimal components for testing
            self._initialize_minimal_components()
        
    def _load_phase3_results(self):
        """Load Phase 3 analysis results"""
        
        # Find latest comprehensive report
        results_files = list(self.phase3_results_path.glob('comprehensive_research_report_*.json'))
        if not results_files:
            raise FileNotFoundError("No Phase 3 results found")
            
        latest_report = sorted(results_files)[-1]
        
        with open(latest_report, 'r') as f:
            self.phase3_results = json.load(f)
            
        logger.info(f"Loaded Phase 3 results from {latest_report}")
        
    def _initialize_components(self):
        """Initialize all recommendation components"""
        
        # Initialize personality recommenders
        personalities = self.phase3_results['study_1_results']['personalities']
        for personality_id, personality_data in personalities.items():
            self.personality_recommenders[personality_id] = PersonalityRecommender(
                personality_id, personality_data
            )
            
        # Initialize temporal weighting system
        change_points = self.phase3_results['study_2_results'].get('major_shifts', [])
        self.temporal_weighting = TemporalWeightingSystem(change_points)
        
        # Initialize bridge engine
        bridge_candidates = self.phase3_results['study_3_results'].get('top_bridge_songs', [])
        if not bridge_candidates:
            # Create dummy bridge songs if none available
            bridge_candidates = [
                {'track_name': 'Perfect', 'artist_name': 'Ed Sheeran', 'bridge_score': 6.0, 'bridge_reasons': ['cross_cultural_artist']},
                {'track_name': 'Collide', 'artist_name': 'Ed Sheeran', 'bridge_score': 5.5, 'bridge_reasons': ['high_valence']},
                {'track_name': 'Photograph', 'artist_name': 'Cody Fry', 'bridge_score': 5.0, 'bridge_reasons': ['acoustic_content']}
            ]
        self.bridge_engine = CulturalBridgeEngine(bridge_candidates)
        
        logger.info(f"Initialized engine with {len(self.personality_recommenders)} personalities, "
                   f"{len(change_points)} change points, {len(bridge_candidates)} bridge songs")
    
    def _initialize_minimal_components(self):
        """Initialize minimal components for testing/development when Phase 3 results unavailable"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("🔧 Initializing minimal recommendation components...")
        
        # Create basic personality recommenders with default data
        default_personalities = {
            'vietnamese_indie': {
                'top_artists': ['buitruonglinh', 'Den Vau', 'Hoang Thuy Linh'],
                'cultural_profile': {'vietnamese': 0.8, 'western': 0.2},
                'strength': 0.4
            },
            'mixed_cultural': {
                'top_artists': ['Christina Grimmie', 'Taylor Swift', 'Den Vau'],
                'cultural_profile': {'vietnamese': 0.4, 'western': 0.5, 'chinese': 0.1},
                'strength': 0.3
            },
            'western_mixed': {
                'top_artists': ['14 Casper', 'Ed Sheeran', 'Bruno Mars'],
                'cultural_profile': {'vietnamese': 0.1, 'western': 0.9},
                'strength': 0.3
            }
        }
        
        for personality_id, personality_data in default_personalities.items():
            self.personality_recommenders[personality_id] = PersonalityRecommender(
                personality_id, personality_data
            )
        
        # Initialize basic temporal weighting (no change points)
        self.temporal_weighting = TemporalWeightingSystem(change_points=[])
        
        # Initialize basic bridge engine (empty bridges)
        self.bridge_engine = CulturalBridgeEngine(bridge_songs=[])
        
        logger.info("✅ Minimal components initialized - ready for testing")
    
    def create_user_profile(self, recent_listening_data: pd.DataFrame) -> UserProfile:
        """Create dynamic user profile from recent listening behavior"""
        
        if len(recent_listening_data) == 0:
            # Default profile
            return UserProfile(
                personality_weights={'personality_1': 0.4, 'personality_2': 0.3, 'personality_3': 0.3},
                current_preferences={'energy': 0.5, 'valence': 0.5, 'danceability': 0.6},
                cultural_preferences={'vietnamese': 0.5, 'western': 0.3, 'chinese': 0.2},
                recent_change_points=[],
                bridge_song_affinity=0.5,
                temporal_context='stable'
            )
        
        # Calculate personality weights based on recent listening
        personality_weights = {}
        for personality_id, recommender in self.personality_recommenders.items():
            weights = []
            for _, track in recent_listening_data.iterrows():
                track_features = {
                    'energy': track.get('audio_energy', 0.5),
                    'valence': track.get('audio_valence', 0.5),
                    'danceability': track.get('audio_danceability', 0.6),
                    'acousticness': track.get('audio_acousticness', 0.4),
                    'artist_name': track.get('artist_name', ''),
                    'cultural_classification': track.get('dominant_culture', 'unknown')
                }
                score = recommender.calculate_personality_score(track_features)
                weights.append(score)
            personality_weights[personality_id] = np.mean(weights) if weights else 0.33
            
        # Normalize personality weights
        total_weight = sum(personality_weights.values())
        if total_weight > 0:
            personality_weights = {k: v/total_weight for k, v in personality_weights.items()}
        
        # Calculate current audio preferences (use defaults if columns missing)
        audio_features = ['audio_energy', 'audio_valence', 'audio_danceability', 'audio_acousticness']
        current_preferences = {}
        for feature in audio_features:
            clean_feature = feature.replace('audio_', '')
            if feature in recent_listening_data.columns:
                current_preferences[clean_feature] = recent_listening_data[feature].mean()
            else:
                # Default preferences based on common listening patterns
                if clean_feature == 'energy':
                    current_preferences[clean_feature] = 0.6
                elif clean_feature == 'valence':
                    current_preferences[clean_feature] = 0.5
                elif clean_feature == 'danceability':
                    current_preferences[clean_feature] = 0.65
                else:
                    current_preferences[clean_feature] = 0.4
        
        # Calculate cultural preferences (use defaults if column missing)
        if 'dominant_culture' in recent_listening_data.columns:
            cultural_counts = recent_listening_data['dominant_culture'].value_counts(normalize=True)
            cultural_preferences = {
                'vietnamese': cultural_counts.get('vietnamese', 0.4),
                'western': cultural_counts.get('western', 0.3), 
                'chinese': cultural_counts.get('chinese', 0.1),
                'other': cultural_counts.get('unknown', 0.2)
            }
        else:
            # Default cultural preferences based on Phase 3 findings
            cultural_preferences = {
                'vietnamese': 0.52,  # Based on Phase 3 cultural analysis
                'western': 0.25,
                'chinese': 0.05,
                'other': 0.18
            }
        
        # Determine temporal context
        current_time = pd.Timestamp.now(tz='UTC')
        temporal_context = self.temporal_weighting.get_current_preference_context(current_time)
        
        return UserProfile(
            personality_weights=personality_weights,
            current_preferences=current_preferences,
            cultural_preferences=cultural_preferences,
            recent_change_points=[],
            bridge_song_affinity=0.6,  # Default moderate bridge affinity
            temporal_context=temporal_context
        )
    
    def generate_recommendations(
        self,
        user_profile: UserProfile,
        candidate_tracks: pd.DataFrame,
        n_recommendations: int = 10,
        include_bridges: bool = True,
        exploration_factor: float = 0.2
    ) -> List[RecommendationResult]:
        """Generate personalized recommendations"""
        
        recommendations = []
        
        for _, track in candidate_tracks.iterrows():
            # Extract track features
            track_features = {
                'track_name': track.get('track_name', ''),
                'artist_name': track.get('artist_name', ''),
                'energy': track.get('audio_energy', 0.5),
                'valence': track.get('audio_valence', 0.5),
                'danceability': track.get('audio_danceability', 0.6),
                'acousticness': track.get('audio_acousticness', 0.4),
                'speechiness': track.get('audio_speechiness', 0.1),
                'cultural_classification': track.get('dominant_culture', 'unknown')
            }
            
            # Calculate personality scores
            personality_scores = {}
            weighted_personality_score = 0.0
            
            for personality_id, recommender in self.personality_recommenders.items():
                score = recommender.calculate_personality_score(track_features)
                personality_scores[personality_id] = score
                
                weight = user_profile.personality_weights.get(personality_id, 0.33)
                weighted_personality_score += score * weight
            
            # Calculate bridge score
            bridge_score = self.bridge_engine.calculate_bridge_score(track_features)
            
            # Calculate temporal weight (assuming recent track)
            temporal_weight = 0.8  # Default for candidate tracks
            
            # Combine scores
            base_score = weighted_personality_score * 0.6
            bridge_contribution = bridge_score * exploration_factor if include_bridges else 0
            temporal_contribution = temporal_weight * 0.2
            
            final_score = base_score + bridge_contribution + temporal_contribution
            
            # Generate reasoning
            top_personality = max(personality_scores.keys(), key=lambda k: personality_scores[k])
            reasoning_parts = [
                f"Matches {top_personality} ({personality_scores[top_personality]:.2f})"
            ]
            if bridge_score > 0.5:
                reasoning_parts.append(f"Bridge potential ({bridge_score:.2f})")
            if track_features['cultural_classification'] in user_profile.cultural_preferences:
                pref = user_profile.cultural_preferences[track_features['cultural_classification']]
                reasoning_parts.append(f"Cultural fit ({pref:.2f})")
            
            recommendation = RecommendationResult(
                track_id=track.get('track_id', ''),
                track_name=track_features['track_name'],
                artist_name=track_features['artist_name'],
                score=final_score,
                personality_scores=personality_scores,
                bridge_score=bridge_score,
                temporal_weight=temporal_weight,
                cultural_classification=track_features['cultural_classification'],
                reasoning="; ".join(reasoning_parts)
            )
            
            recommendations.append(recommendation)
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:n_recommendations]