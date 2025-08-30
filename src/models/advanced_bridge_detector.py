"""
Advanced Cultural Bridge Detection Models

Sophisticated ML models for identifying songs that bridge Vietnamese and Western music cultures.
Uses audio features, listening patterns, and user behavior to detect cultural gateway songs.

Enhances the existing bridge detection with advanced machine learning approaches.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    import joblib
except ImportError:
    print("Warning: Some ML libraries not available. Install with: pip install scikit-learn")


@dataclass
class BridgeSongCandidate:
    """Candidate bridge song with detection scores"""
    track_id: str
    track_name: str
    artist_name: str
    bridge_probability: float
    cultural_span: float
    transition_effectiveness: float
    novelty_score: float
    audio_bridge_features: Dict[str, float]
    behavioral_evidence: Dict[str, float]
    confidence: float


@dataclass
class CulturalTransition:
    """Represents a transition between cultures in listening history"""
    from_culture: str
    to_culture: str
    transition_track: str
    transition_strength: float
    temporal_context: str
    user_exploration_pattern: float


class AudioFeatureAnalyzer:
    """
    Analyzes audio features to identify characteristics of bridge songs.
    
    Identifies audio patterns that make songs appealing across cultures.
    """
    
    def __init__(self):
        self.bridge_audio_profile = {}
        self.cultural_audio_profiles = {}
        
    def analyze_bridge_audio_patterns(self, streaming_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze audio features of known bridge songs vs cultural-specific songs"""
        
        # Classify songs by culture
        data = streaming_data.copy()
        
        def classify_culture_detailed(artist_name):
            if pd.isna(artist_name):
                return 'unknown'
            artist_lower = str(artist_name).lower()
            vietnamese_indicators = ['buitruonglinh', 'vsoul', 'khói', 'đen', 'mck', 'obito']
            if any(ind in artist_lower for ind in vietnamese_indicators) or \
               any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                return 'vietnamese'
            else:
                return 'western'
        
        data['cultural_class'] = data['artist_name'].apply(classify_culture_detailed)
        
        # Identify potential bridge songs based on cross-cultural listening patterns
        bridge_candidates = self._identify_behavioral_bridges(data)
        
        # Analyze audio features (create synthetic features for demonstration)
        np.random.seed(42)  # For reproducible synthetic features
        unique_tracks = data[['track_id', 'track_name', 'artist_name', 'cultural_class']].drop_duplicates()
        
        # Create synthetic audio features
        audio_features = {
            'energy': np.random.beta(2, 2, len(unique_tracks)),
            'valence': np.random.beta(2, 2, len(unique_tracks)),
            'danceability': np.random.beta(2, 2, len(unique_tracks)),
            'acousticness': np.random.beta(1, 3, len(unique_tracks)),
            'speechiness': np.random.beta(1, 5, len(unique_tracks)),
            'instrumentalness': np.random.beta(1, 4, len(unique_tracks)),
            'liveness': np.random.beta(1, 3, len(unique_tracks)),
            'tempo': np.random.normal(120, 30, len(unique_tracks)),
            'loudness': np.random.normal(-8, 5, len(unique_tracks))
        }
        
        for feature, values in audio_features.items():
            unique_tracks[f'audio_{feature}'] = values
        
        # Mark bridge candidates
        unique_tracks['is_bridge'] = unique_tracks['track_name'].isin(bridge_candidates)
        
        # Analyze audio differences
        audio_cols = [col for col in unique_tracks.columns if col.startswith('audio_')]
        
        # Bridge song audio profile
        bridge_songs = unique_tracks[unique_tracks['is_bridge']]
        if len(bridge_songs) > 0:
            bridge_profile = bridge_songs[audio_cols].mean().to_dict()
        else:
            bridge_profile = {col: 0.5 for col in audio_cols}
        
        # Cultural audio profiles
        vietnamese_profile = unique_tracks[unique_tracks['cultural_class'] == 'vietnamese'][audio_cols].mean().to_dict()
        western_profile = unique_tracks[unique_tracks['cultural_class'] == 'western'][audio_cols].mean().to_dict()
        
        # Calculate bridge characteristics
        bridge_characteristics = {}
        for feature in audio_cols:
            vn_val = vietnamese_profile.get(feature, 0.5)
            west_val = western_profile.get(feature, 0.5)
            bridge_val = bridge_profile.get(feature, 0.5)
            
            # Bridge songs should be intermediate between cultures
            cultural_span = abs(vn_val - west_val)
            bridge_position = abs(bridge_val - (vn_val + west_val) / 2) if cultural_span > 0 else 0
            
            bridge_characteristics[feature] = {
                'vietnamese_avg': vn_val,
                'western_avg': west_val,
                'bridge_avg': bridge_val,
                'cultural_span': cultural_span,
                'bridge_centrality': 1 - (bridge_position / (cultural_span / 2)) if cultural_span > 0 else 1
            }
        
        self.bridge_audio_profile = bridge_profile
        self.cultural_audio_profiles = {
            'vietnamese': vietnamese_profile,
            'western': western_profile
        }
        
        return {
            'bridge_characteristics': bridge_characteristics,
            'bridge_candidates': list(bridge_candidates),
            'audio_profiles': {
                'bridge': bridge_profile,
                'vietnamese': vietnamese_profile,
                'western': western_profile
            }
        }
    
    def _identify_behavioral_bridges(self, data: pd.DataFrame) -> List[str]:
        """Identify bridge songs based on cross-cultural listening behavior"""
        
        # Find songs that are listened to by users who explore both cultures
        user_cultural_diversity = data.groupby('artist_name')['cultural_class'].nunique()
        diverse_artists = user_cultural_diversity[user_cultural_diversity > 1].index
        
        # Songs from artists with cross-cultural appeal
        cross_cultural_songs = data[data['artist_name'].isin(diverse_artists)]['track_name'].unique()
        
        # Songs with high replay value across different time contexts
        track_replay_patterns = data.groupby('track_name').agg({
            'played_at': lambda x: pd.to_datetime(x).dt.hour.nunique(),  # Played across different hours
            'artist_name': 'first',
            'cultural_class': 'first'
        })
        
        # High replay diversity indicates bridge potential
        diverse_replay_songs = track_replay_patterns[track_replay_patterns['played_at'] >= 5]['track_name'].index
        
        # Combine criteria
        bridge_candidates = set(cross_cultural_songs) | set(diverse_replay_songs)
        
        return list(bridge_candidates)[:50]  # Limit to top candidates


class CulturalTransitionDetector:
    """
    Detects patterns in how users transition between cultures in their listening.
    
    Identifies songs that facilitate smooth cultural transitions.
    """
    
    def __init__(self):
        self.transition_patterns = {}
        self.transition_songs = {}
        
    def detect_cultural_transitions(self, streaming_data: pd.DataFrame) -> List[CulturalTransition]:
        """Detect cultural transitions in listening history"""
        
        data = streaming_data.copy()
        data['played_at'] = pd.to_datetime(data['played_at'])
        data = data.sort_values('played_at')
        
        # Classify culture
        def classify_culture(artist_name):
            if pd.isna(artist_name):
                return 'unknown'
            artist_lower = str(artist_name).lower()
            vietnamese_indicators = ['buitruonglinh', 'vsoul', 'khói', 'đen', 'mck', 'obito']
            if any(ind in artist_lower for ind in vietnamese_indicators) or \
               any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                return 'vietnamese'
            else:
                return 'western'
        
        data['cultural_class'] = data['artist_name'].apply(classify_culture)
        
        # Detect transitions
        transitions = []
        prev_culture = None
        culture_run_length = 0
        
        for i, row in data.iterrows():
            current_culture = row['cultural_class']
            
            if current_culture == 'unknown':
                continue
            
            if prev_culture is None:
                prev_culture = current_culture
                culture_run_length = 1
                continue
            
            if current_culture != prev_culture:
                # Culture transition detected
                if culture_run_length >= 3:  # Only consider significant runs
                    
                    # Analyze temporal context
                    hour = row['played_at'].hour
                    if 6 <= hour < 12:
                        temporal_context = 'morning'
                    elif 12 <= hour < 18:
                        temporal_context = 'afternoon'
                    elif 18 <= hour < 22:
                        temporal_context = 'evening'
                    else:
                        temporal_context = 'night'
                    
                    # Calculate transition strength based on cultural distance
                    strength = self._calculate_transition_strength(prev_culture, current_culture, culture_run_length)
                    
                    # User exploration pattern (how often they switch cultures)
                    exploration_pattern = 0.5  # Placeholder - could be calculated from user history
                    
                    transition = CulturalTransition(
                        from_culture=prev_culture,
                        to_culture=current_culture,
                        transition_track=row['track_name'],
                        transition_strength=strength,
                        temporal_context=temporal_context,
                        user_exploration_pattern=exploration_pattern
                    )
                    
                    transitions.append(transition)
                
                prev_culture = current_culture
                culture_run_length = 1
            else:
                culture_run_length += 1
        
        return transitions
    
    def _calculate_transition_strength(self, from_culture: str, to_culture: str, run_length: int) -> float:
        """Calculate strength of cultural transition"""
        
        # Longer runs indicate stronger cultural immersion, making transition more significant
        run_strength = min(run_length / 10.0, 1.0)
        
        # Vietnamese <-> Western transitions are most significant for this dataset
        if (from_culture == 'vietnamese' and to_culture == 'western') or \
           (from_culture == 'western' and to_culture == 'vietnamese'):
            cultural_distance = 1.0
        else:
            cultural_distance = 0.5
        
        return run_strength * cultural_distance
    
    def analyze_transition_patterns(self, transitions: List[CulturalTransition]) -> Dict[str, Any]:
        """Analyze patterns in cultural transitions"""
        
        if not transitions:
            return {}
        
        transition_df = pd.DataFrame([
            {
                'from_culture': t.from_culture,
                'to_culture': t.to_culture,
                'transition_track': t.transition_track,
                'strength': t.transition_strength,
                'temporal_context': t.temporal_context
            }
            for t in transitions
        ])
        
        # Analyze patterns
        patterns = {}
        
        # Most common transition directions
        direction_counts = transition_df.groupby(['from_culture', 'to_culture']).size().sort_values(ascending=False)
        patterns['transition_directions'] = direction_counts.to_dict()
        
        # Temporal patterns
        temporal_patterns = transition_df.groupby('temporal_context')['strength'].mean().sort_values(ascending=False)
        patterns['temporal_preferences'] = temporal_patterns.to_dict()
        
        # Most effective transition songs
        transition_songs = transition_df.groupby('transition_track')['strength'].agg(['count', 'mean'])
        transition_songs = transition_songs[transition_songs['count'] >= 2]  # At least 2 transitions
        effective_songs = transition_songs.sort_values('mean', ascending=False).head(20)
        patterns['effective_transition_songs'] = effective_songs.to_dict('index')
        
        self.transition_patterns = patterns
        return patterns


class AdvancedBridgeDetector:
    """
    Advanced ML model for detecting cultural bridge songs.
    
    Combines audio features, behavioral patterns, and transition analysis.
    """
    
    def __init__(self):
        self.bridge_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=15)
        
        self.audio_analyzer = AudioFeatureAnalyzer()
        self.transition_detector = CulturalTransitionDetector()
        
        self.is_trained = False
        
    def prepare_training_data(self, streaming_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare comprehensive training data for bridge detection"""
        
        # Analyze audio patterns
        audio_analysis = self.audio_analyzer.analyze_bridge_audio_patterns(streaming_data)
        
        # Detect transitions
        transitions = self.transition_detector.detect_cultural_transitions(streaming_data)
        transition_patterns = self.transition_detector.analyze_transition_patterns(transitions)
        
        # Create feature matrix
        unique_tracks = streaming_data[['track_id', 'track_name', 'artist_name']].drop_duplicates()
        
        features = []
        labels = []
        
        # Get known bridge candidates
        bridge_candidates = set(audio_analysis.get('bridge_candidates', []))
        effective_transition_songs = set(transition_patterns.get('effective_transition_songs', {}).keys())
        
        # Combine bridge indicators
        all_bridge_songs = bridge_candidates | effective_transition_songs
        
        for _, track in unique_tracks.iterrows():
            track_name = track['track_name']
            artist_name = track['artist_name']
            
            # Calculate features
            feature_vector = self._calculate_track_features(
                track, streaming_data, audio_analysis, transition_patterns
            )
            
            # Label: 1 if bridge song, 0 otherwise
            is_bridge = 1 if track_name in all_bridge_songs else 0
            
            features.append(feature_vector)
            labels.append(is_bridge)
        
        return np.array(features), np.array(labels)
    
    def _calculate_track_features(
        self, 
        track: pd.Series, 
        streaming_data: pd.DataFrame,
        audio_analysis: Dict[str, Any],
        transition_patterns: Dict[str, Any]
    ) -> List[float]:
        """Calculate comprehensive features for bridge detection"""
        
        track_name = track['track_name']
        artist_name = track['artist_name']
        
        # Get track's listening data
        track_data = streaming_data[streaming_data['track_name'] == track_name]
        
        features = []
        
        # 1. Basic listening statistics
        features.extend([
            len(track_data),  # Total plays
            track_data['artist_name'].nunique() if not track_data.empty else 0,  # Artist variations
            track_data['played_at'].dt.hour.nunique() if not track_data.empty else 0,  # Hour diversity
            track_data['played_at'].dt.dayofweek.nunique() if not track_data.empty else 0,  # Day diversity
        ])
        
        # 2. Synthetic audio features (normalized)
        np.random.seed(hash(track_name) % 2**32)  # Reproducible per track
        audio_features = [
            np.random.beta(2, 2),  # energy
            np.random.beta(2, 2),  # valence
            np.random.beta(2, 2),  # danceability
            np.random.beta(1, 3),  # acousticness
            np.random.beta(1, 5),  # speechiness
            np.random.beta(1, 4),  # instrumentalness
            np.random.beta(1, 3),  # liveness
            (np.random.normal(120, 30) - 60) / 120,  # normalized tempo
            (np.random.normal(-8, 5) + 30) / 30,  # normalized loudness
        ]
        features.extend(audio_features)
        
        # 3. Cultural features
        artist_lower = str(artist_name).lower()
        is_vietnamese = 1 if any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ') else 0
        
        features.extend([
            is_vietnamese,
            1 - is_vietnamese,  # is_western
        ])
        
        # 4. Transition effectiveness features
        transition_songs = transition_patterns.get('effective_transition_songs', {})
        transition_score = 0.0
        if track_name in transition_songs:
            transition_score = transition_songs[track_name].get('mean', 0.0)
        
        features.append(transition_score)
        
        # 5. Audio bridging characteristics
        bridge_characteristics = audio_analysis.get('bridge_characteristics', {})
        audio_bridge_score = 0.0
        
        for i, feature_name in enumerate(['audio_energy', 'audio_valence', 'audio_danceability']):
            if feature_name in bridge_characteristics:
                centrality = bridge_characteristics[feature_name].get('bridge_centrality', 0.5)
                audio_bridge_score += centrality
        
        features.append(audio_bridge_score / 3.0)  # Average centrality
        
        # Ensure we have exactly the expected number of features
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]  # Limit to 20 features
    
    def train(self, streaming_data: pd.DataFrame) -> Dict[str, float]:
        """Train the advanced bridge detection model"""
        
        # Prepare training data
        X, y = self.prepare_training_data(streaming_data)
        
        if len(X) == 0 or sum(y) < 5:  # Need at least 5 positive examples
            return {'error': 'Insufficient training data for bridge detection'}
        
        # Handle class imbalance by balancing the dataset
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == 0)[0]
        
        # Sample negative examples to balance
        n_positive = len(positive_indices)
        n_negative_sample = min(len(negative_indices), n_positive * 3)  # 3:1 ratio
        
        sampled_negative = np.random.choice(negative_indices, n_negative_sample, replace=False)
        balanced_indices = np.concatenate([positive_indices, sampled_negative])
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        # Feature scaling and selection
        X_scaled = self.scaler.fit_transform(X_balanced)
        X_selected = self.feature_selector.fit_transform(X_scaled, y_balanced)
        
        # Train models
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
        )
        
        # Train classifier
        self.bridge_classifier.fit(X_train, y_train)
        
        # Train anomaly detector on positive examples (bridge songs)
        bridge_examples = X_train[y_train == 1]
        if len(bridge_examples) > 5:
            self.anomaly_detector.fit(bridge_examples)
        
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.bridge_classifier.predict(X_test)
        y_pred_proba = self.bridge_classifier.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': np.mean(y_pred == y_test),
            'precision': np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1),
            'recall': np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1),
            'auc_score': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'n_bridge_songs': np.sum(y_balanced == 1),
            'n_total_songs': len(y_balanced)
        }
        
        return metrics
    
    def detect_bridge_songs(
        self, 
        candidate_tracks: pd.DataFrame,
        streaming_data: pd.DataFrame,
        threshold: float = 0.6
    ) -> List[BridgeSongCandidate]:
        """Detect bridge songs from candidate tracks"""
        
        if not self.is_trained:
            return []
        
        # Analyze current patterns
        audio_analysis = self.audio_analyzer.analyze_bridge_audio_patterns(streaming_data)
        transitions = self.transition_detector.detect_cultural_transitions(streaming_data)
        transition_patterns = self.transition_detector.analyze_transition_patterns(transitions)
        
        bridge_candidates = []
        
        for _, track in candidate_tracks.iterrows():
            try:
                # Calculate features
                features = self._calculate_track_features(
                    track, streaming_data, audio_analysis, transition_patterns
                )
                
                # Scale and select features
                features_scaled = self.scaler.transform([features])
                features_selected = self.feature_selector.transform(features_scaled)
                
                # Predict bridge probability
                bridge_probability = self.bridge_classifier.predict_proba(features_selected)[0, 1]
                
                # Calculate additional scores
                cultural_span = self._calculate_cultural_span(track, audio_analysis)
                transition_effectiveness = self._calculate_transition_effectiveness(track, transition_patterns)
                novelty_score = self._calculate_novelty_score(track, streaming_data)
                
                # Confidence based on classifier certainty
                confidence = abs(bridge_probability - 0.5) * 2  # 0 = uncertain, 1 = very certain
                
                if bridge_probability >= threshold:
                    candidate = BridgeSongCandidate(
                        track_id=track.get('track_id', ''),
                        track_name=track.get('track_name', 'Unknown'),
                        artist_name=track.get('artist_name', 'Unknown'),
                        bridge_probability=bridge_probability,
                        cultural_span=cultural_span,
                        transition_effectiveness=transition_effectiveness,
                        novelty_score=novelty_score,
                        audio_bridge_features={
                            'energy_centrality': 0.7,  # Placeholder
                            'valence_appeal': 0.8,
                            'cross_cultural_markers': 0.6
                        },
                        behavioral_evidence={
                            'cross_cultural_plays': 0.7,
                            'transition_frequency': transition_effectiveness,
                            'temporal_diversity': 0.6
                        },
                        confidence=confidence
                    )
                    
                    bridge_candidates.append(candidate)
                    
            except Exception as e:
                continue  # Skip tracks that cause errors
        
        # Sort by bridge probability
        bridge_candidates.sort(key=lambda x: x.bridge_probability, reverse=True)
        return bridge_candidates
    
    def _calculate_cultural_span(self, track: pd.Series, audio_analysis: Dict[str, Any]) -> float:
        """Calculate how well a track spans different cultures"""
        
        # Simplified calculation based on audio centrality
        bridge_characteristics = audio_analysis.get('bridge_characteristics', {})
        
        span_score = 0.0
        count = 0
        
        for feature_name in ['audio_energy', 'audio_valence', 'audio_danceability']:
            if feature_name in bridge_characteristics:
                centrality = bridge_characteristics[feature_name].get('bridge_centrality', 0.5)
                span_score += centrality
                count += 1
        
        return span_score / max(count, 1)
    
    def _calculate_transition_effectiveness(self, track: pd.Series, transition_patterns: Dict[str, Any]) -> float:
        """Calculate how effective a track is for cultural transitions"""
        
        track_name = track['track_name']
        transition_songs = transition_patterns.get('effective_transition_songs', {})
        
        if track_name in transition_songs:
            return transition_songs[track_name].get('mean', 0.0)
        
        return 0.0
    
    def _calculate_novelty_score(self, track: pd.Series, streaming_data: pd.DataFrame) -> float:
        """Calculate novelty score for the track"""
        
        track_name = track['track_name']
        track_plays = streaming_data[streaming_data['track_name'] == track_name]
        
        # Novelty based on play frequency (less played = more novel)
        total_plays = len(streaming_data)
        track_frequency = len(track_plays) / total_plays if total_plays > 0 else 0
        
        # Convert to novelty score (higher frequency = lower novelty)
        novelty = 1 - min(track_frequency * 100, 1.0)  # Scale down frequency
        
        return novelty


def create_bridge_detection_suite(streaming_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create and train a complete suite of bridge detection models.
    
    Returns trained models ready for bridge song detection.
    """
    
    print("🌉 Creating Advanced Bridge Detection Suite...")
    
    models = {}
    
    # 1. Advanced Bridge Detector
    print("🤖 Training Advanced Bridge Detector...")
    bridge_detector = AdvancedBridgeDetector()
    bridge_metrics = bridge_detector.train(streaming_data)
    models['bridge_detector'] = {
        'model': bridge_detector,
        'metrics': bridge_metrics,
        'description': 'Advanced ML model for cultural bridge song detection'
    }
    
    # 2. Audio Feature Analyzer
    print("🎵 Analyzing Audio Bridge Patterns...")
    audio_analyzer = AudioFeatureAnalyzer()
    audio_analysis = audio_analyzer.analyze_bridge_audio_patterns(streaming_data)
    models['audio_analyzer'] = {
        'model': audio_analyzer,
        'analysis': audio_analysis,
        'description': 'Audio feature analysis for bridge song characteristics'
    }
    
    # 3. Cultural Transition Detector
    print("🔄 Detecting Cultural Transitions...")
    transition_detector = CulturalTransitionDetector()
    transitions = transition_detector.detect_cultural_transitions(streaming_data)
    transition_patterns = transition_detector.analyze_transition_patterns(transitions)
    models['transition_detector'] = {
        'model': transition_detector,
        'transitions': transitions,
        'patterns': transition_patterns,
        'description': 'Cultural transition pattern analysis'
    }
    
    print("✅ Bridge Detection Suite Created!")
    print(f"🎯 Bridge Detection Accuracy: {bridge_metrics.get('accuracy', 0):.3f}")
    print(f"🔍 Detected {len(transitions)} cultural transitions")
    print(f"🎵 Analyzed {len(audio_analysis.get('bridge_candidates', []))} bridge candidates")
    
    return models


if __name__ == "__main__":
    # Example usage
    print("🌉 Advanced Cultural Bridge Detection")
    print("=" * 40)
    
    try:
        # Load sample data
        streaming_data = pd.read_parquet('../../data/processed/streaming_data_processed.parquet')
        
        # Create bridge detection suite
        models = create_bridge_detection_suite(streaming_data)
        
        # Example bridge detection
        bridge_detector = models['bridge_detector']['model']
        
        # Sample candidate tracks
        candidate_tracks = streaming_data[['track_id', 'track_name', 'artist_name']].drop_duplicates().head(100)
        
        # Detect bridge songs
        bridge_songs = bridge_detector.detect_bridge_songs(
            candidate_tracks, streaming_data, threshold=0.7
        )
        
        print(f"\n🌉 Detected {len(bridge_songs)} Bridge Songs:")
        for i, bridge in enumerate(bridge_songs[:10], 1):
            print(f"   {i}. {bridge.track_name} - {bridge.artist_name}")
            print(f"      Bridge Probability: {bridge.bridge_probability:.3f}")
            print(f"      Cultural Span: {bridge.cultural_span:.3f}")
            print(f"      Confidence: {bridge.confidence:.3f}")
        
        # Analyze transitions
        transitions = models['transition_detector']['transitions']
        print(f"\n🔄 Cultural Transitions Detected: {len(transitions)}")
        
        transition_patterns = models['transition_detector']['patterns']
        if 'transition_directions' in transition_patterns:
            print("   Most Common Transitions:")
            for (from_c, to_c), count in list(transition_patterns['transition_directions'].items())[:5]:
                print(f"     {from_c} → {to_c}: {count} times")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure data files are available for testing.")
