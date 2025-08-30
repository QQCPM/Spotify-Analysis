"""
Context-Aware Music Recommendation Models

Advanced models that understand listening context, time-of-day preferences,
and session-based patterns for highly personalized recommendations.

Leverages temporal insights to provide contextually relevant music suggestions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.cluster import KMeans
    import joblib
except ImportError:
    print("Warning: Some ML libraries not available. Install with: pip install scikit-learn")


@dataclass
class ContextualRecommendation:
    """Result from context-aware recommendation"""
    track_id: str
    track_name: str
    artist_name: str
    context_score: float
    time_appropriateness: float
    mood_match: float
    cultural_fit: float
    reasoning: str


@dataclass 
class ListeningContext:
    """Represents current listening context"""
    current_time: datetime
    is_weekend: bool
    time_of_day: str  # morning, afternoon, evening, night
    work_hours: bool
    recent_cultural_preference: float  # Vietnamese ratio
    session_type: str  # new_session, continuation
    energy_preference: float  # 0-1 scale


class TemporalContextAnalyzer:
    """
    Analyzes listening patterns to understand contextual preferences.
    
    Identifies when specific artists, genres, or cultural music is preferred.
    """
    
    def __init__(self):
        self.context_patterns = {}
        self.cultural_time_preferences = {}
        self.artist_context_profiles = {}
        
    def analyze_temporal_contexts(self, streaming_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal listening contexts from historical data"""
        
        data = streaming_data.copy()
        data['played_at'] = pd.to_datetime(data['played_at'])
        
        # Extract context features
        data['hour'] = data['played_at'].dt.hour
        data['day_of_week'] = data['played_at'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Time of day classification
        def classify_time_of_day(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 22:
                return 'evening'
            else:
                return 'night'
        
        data['time_of_day'] = data['hour'].apply(classify_time_of_day)
        data['work_hours'] = ((data['day_of_week'] < 5) & (data['hour'] >= 9) & (data['hour'] < 17)).astype(int)
        
        # Cultural classification
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
        
        # Analyze context patterns
        context_analysis = {}
        
        # 1. Time of day preferences
        time_preferences = data.groupby(['time_of_day', 'cultural_class']).size().unstack(fill_value=0)
        time_preferences_norm = time_preferences.div(time_preferences.sum(axis=1), axis=0)
        
        context_analysis['time_cultural_preferences'] = time_preferences_norm.to_dict()
        
        # 2. Work vs leisure preferences
        work_preferences = data.groupby(['work_hours', 'cultural_class']).size().unstack(fill_value=0)
        work_preferences_norm = work_preferences.div(work_preferences.sum(axis=1), axis=0)
        
        context_analysis['work_cultural_preferences'] = work_preferences_norm.to_dict()
        
        # 3. Weekend vs weekday patterns
        weekend_preferences = data.groupby(['is_weekend', 'cultural_class']).size().unstack(fill_value=0)
        weekend_preferences_norm = weekend_preferences.div(weekend_preferences.sum(axis=1), axis=0)
        
        context_analysis['weekend_cultural_preferences'] = weekend_preferences_norm.to_dict()
        
        # 4. Artist-specific temporal patterns
        top_artists = data['artist_name'].value_counts().head(50).index
        artist_patterns = {}
        
        for artist in top_artists:
            artist_data = data[data['artist_name'] == artist]
            
            if len(artist_data) > 10:  # Only analyze artists with sufficient data
                patterns = {
                    'favorite_time': artist_data['time_of_day'].mode().iloc[0] if not artist_data['time_of_day'].mode().empty else 'evening',
                    'favorite_hour': artist_data['hour'].mode().iloc[0] if not artist_data['hour'].mode().empty else 19,
                    'weekend_preference': artist_data['is_weekend'].mean(),
                    'work_hours_ratio': artist_data['work_hours'].mean(),
                    'cultural_class': artist_data['cultural_class'].iloc[0],
                    'total_plays': len(artist_data)
                }
                artist_patterns[artist] = patterns
        
        context_analysis['artist_temporal_patterns'] = artist_patterns
        
        # 5. Hourly activity patterns
        hourly_activity = data.groupby('hour').agg({
            'track_id': 'count',
            'cultural_class': lambda x: (x == 'vietnamese').mean()
        }).round(3)
        
        context_analysis['hourly_patterns'] = {
            'activity_by_hour': hourly_activity['track_id'].to_dict(),
            'vietnamese_ratio_by_hour': hourly_activity['cultural_class'].to_dict()
        }
        
        self.context_patterns = context_analysis
        return context_analysis
    
    def get_context_preferences(self, context: ListeningContext) -> Dict[str, float]:
        """Get cultural preferences for a specific context"""
        
        if not self.context_patterns:
            return {'vietnamese': 0.5, 'western': 0.5}  # Default
        
        # Get time-based preferences
        time_prefs = self.context_patterns.get('time_cultural_preferences', {})
        time_cultural_prefs = time_prefs.get(context.time_of_day, {'vietnamese': 0.5, 'western': 0.5})
        
        # Get work/leisure preferences
        work_prefs = self.context_patterns.get('work_cultural_preferences', {})
        work_cultural_prefs = work_prefs.get(int(context.work_hours), {'vietnamese': 0.5, 'western': 0.5})
        
        # Get weekend preferences
        weekend_prefs = self.context_patterns.get('weekend_cultural_preferences', {})
        weekend_cultural_prefs = weekend_prefs.get(int(context.is_weekend), {'vietnamese': 0.5, 'western': 0.5})
        
        # Combine preferences (weighted average)
        combined_prefs = {}
        for culture in ['vietnamese', 'western']:
            combined_prefs[culture] = (
                time_cultural_prefs.get(culture, 0.5) * 0.4 +
                work_cultural_prefs.get(culture, 0.5) * 0.3 +
                weekend_cultural_prefs.get(culture, 0.5) * 0.3
            )
        
        return combined_prefs


class SessionBasedRecommender:
    """
    Provides recommendations based on current listening session context.
    
    Understands session patterns and suggests music that fits the flow.
    """
    
    def __init__(self):
        self.session_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.context_analyzer = TemporalContextAnalyzer()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def prepare_session_data(self, streaming_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare session-based training data"""
        
        data = streaming_data.copy()
        data['played_at'] = pd.to_datetime(data['played_at'])
        
        # Create sessions
        data = data.sort_values('played_at')
        data['time_gap'] = data['played_at'].diff().dt.total_seconds() / 60  # minutes
        data['new_session'] = (data['time_gap'] > 30) | (data['time_gap'].isna())
        data['session_id'] = data['new_session'].cumsum()
        
        # Extract features for each track within sessions
        features = []
        labels = []
        
        for session_id in data['session_id'].unique():
            session_data = data[data['session_id'] == session_id]
            
            if len(session_data) < 3:  # Skip very short sessions
                continue
            
            for i, (_, track) in enumerate(session_data.iterrows()):
                # Features: context + position in session
                hour = track['played_at'].hour
                day_of_week = track['played_at'].dayofweek
                is_weekend = 1 if day_of_week >= 5 else 0
                session_position = i / len(session_data)  # Relative position in session
                
                # Time of day encoding
                time_of_day_encoding = {
                    'morning': [1, 0, 0, 0],
                    'afternoon': [0, 1, 0, 0],
                    'evening': [0, 0, 1, 0],
                    'night': [0, 0, 0, 1]
                }
                
                if 6 <= hour < 12:
                    time_encoding = time_of_day_encoding['morning']
                elif 12 <= hour < 18:
                    time_encoding = time_of_day_encoding['afternoon']
                elif 18 <= hour < 22:
                    time_encoding = time_of_day_encoding['evening']
                else:
                    time_encoding = time_of_day_encoding['night']
                
                # Cultural classification
                artist_lower = str(track['artist_name']).lower()
                is_vietnamese = 1 if any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ') else 0
                
                feature_vector = [
                    hour / 24.0,  # Normalized hour
                    day_of_week / 6.0,  # Normalized day
                    is_weekend,
                    session_position,
                    len(session_data),  # Session length
                    is_vietnamese
                ] + time_encoding
                
                features.append(feature_vector)
                labels.append(track['artist_name'])  # Predict artist as proxy for preference
        
        X = np.array(features)
        
        # Encode labels (artists)
        unique_artists = list(set(labels))
        if len(unique_artists) > 100:  # Limit to top artists for classification
            artist_counts = pd.Series(labels).value_counts()
            top_artists = artist_counts.head(50).index.tolist()
            
            # Filter to only top artists
            filtered_features = []
            filtered_labels = []
            
            for feature, label in zip(features, labels):
                if label in top_artists:
                    filtered_features.append(feature)
                    filtered_labels.append(label)
            
            X = np.array(filtered_features)
            labels = filtered_labels
        
        y = self.label_encoder.fit_transform(labels)
        
        return X, y
    
    def train(self, streaming_data: pd.DataFrame) -> Dict[str, float]:
        """Train the session-based recommender"""
        
        # Analyze temporal contexts
        self.context_analyzer.analyze_temporal_contexts(streaming_data)
        
        # Prepare session data
        X, y = self.prepare_session_data(streaming_data)
        
        if len(X) == 0:
            return {'error': 'No valid training data'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.session_classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.session_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.session_classifier, X_scaled, y, cv=5)
        
        metrics = {
            'accuracy': accuracy,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_classes': len(self.label_encoder.classes_),
            'n_samples': len(X)
        }
        
        return metrics
    
    def get_contextual_recommendations(
        self,
        context: ListeningContext,
        candidate_tracks: pd.DataFrame,
        n_recommendations: int = 10
    ) -> List[ContextualRecommendation]:
        """Get context-aware recommendations"""
        
        if not self.is_trained:
            return []
        
        # Get context preferences
        context_prefs = self.context_analyzer.get_context_preferences(context)
        
        recommendations = []
        
        for _, track in candidate_tracks.iterrows():
            # Calculate context score
            context_score = self._calculate_context_score(track, context, context_prefs)
            
            # Time appropriateness
            time_appropriateness = self._calculate_time_appropriateness(track, context)
            
            # Cultural fit
            cultural_fit = self._calculate_cultural_fit(track, context_prefs)
            
            # Overall scoring
            overall_score = (
                context_score * 0.4 +
                time_appropriateness * 0.3 +
                cultural_fit * 0.3
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(track, context, context_prefs)
            
            recommendation = ContextualRecommendation(
                track_id=track.get('track_id', ''),
                track_name=track.get('track_name', 'Unknown'),
                artist_name=track.get('artist_name', 'Unknown'),
                context_score=overall_score,
                time_appropriateness=time_appropriateness,
                mood_match=0.7,  # Placeholder - could be enhanced with audio features
                cultural_fit=cultural_fit,
                reasoning=reasoning
            )
            
            recommendations.append(recommendation)
        
        # Sort by context score and return top N
        recommendations.sort(key=lambda x: x.context_score, reverse=True)
        return recommendations[:n_recommendations]
    
    def _calculate_context_score(self, track: pd.Series, context: ListeningContext, context_prefs: Dict[str, float]) -> float:
        """Calculate how well a track fits the current context"""
        
        # Get artist patterns if available
        artist_patterns = self.context_analyzer.context_patterns.get('artist_temporal_patterns', {})
        artist_name = track.get('artist_name', '')
        
        if artist_name in artist_patterns:
            patterns = artist_patterns[artist_name]
            
            # Time match
            time_match = 1.0 if patterns['favorite_time'] == context.time_of_day else 0.5
            
            # Weekend preference match
            weekend_match = abs(patterns['weekend_preference'] - (1 if context.is_weekend else 0))
            weekend_match = 1 - weekend_match  # Convert to similarity
            
            # Work hours match
            work_match = abs(patterns['work_hours_ratio'] - (1 if context.work_hours else 0))
            work_match = 1 - work_match
            
            return (time_match * 0.4 + weekend_match * 0.3 + work_match * 0.3)
        
        return 0.5  # Default score for unknown artists
    
    def _calculate_time_appropriateness(self, track: pd.Series, context: ListeningContext) -> float:
        """Calculate how appropriate a track is for the current time"""
        
        # Get hourly patterns
        hourly_patterns = self.context_analyzer.context_patterns.get('hourly_patterns', {})
        activity_by_hour = hourly_patterns.get('activity_by_hour', {})
        
        current_hour = context.current_time.hour
        
        if current_hour in activity_by_hour:
            # Normalize activity level (higher activity = more appropriate time)
            max_activity = max(activity_by_hour.values())
            current_activity = activity_by_hour[current_hour]
            return current_activity / max_activity if max_activity > 0 else 0.5
        
        return 0.5  # Default
    
    def _calculate_cultural_fit(self, track: pd.Series, context_prefs: Dict[str, float]) -> float:
        """Calculate how well track's culture fits current preferences"""
        
        artist_name = str(track.get('artist_name', '')).lower()
        
        # Classify track culture
        if any(char in artist_name for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
            track_culture = 'vietnamese'
        else:
            track_culture = 'western'
        
        return context_prefs.get(track_culture, 0.5)
    
    def _generate_reasoning(self, track: pd.Series, context: ListeningContext, context_prefs: Dict[str, float]) -> str:
        """Generate human-readable reasoning for recommendation"""
        
        reasons = []
        
        # Time-based reasoning
        if context.time_of_day == 'morning':
            reasons.append("Good morning energy")
        elif context.time_of_day == 'evening':
            reasons.append("Perfect for evening relaxation")
        elif context.time_of_day == 'night':
            reasons.append("Suitable for late-night listening")
        
        # Weekend reasoning
        if context.is_weekend:
            reasons.append("Weekend vibe")
        elif context.work_hours:
            reasons.append("Work-friendly choice")
        
        # Cultural reasoning
        artist_name = str(track.get('artist_name', '')).lower()
        if any(char in artist_name for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
            if context_prefs.get('vietnamese', 0.5) > 0.6:
                reasons.append("Matches current Vietnamese preference")
        else:
            if context_prefs.get('western', 0.5) > 0.6:
                reasons.append("Fits Western music preference")
        
        return "; ".join(reasons) if reasons else "Contextually appropriate"


class MoodBasedRecommender:
    """
    Recommends music based on inferred mood from temporal patterns.
    
    Uses time patterns and listening history to infer mood and suggest appropriate music.
    """
    
    def __init__(self):
        self.mood_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.mood_clusters = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_mood_features(self, streaming_data: pd.DataFrame) -> pd.DataFrame:
        """Create features that correlate with different moods"""
        
        data = streaming_data.copy()
        data['played_at'] = pd.to_datetime(data['played_at'])
        
        # Create sessions
        data = data.sort_values('played_at')
        data['time_gap'] = data['played_at'].diff().dt.total_seconds() / 60
        data['new_session'] = (data['time_gap'] > 30) | (data['time_gap'].isna())
        data['session_id'] = data['new_session'].cumsum()
        
        # Session-based mood features
        session_features = []
        
        for session_id in data['session_id'].unique():
            session_data = data[data['session_id'] == session_id]
            
            if len(session_data) < 3:  # Skip short sessions
                continue
            
            # Temporal features
            start_time = session_data['played_at'].min()
            hour = start_time.hour
            day_of_week = start_time.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Session characteristics
            session_length = len(session_data)
            session_duration = (session_data['played_at'].max() - session_data['played_at'].min()).total_seconds() / 3600
            
            # Cultural diversity in session
            unique_cultures = session_data['artist_name'].apply(
                lambda x: 'vietnamese' if any(char in str(x).lower() for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ') else 'western'
            ).nunique()
            
            # Artist repetition (might indicate focused mood)
            artist_repetition = session_data['artist_name'].value_counts().max() / len(session_data)
            
            # Infer mood label from patterns (simplified)
            mood_label = self._infer_mood_from_patterns(hour, is_weekend, session_length, artist_repetition)
            
            session_features.append({
                'session_id': session_id,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'session_length': session_length,
                'session_duration': session_duration,
                'cultural_diversity': unique_cultures,
                'artist_repetition': artist_repetition,
                'mood_label': mood_label
            })
        
        return pd.DataFrame(session_features)
    
    def _infer_mood_from_patterns(self, hour: int, is_weekend: int, session_length: int, artist_repetition: float) -> str:
        """Infer mood from listening patterns (simplified heuristic)"""
        
        # Morning patterns
        if 6 <= hour < 12:
            if session_length > 20:
                return 'energetic'
            else:
                return 'calm'
        
        # Work hours
        elif 9 <= hour < 17 and not is_weekend:
            if artist_repetition > 0.5:
                return 'focused'
            else:
                return 'background'
        
        # Evening patterns
        elif 18 <= hour < 22:
            if is_weekend and session_length > 30:
                return 'social'
            else:
                return 'relaxed'
        
        # Night patterns
        else:
            if artist_repetition > 0.6:
                return 'contemplative'
            else:
                return 'chill'
    
    def train(self, streaming_data: pd.DataFrame) -> Dict[str, float]:
        """Train mood-based recommendation model"""
        
        # Create mood features
        mood_data = self.create_mood_features(streaming_data)
        
        if len(mood_data) < 10:
            return {'error': 'Insufficient data for mood modeling'}
        
        # Prepare features
        feature_cols = ['hour', 'day_of_week', 'is_weekend', 'session_length', 
                       'session_duration', 'cultural_diversity', 'artist_repetition']
        
        X = mood_data[feature_cols].fillna(0).values
        y = mood_data['mood_label'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train mood classifier
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.mood_classifier.fit(X_train, y_train)
        
        # Train mood clusters for pattern discovery
        self.mood_clusters.fit(X_scaled)
        
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.mood_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'n_moods': len(np.unique(y)),
            'n_sessions': len(mood_data)
        }
        
        return metrics
    
    def predict_mood(self, context: ListeningContext) -> str:
        """Predict current mood based on context"""
        
        if not self.is_trained:
            return 'unknown'
        
        # Create feature vector for current context
        features = np.array([[
            context.current_time.hour,
            context.current_time.weekday(),
            int(context.is_weekend),
            15,  # Estimated session length
            1.0,  # Estimated session duration
            2,    # Estimated cultural diversity
            0.3   # Estimated artist repetition
        ]])
        
        features_scaled = self.scaler.transform(features)
        predicted_mood = self.mood_classifier.predict(features_scaled)[0]
        
        return predicted_mood


def create_context_models_suite(streaming_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create and train a complete suite of context-aware models.
    
    Returns trained models ready for contextual recommendations.
    """
    
    print("🎭 Creating Context-Aware Models Suite...")
    
    models = {}
    
    # 1. Session-Based Recommender
    print("🎧 Training Session-Based Recommender...")
    session_recommender = SessionBasedRecommender()
    session_metrics = session_recommender.train(streaming_data)
    models['session_recommender'] = {
        'model': session_recommender,
        'metrics': session_metrics,
        'description': 'Context-aware recommendations based on listening sessions'
    }
    
    # 2. Mood-Based Recommender
    print("😊 Training Mood-Based Recommender...")
    mood_recommender = MoodBasedRecommender()
    mood_metrics = mood_recommender.train(streaming_data)
    models['mood_recommender'] = {
        'model': mood_recommender,
        'metrics': mood_metrics,
        'description': 'Recommendations based on inferred mood from patterns'
    }
    
    print("✅ Context-Aware Models Suite Created!")
    print(f"🎧 Session Accuracy: {session_metrics.get('accuracy', 0):.3f}")
    print(f"😊 Mood Accuracy: {mood_metrics.get('accuracy', 0):.3f}")
    
    return models


if __name__ == "__main__":
    # Example usage
    print("🎭 Context-Aware Recommendation Models")
    print("=" * 45)
    
    try:
        # Load sample data
        streaming_data = pd.read_parquet('../../data/processed/streaming_data_processed.parquet')
        
        # Create models suite
        models = create_context_models_suite(streaming_data)
        
        # Example context
        current_context = ListeningContext(
            current_time=datetime.now(),
            is_weekend=datetime.now().weekday() >= 5,
            time_of_day='evening',
            work_hours=False,
            recent_cultural_preference=0.6,
            session_type='new_session',
            energy_preference=0.7
        )
        
        # Get contextual recommendations
        session_model = models['session_recommender']['model']
        mood_model = models['mood_recommender']['model']
        
        # Predict current mood
        predicted_mood = mood_model.predict_mood(current_context)
        print(f"\n🎭 Predicted Mood: {predicted_mood}")
        
        # Sample candidate tracks (in real use, this would come from your music database)
        candidate_tracks = streaming_data[['track_id', 'track_name', 'artist_name']].drop_duplicates().head(20)
        
        # Get contextual recommendations
        recommendations = session_model.get_contextual_recommendations(
            current_context, candidate_tracks, n_recommendations=5
        )
        
        print(f"\n🎵 Contextual Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec.track_name} - {rec.artist_name}")
            print(f"      Score: {rec.context_score:.3f} | {rec.reasoning}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure data files are available for testing.")
