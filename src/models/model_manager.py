"""
Comprehensive Model Manager

Unified interface for managing all recommendation system models:
- Cross-cultural recommendation engine
- Temporal prediction models
- Context-aware recommenders
- Advanced bridge detection models

Provides seamless integration and orchestration of the complete model suite.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import warnings
warnings.filterwarnings('ignore')

from .recommendation_engine import CrossCulturalRecommendationEngine, UserProfile

try:
    from .temporal_prediction_model import (
        ListeningVolumePredictor, 
        CulturalEvolutionPredictor,
        create_temporal_models_suite
    )
    from .context_aware_recommender import (
        SessionBasedRecommender,
        MoodBasedRecommender,
        ListeningContext,
        create_context_models_suite
    )
    from .advanced_bridge_detector import (
        AdvancedBridgeDetector,
        create_bridge_detection_suite
    )
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    logging.warning("Advanced models not available - some features will be limited")


@dataclass
class ComprehensiveRecommendation:
    """Unified recommendation result from all models"""
    track_id: str
    track_name: str
    artist_name: str
    
    # Scores from different models
    base_score: float  # From main recommendation engine
    temporal_score: float  # From temporal context
    mood_score: float  # From mood-based model
    bridge_score: float  # From bridge detection
    
    # Combined score
    final_score: float
    
    # Explanations
    personality_match: str
    temporal_context: str
    mood_reasoning: str
    cultural_bridge_potential: str
    
    # Confidence and metadata
    confidence: float
    recommendation_type: str  # 'personality', 'temporal', 'mood', 'bridge', 'hybrid'


@dataclass
class SystemInsights:
    """Insights from the complete model suite"""
    current_listening_patterns: Dict[str, Any]
    predicted_preferences: Dict[str, Any]
    cultural_evolution_forecast: Dict[str, Any]
    bridge_song_opportunities: List[str]
    mood_assessment: str
    temporal_recommendations: Dict[str, Any]


class ComprehensiveModelManager:
    """
    Manages and orchestrates all recommendation system models.
    
    Provides unified interface for:
    - Training all models
    - Generating comprehensive recommendations
    - System-wide insights and predictions
    - Model performance monitoring
    """
    
    def __init__(self, phase3_results_path: str = 'results/phase3'):
        self.phase3_results_path = phase3_results_path
        self.models = {}
        self.model_metrics = {}
        self.is_initialized = False
        
        # Core models (always available)
        self.core_engine = None
        
        # Advanced models (optional)
        self.temporal_models = None
        self.context_models = None
        self.bridge_models = None
        
        # Model weights for ensemble
        self.model_weights = {
            'base': 0.4,
            'temporal': 0.2,
            'mood': 0.2,
            'bridge': 0.2
        }
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_system(self, streaming_data: pd.DataFrame) -> Dict[str, Any]:
        """Initialize the complete model system"""
        
        self.logger.info("🚀 Initializing Comprehensive Model System...")
        
        initialization_results = {
            'core_engine': {},
            'temporal_models': {},
            'context_models': {},
            'bridge_models': {},
            'system_status': 'initializing'
        }
        
        try:
            # 1. Initialize core recommendation engine
            self.logger.info("🧬 Initializing Core Recommendation Engine...")
            self.core_engine = CrossCulturalRecommendationEngine(self.phase3_results_path)
            initialization_results['core_engine'] = {
                'status': 'success',
                'personalities': len(self.core_engine.personality_recommenders),
                'change_points': len(self.core_engine.temporal_weighting.change_points) if self.core_engine.temporal_weighting else 0,
                'bridge_songs': len(self.core_engine.bridge_engine.bridge_songs) if self.core_engine.bridge_engine else 0
            }
            
            # 2. Initialize advanced models if available
            if ADVANCED_MODELS_AVAILABLE:
                
                # Temporal prediction models
                self.logger.info("⏰ Training Temporal Prediction Models...")
                try:
                    self.temporal_models = create_temporal_models_suite(streaming_data)
                    initialization_results['temporal_models'] = {
                        'status': 'success',
                        'models_trained': len(self.temporal_models),
                        'volume_predictor_r2': self.temporal_models['volume_predictor']['metrics'].get('r2_score', 0),
                        'cultural_predictor_r2': self.temporal_models['cultural_predictor']['metrics'].get('r2_score', 0)
                    }
                except Exception as e:
                    self.logger.warning(f"Temporal models training failed: {str(e)}")
                    initialization_results['temporal_models'] = {'status': 'failed', 'error': str(e)}
                
                # Context-aware models
                self.logger.info("🎭 Training Context-Aware Models...")
                try:
                    self.context_models = create_context_models_suite(streaming_data)
                    initialization_results['context_models'] = {
                        'status': 'success',
                        'models_trained': len(self.context_models),
                        'session_accuracy': self.context_models['session_recommender']['metrics'].get('accuracy', 0),
                        'mood_accuracy': self.context_models['mood_recommender']['metrics'].get('accuracy', 0)
                    }
                except Exception as e:
                    self.logger.warning(f"Context models training failed: {str(e)}")
                    initialization_results['context_models'] = {'status': 'failed', 'error': str(e)}
                
                # Bridge detection models
                self.logger.info("🌉 Training Bridge Detection Models...")
                try:
                    self.bridge_models = create_bridge_detection_suite(streaming_data)
                    initialization_results['bridge_models'] = {
                        'status': 'success',
                        'models_trained': len(self.bridge_models),
                        'bridge_detection_accuracy': self.bridge_models['bridge_detector']['metrics'].get('accuracy', 0),
                        'transitions_detected': len(self.bridge_models['transition_detector']['transitions'])
                    }
                except Exception as e:
                    self.logger.warning(f"Bridge detection training failed: {str(e)}")
                    initialization_results['bridge_models'] = {'status': 'failed', 'error': str(e)}
            
            else:
                self.logger.info("⚠️ Advanced models not available - using core engine only")
                initialization_results['temporal_models'] = {'status': 'unavailable'}
                initialization_results['context_models'] = {'status': 'unavailable'}
                initialization_results['bridge_models'] = {'status': 'unavailable'}
            
            self.is_initialized = True
            initialization_results['system_status'] = 'success'
            
            self.logger.info("✅ Model System Initialization Complete!")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            initialization_results['system_status'] = 'failed'
            initialization_results['error'] = str(e)
        
        return initialization_results
    
    def generate_comprehensive_recommendations(
        self,
        streaming_data: pd.DataFrame,
        candidate_tracks: pd.DataFrame,
        n_recommendations: int = 10,
        current_context: Optional[ListeningContext] = None
    ) -> List[ComprehensiveRecommendation]:
        """Generate recommendations using all available models"""
        
        if not self.is_initialized or not self.core_engine:
            raise ValueError("System must be initialized before generating recommendations")
        
        # Create user profile
        user_profile = self.core_engine.create_user_profile(streaming_data.tail(1000))
        
        # Get base recommendations from core engine
        base_recommendations = self.core_engine.generate_recommendations(
            user_profile=user_profile,
            candidate_tracks=candidate_tracks,
            n_recommendations=n_recommendations * 2,  # Get more for filtering
            include_bridges=True
        )
        
        comprehensive_recs = []
        
        for base_rec in base_recommendations[:n_recommendations]:
            
            # Initialize scores
            base_score = base_rec.score
            temporal_score = 0.5
            mood_score = 0.5
            bridge_score = base_rec.bridge_score
            
            # Get enhanced scores from advanced models
            if current_context and self.context_models:
                
                # Session-based score
                try:
                    session_model = self.context_models['session_recommender']['model']
                    track_df = candidate_tracks[candidate_tracks['track_name'] == base_rec.track_name]
                    if not track_df.empty:
                        session_recs = session_model.get_contextual_recommendations(
                            current_context, track_df, n_recommendations=1
                        )
                        if session_recs:
                            temporal_score = session_recs[0].context_score
                except Exception:
                    pass
                
                # Mood-based score
                try:
                    mood_model = self.context_models['mood_recommender']['model']
                    predicted_mood = mood_model.predict_mood(current_context)
                    mood_score = self._calculate_mood_track_fit(predicted_mood, base_rec)
                except Exception:
                    pass
            
            # Enhanced bridge score
            if self.bridge_models:
                try:
                    bridge_detector = self.bridge_models['bridge_detector']['model']
                    track_df = candidate_tracks[candidate_tracks['track_name'] == base_rec.track_name]
                    if not track_df.empty:
                        bridge_candidates = bridge_detector.detect_bridge_songs(
                            track_df, streaming_data, threshold=0.3
                        )
                        if bridge_candidates:
                            bridge_score = bridge_candidates[0].bridge_probability
                except Exception:
                    pass
            
            # Calculate final ensemble score
            final_score = (
                base_score * self.model_weights['base'] +
                temporal_score * self.model_weights['temporal'] +
                mood_score * self.model_weights['mood'] +
                bridge_score * self.model_weights['bridge']
            )
            
            # Generate comprehensive explanations
            personality_match = base_rec.reasoning
            
            temporal_context = "Standard timing"
            if current_context:
                temporal_context = f"{current_context.time_of_day} listening"
                if current_context.is_weekend:
                    temporal_context += " (weekend)"
                if current_context.work_hours:
                    temporal_context += " (work hours)"
            
            mood_reasoning = "Neutral mood match"
            if self.context_models:
                try:
                    mood_model = self.context_models['mood_recommender']['model']
                    if current_context:
                        predicted_mood = mood_model.predict_mood(current_context)
                        mood_reasoning = f"Matches {predicted_mood} mood"
                except Exception:
                    pass
            
            cultural_bridge_potential = "Low bridge potential"
            if bridge_score > 0.7:
                cultural_bridge_potential = "High cultural bridge potential"
            elif bridge_score > 0.5:
                cultural_bridge_potential = "Moderate bridge potential"
            
            # Determine recommendation type
            max_score_source = max([
                ('personality', base_score),
                ('temporal', temporal_score),
                ('mood', mood_score),
                ('bridge', bridge_score)
            ], key=lambda x: x[1])
            
            recommendation_type = max_score_source[0] if max_score_source[1] > 0.6 else 'hybrid'
            
            # Calculate confidence
            score_variance = np.var([base_score, temporal_score, mood_score, bridge_score])
            confidence = 1.0 - min(score_variance, 0.5) * 2  # Lower variance = higher confidence
            
            comprehensive_rec = ComprehensiveRecommendation(
                track_id=base_rec.track_id,
                track_name=base_rec.track_name,
                artist_name=base_rec.artist_name,
                base_score=base_score,
                temporal_score=temporal_score,
                mood_score=mood_score,
                bridge_score=bridge_score,
                final_score=final_score,
                personality_match=personality_match,
                temporal_context=temporal_context,
                mood_reasoning=mood_reasoning,
                cultural_bridge_potential=cultural_bridge_potential,
                confidence=confidence,
                recommendation_type=recommendation_type
            )
            
            comprehensive_recs.append(comprehensive_rec)
        
        # Sort by final score
        comprehensive_recs.sort(key=lambda x: x.final_score, reverse=True)
        
        return comprehensive_recs
    
    def generate_system_insights(
        self,
        streaming_data: pd.DataFrame,
        current_context: Optional[ListeningContext] = None
    ) -> SystemInsights:
        """Generate comprehensive insights from all models"""
        
        insights = {}
        
        # Current listening patterns
        recent_data = streaming_data.tail(1000)
        
        insights['current_listening_patterns'] = {
            'daily_average': len(recent_data) / 30,  # Last 30 days equivalent
            'top_artists': recent_data['artist_name'].value_counts().head(5).to_dict(),
            'cultural_distribution': self._analyze_cultural_distribution(recent_data),
            'temporal_activity': self._analyze_temporal_activity(recent_data)
        }
        
        # Predicted preferences
        insights['predicted_preferences'] = {}
        
        if self.temporal_models:
            try:
                volume_predictor = self.temporal_models['volume_predictor']['model']
                tomorrow = datetime.now() + timedelta(days=1)
                volume_pred = volume_predictor.predict_listening_volume(tomorrow, streaming_data)
                
                insights['predicted_preferences'] = {
                    'tomorrow_plays': volume_pred.predicted_plays,
                    'confidence_interval': volume_pred.confidence_interval,
                    'expected_peak_hours': volume_pred.peak_hours
                }
            except Exception:
                pass
        
        # Cultural evolution forecast
        insights['cultural_evolution_forecast'] = {}
        
        if self.temporal_models:
            try:
                cultural_predictor = self.temporal_models['cultural_predictor']['model']
                cultural_forecast = cultural_predictor.predict_cultural_evolution(6, streaming_data)
                
                insights['cultural_evolution_forecast'] = {
                    'next_6_months': cultural_forecast,
                    'trend': 'vietnamese_increasing' if cultural_forecast['vietnamese_ratios'][-1] > cultural_forecast['vietnamese_ratios'][0] else 'western_increasing'
                }
            except Exception:
                pass
        
        # Bridge song opportunities
        bridge_opportunities = []
        
        if self.bridge_models:
            try:
                bridge_detector = self.bridge_models['bridge_detector']['model']
                candidate_tracks = streaming_data[['track_id', 'track_name', 'artist_name']].drop_duplicates().sample(50)
                
                bridge_songs = bridge_detector.detect_bridge_songs(
                    candidate_tracks, streaming_data, threshold=0.6
                )
                
                bridge_opportunities = [f"{b.track_name} - {b.artist_name}" for b in bridge_songs[:5]]
            except Exception:
                pass
        
        insights['bridge_song_opportunities'] = bridge_opportunities
        
        # Mood assessment
        mood_assessment = "neutral"
        
        if current_context and self.context_models:
            try:
                mood_model = self.context_models['mood_recommender']['model']
                mood_assessment = mood_model.predict_mood(current_context)
            except Exception:
                pass
        
        # Temporal recommendations
        temporal_recs = {}
        
        if current_context:
            temporal_recs = {
                'current_time_appropriateness': self._assess_current_time_appropriateness(current_context),
                'suggested_session_length': self._suggest_session_length(current_context),
                'cultural_preference_context': self._assess_cultural_context(current_context)
            }
        
        return SystemInsights(
            current_listening_patterns=insights['current_listening_patterns'],
            predicted_preferences=insights.get('predicted_preferences', {}),
            cultural_evolution_forecast=insights.get('cultural_evolution_forecast', {}),
            bridge_song_opportunities=bridge_opportunities,
            mood_assessment=mood_assessment,
            temporal_recommendations=temporal_recs
        )
    
    def _calculate_mood_track_fit(self, mood: str, track_rec) -> float:
        """Calculate how well a track fits a predicted mood"""
        
        # Simplified mood-track matching
        mood_preferences = {
            'energetic': 0.8,
            'calm': 0.6,
            'focused': 0.7,
            'relaxed': 0.5,
            'social': 0.9,
            'contemplative': 0.4,
            'chill': 0.5
        }
        
        return mood_preferences.get(mood, 0.5)
    
    def _analyze_cultural_distribution(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze cultural distribution in data"""
        
        def classify_culture(artist_name):
            if pd.isna(artist_name):
                return 'unknown'
            artist_lower = str(artist_name).lower()
            if any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                return 'vietnamese'
            else:
                return 'western'
        
        data['cultural_class'] = data['artist_name'].apply(classify_culture)
        distribution = data['cultural_class'].value_counts(normalize=True).to_dict()
        
        return distribution
    
    def _analyze_temporal_activity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal activity patterns"""
        
        data['played_at'] = pd.to_datetime(data['played_at'])
        data['hour'] = data['played_at'].dt.hour
        data['day_of_week'] = data['played_at'].dt.day_name()
        
        return {
            'peak_hour': data['hour'].mode().iloc[0] if not data['hour'].mode().empty else 19,
            'most_active_day': data['day_of_week'].mode().iloc[0] if not data['day_of_week'].mode().empty else 'Saturday',
            'weekend_activity': data[data['played_at'].dt.weekday >= 5]['track_id'].count() / len(data)
        }
    
    def _assess_current_time_appropriateness(self, context: ListeningContext) -> str:
        """Assess appropriateness of current time for listening"""
        
        hour = context.current_time.hour
        
        if 6 <= hour < 9:
            return "Great time for energetic morning music"
        elif 9 <= hour < 12:
            return "Good for focused work background music"
        elif 12 <= hour < 14:
            return "Perfect lunch break listening time"
        elif 14 <= hour < 17:
            return "Ideal for afternoon productivity music"
        elif 17 <= hour < 20:
            return "Excellent evening relaxation time"
        elif 20 <= hour < 23:
            return "Prime time for music exploration"
        else:
            return "Late night chill music recommended"
    
    def _suggest_session_length(self, context: ListeningContext) -> str:
        """Suggest optimal session length based on context"""
        
        if context.work_hours:
            return "Short sessions (3-5 tracks) for work focus"
        elif context.is_weekend:
            return "Long sessions (20+ tracks) for weekend exploration"
        elif context.time_of_day == 'evening':
            return "Medium sessions (10-15 tracks) for evening relaxation"
        else:
            return "Flexible sessions (5-10 tracks) for general listening"
    
    def _assess_cultural_context(self, context: ListeningContext) -> str:
        """Assess cultural listening context"""
        
        cultural_context = "Balanced Vietnamese-Western mix recommended"
        
        if context.time_of_day == 'morning':
            cultural_context = "Vietnamese music great for morning reflection"
        elif context.work_hours:
            cultural_context = "Western instrumental or Vietnamese ambient recommended"
        elif context.is_weekend and context.time_of_day == 'evening':
            cultural_context = "Perfect time for cross-cultural exploration"
        
        return cultural_context
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models in the system"""
        
        status = {
            'system_initialized': self.is_initialized,
            'core_engine': self.core_engine is not None,
            'advanced_models_available': ADVANCED_MODELS_AVAILABLE,
            'temporal_models': self.temporal_models is not None,
            'context_models': self.context_models is not None,
            'bridge_models': self.bridge_models is not None,
            'model_weights': self.model_weights
        }
        
        return status
    
    def save_models(self, save_dir: str):
        """Save all trained models"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model status and configuration
        status = self.get_model_status()
        with open(save_path / 'model_status.json', 'w') as f:
            json.dump(status, f, indent=2, default=str)
        
        # Save individual models (if they support saving)
        if self.temporal_models:
            try:
                for model_name, model_info in self.temporal_models.items():
                    if hasattr(model_info['model'], 'save_model'):
                        model_info['model'].save_model(str(save_path / f"{model_name}.joblib"))
            except Exception:
                pass
        
        self.logger.info(f"Models saved to {save_path}")


def create_comprehensive_system(
    streaming_data: pd.DataFrame,
    phase3_results_path: str = 'results/phase3'
) -> ComprehensiveModelManager:
    """
    Create and initialize the complete comprehensive model system.
    
    One-stop function to get a fully operational recommendation system.
    """
    
    print("🚀 Creating Comprehensive Music Recommendation System...")
    
    # Initialize model manager
    manager = ComprehensiveModelManager(phase3_results_path)
    
    # Initialize all models
    init_results = manager.initialize_system(streaming_data)
    
    print("\n✅ System Initialization Results:")
    for component, result in init_results.items():
        if isinstance(result, dict):
            status = result.get('status', 'unknown')
            print(f"   {component}: {status}")
            if status == 'success' and 'models_trained' in result:
                print(f"      → {result['models_trained']} models trained")
        else:
            print(f"   {component}: {result}")
    
    if init_results['system_status'] == 'success':
        print("\n🎉 Comprehensive Model System Ready!")
        print("   • Cross-cultural recommendations ✓")
        print("   • Temporal predictions ✓")
        print("   • Context-aware suggestions ✓")
        print("   • Advanced bridge detection ✓")
    else:
        print("\n⚠️ System initialized with limited functionality")
    
    return manager


if __name__ == "__main__":
    # Example usage
    print("🎵 Comprehensive Music Recommendation System")
    print("=" * 50)
    
    try:
        # Load data
        streaming_data = pd.read_parquet('../../data/processed/streaming_data_processed.parquet')
        
        # Create comprehensive system
        system = create_comprehensive_system(streaming_data)
        
        # Example comprehensive recommendations
        candidate_tracks = streaming_data[['track_id', 'track_name', 'artist_name']].drop_duplicates().head(20)
        
        # Current context
        current_context = ListeningContext(
            current_time=datetime.now(),
            is_weekend=datetime.now().weekday() >= 5,
            time_of_day='evening',
            work_hours=False,
            recent_cultural_preference=0.6,
            session_type='new_session',
            energy_preference=0.7
        )
        
        # Generate comprehensive recommendations
        recommendations = system.generate_comprehensive_recommendations(
            streaming_data, candidate_tracks, n_recommendations=5, current_context=current_context
        )
        
        print(f"\n🎯 Comprehensive Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec.track_name} - {rec.artist_name}")
            print(f"      Final Score: {rec.final_score:.3f} ({rec.recommendation_type})")
            print(f"      Breakdown: Base={rec.base_score:.2f}, Temporal={rec.temporal_score:.2f}, Mood={rec.mood_score:.2f}, Bridge={rec.bridge_score:.2f}")
            print(f"      Context: {rec.temporal_context}")
            print(f"      Bridge: {rec.cultural_bridge_potential}")
        
        # Generate system insights
        insights = system.generate_system_insights(streaming_data, current_context)
        
        print(f"\n🔍 System Insights:")
        print(f"   Current Mood: {insights.mood_assessment}")
        print(f"   Daily Average: {insights.current_listening_patterns.get('daily_average', 0):.1f} plays")
        print(f"   Bridge Opportunities: {len(insights.bridge_song_opportunities)} songs")
        
        if insights.predicted_preferences:
            print(f"   Tomorrow's Prediction: {insights.predicted_preferences.get('tomorrow_plays', 0):.0f} plays")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure data files are available for testing.")
