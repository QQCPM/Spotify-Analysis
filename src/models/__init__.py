"""Machine Learning Models Module

Contains comprehensive recommendation system implementations:
- Cross-cultural recommendation engine with personality-based recommendations
- Temporal prediction models for listening volume and cultural evolution
- Context-aware recommenders for session-based and mood-based suggestions
- Advanced bridge detection models for cultural gateway songs
- Matrix factorization models (SVD, NMF, temporal)
- Neural temporal models (LSTM-based)
- Latent factor analysis and interpretability tools
"""

# Import main recommendation engine
from .recommendation_engine import (
    CrossCulturalRecommendationEngine,
    PersonalityRecommender,
    TemporalWeightingSystem,
    CulturalBridgeEngine,
    UserProfile,
    RecommendationResult
)

# Import temporal prediction models
try:
    from .temporal_prediction_model import (
        TemporalFeatureEngineer,
        ListeningVolumePredictor,
        CulturalEvolutionPredictor,
        TemporalPrediction,
        create_temporal_models_suite
    )
except ImportError:
    # Handle missing dependencies gracefully
    pass

# Import context-aware models
try:
    from .context_aware_recommender import (
        TemporalContextAnalyzer,
        SessionBasedRecommender,
        MoodBasedRecommender,
        ContextualRecommendation,
        ListeningContext,
        create_context_models_suite
    )
except ImportError:
    # Handle missing dependencies gracefully
    pass

# Import advanced bridge detection
try:
    from .advanced_bridge_detector import (
        AdvancedBridgeDetector,
        AudioFeatureAnalyzer,
        CulturalTransitionDetector,
        BridgeSongCandidate,
        CulturalTransition,
        create_bridge_detection_suite
    )
except ImportError:
    # Handle missing dependencies gracefully
    pass

__all__ = [
    # Core recommendation engine
    'CrossCulturalRecommendationEngine',
    'PersonalityRecommender',
    'TemporalWeightingSystem',
    'CulturalBridgeEngine',
    'UserProfile',
    'RecommendationResult',
    
    # Temporal prediction models
    'TemporalFeatureEngineer',
    'ListeningVolumePredictor',
    'CulturalEvolutionPredictor',
    'TemporalPrediction',
    'create_temporal_models_suite',
    
    # Context-aware models
    'TemporalContextAnalyzer',
    'SessionBasedRecommender',
    'MoodBasedRecommender',
    'ContextualRecommendation',
    'ListeningContext',
    'create_context_models_suite',
    
    # Advanced bridge detection
    'AdvancedBridgeDetector',
    'AudioFeatureAnalyzer',
    'CulturalTransitionDetector',
    'BridgeSongCandidate',
    'CulturalTransition',
    'create_bridge_detection_suite'
]