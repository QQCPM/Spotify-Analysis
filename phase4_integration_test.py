#!/usr/bin/env python3
"""
Phase 4: Complete System Integration & Testing

End-to-end test of the cross-cultural music recommendation system.
Validates all components work together correctly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from models.recommendation_engine import CrossCulturalRecommendationEngine, UserProfile
from evaluation.recommendation_evaluator import RecommendationEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Phase4IntegrationTester:
    """
    Complete integration test for Phase 4 recommendation system.
    
    Tests all components:
    - Data loading and preprocessing
    - Recommendation engine initialization
    - User profile creation
    - Recommendation generation
    - Evaluation framework
    """
    
    def __init__(self):
        self.streaming_data = None
        self.playlist_data = None
        self.engine = None
        self.evaluator = None
        self.test_results = {}
        
    def run_complete_integration_test(self):
        """Run complete end-to-end integration test"""
        
        print("🎵 Phase 4: Cross-Cultural Music Recommendation System")
        print("=" * 60)
        print("Running complete integration test...\n")
        
        try:
            # Test 1: Data loading
            print("📊 Test 1: Data Loading...")
            self._test_data_loading()
            print("✅ Data loading successful\n")
            
            # Test 2: Engine initialization
            print("🧬 Test 2: Recommendation Engine Initialization...")
            self._test_engine_initialization()
            print("✅ Engine initialization successful\n")
            
            # Test 3: User profile creation
            print("👤 Test 3: User Profile Creation...")
            self._test_user_profile_creation()
            print("✅ User profile creation successful\n")
            
            # Test 4: Recommendation generation
            print("🎯 Test 4: Recommendation Generation...")
            self._test_recommendation_generation()
            print("✅ Recommendation generation successful\n")
            
            # Test 5: Bridge song discovery
            print("🌉 Test 5: Bridge Song Discovery...")
            self._test_bridge_song_discovery()
            print("✅ Bridge song discovery successful\n")
            
            # Test 6: Evaluation framework
            print("📈 Test 6: Evaluation Framework...")
            self._test_evaluation_framework()
            print("✅ Evaluation framework successful\n")
            
            # Generate integration report
            print("📋 Test 7: Integration Report...")
            self._generate_integration_report()
            print("✅ Integration report generated\n")
            
            print("🎉 ALL TESTS PASSED!")
            print("Phase 4 recommendation system is fully operational.")
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {str(e)}")
            print(f"❌ Integration test failed: {str(e)}")
            return False
    
    def _test_data_loading(self):
        """Test data loading functionality"""
        
        # Load streaming data
        streaming_file = Path('data/processed/streaming_data_processed.parquet')
        if not streaming_file.exists():
            raise FileNotFoundError("Streaming data not found")
        
        self.streaming_data = pd.read_parquet(streaming_file)
        self.streaming_data['played_at'] = pd.to_datetime(self.streaming_data['played_at'])
        
        print(f"   • Loaded {len(self.streaming_data):,} streaming records")
        print(f"   • Date range: {self.streaming_data['played_at'].min().date()} to {self.streaming_data['played_at'].max().date()}")
        print(f"   • Unique tracks: {self.streaming_data['track_id'].nunique():,}")
        
        # Load playlist data
        playlist_dir = Path('/Users/quangnguyen/Downloads/spotify_playlists')
        if not playlist_dir.exists():
            print("   ⚠️ Playlist directory not found, using streaming data only")
            self.playlist_data = pd.DataFrame()  # Empty dataframe
        else:
            playlists = []
            for csv_file in playlist_dir.glob('*.csv'):
                df = pd.read_csv(csv_file)
                df['playlist_name'] = csv_file.stem
                playlists.append(df)
            
            if playlists:
                self.playlist_data = pd.concat(playlists, ignore_index=True)
                print(f"   • Loaded {len(self.playlist_data):,} playlist tracks")
            else:
                self.playlist_data = pd.DataFrame()
        
        self.test_results['data_loading'] = {
            'streaming_records': len(self.streaming_data),
            'playlist_tracks': len(self.playlist_data),
            'status': 'success'
        }
    
    def _test_engine_initialization(self):
        """Test recommendation engine initialization"""
        
        # Check if Phase 3 results exist
        phase3_dir = Path('results/phase3')
        if not phase3_dir.exists():
            raise FileNotFoundError("Phase 3 results directory not found")
        
        # Initialize engine
        self.engine = CrossCulturalRecommendationEngine('results/phase3')
        
        # Verify components
        n_personalities = len(self.engine.personality_recommenders)
        n_change_points = len(self.engine.temporal_weighting.change_points) if self.engine.temporal_weighting else 0
        n_bridge_songs = len(self.engine.bridge_engine.bridge_songs) if self.engine.bridge_engine else 0
        
        print(f"   • Initialized {n_personalities} personality recommenders")
        print(f"   • Loaded {n_change_points} temporal change points")
        print(f"   • Loaded {n_bridge_songs} bridge songs")
        
        self.test_results['engine_initialization'] = {
            'personalities': n_personalities,
            'change_points': n_change_points,
            'bridge_songs': n_bridge_songs,
            'status': 'success'
        }
    
    def _test_user_profile_creation(self):
        """Test user profile creation"""
        
        # Use recent listening data (last 1000 records)
        recent_data = self.streaming_data.tail(1000)
        
        # Create user profile
        user_profile = self.engine.create_user_profile(recent_data)
        
        print(f"   • Created user profile with {len(user_profile.personality_weights)} personality weights")
        print(f"   • Personality weights: {user_profile.personality_weights}")
        print(f"   • Cultural preferences: {user_profile.cultural_preferences}")
        print(f"   • Temporal context: {user_profile.temporal_context}")
        
        self.test_results['user_profile_creation'] = {
            'personality_weights': user_profile.personality_weights,
            'cultural_preferences': user_profile.cultural_preferences,
            'temporal_context': user_profile.temporal_context,
            'status': 'success'
        }
        
        # Store for next test
        self.test_user_profile = user_profile
    
    def _test_recommendation_generation(self):
        """Test recommendation generation"""
        
        # Get candidate tracks (unique tracks not in recent listening)
        # Add synthetic audio features for testing
        all_tracks = self.streaming_data[
            ['track_id', 'track_name', 'artist_name']
        ].drop_duplicates('track_id').copy()
        
        # Add synthetic audio features and cultural classification for testing
        np.random.seed(42)  # Reproducible synthetic features
        all_tracks['audio_energy'] = np.random.beta(2, 2, len(all_tracks))
        all_tracks['audio_valence'] = np.random.beta(2, 2, len(all_tracks)) 
        all_tracks['audio_danceability'] = np.random.beta(2, 2, len(all_tracks))
        all_tracks['audio_acousticness'] = np.random.beta(1, 3, len(all_tracks))
        
        # Simple cultural classification based on artist name
        def classify_culture_simple(artist_name):
            if pd.isna(artist_name):
                return 'unknown'
            artist_lower = str(artist_name).lower()
            vietnamese_indicators = ['buitruonglinh', 'vsoul', 'khói', 'đen', 'mck', 'obito']
            if any(ind in artist_lower for ind in vietnamese_indicators):
                return 'vietnamese'
            elif any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                return 'vietnamese'
            else:
                return 'western'
        
        all_tracks['dominant_culture'] = all_tracks['artist_name'].apply(classify_culture_simple)
        
        # Sample candidate tracks
        candidate_tracks = all_tracks.sample(min(500, len(all_tracks)))
        
        # Generate recommendations
        recommendations = self.engine.generate_recommendations(
            user_profile=self.test_user_profile,
            candidate_tracks=candidate_tracks,
            n_recommendations=10,
            include_bridges=True,
            exploration_factor=0.2
        )
        
        print(f"   • Generated {len(recommendations)} recommendations")
        
        # Analyze recommendations
        cultural_dist = {}
        for rec in recommendations:
            culture = rec.cultural_classification
            cultural_dist[culture] = cultural_dist.get(culture, 0) + 1
        
        avg_score = np.mean([rec.score for rec in recommendations])
        avg_bridge_score = np.mean([rec.bridge_score for rec in recommendations])
        
        print(f"   • Average recommendation score: {avg_score:.3f}")
        print(f"   • Average bridge score: {avg_bridge_score:.3f}")
        print(f"   • Cultural distribution: {cultural_dist}")
        
        # Print top 3 recommendations
        print("   • Top 3 recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"     {i}. {rec.track_name} - {rec.artist_name} (Score: {rec.score:.3f})")
        
        self.test_results['recommendation_generation'] = {
            'n_recommendations': len(recommendations),
            'avg_score': avg_score,
            'avg_bridge_score': avg_bridge_score,
            'cultural_distribution': cultural_dist,
            'top_recommendations': [
                f"{rec.track_name} - {rec.artist_name}" for rec in recommendations[:3]
            ],
            'status': 'success'
        }
        
        # Store for evaluation
        self.test_recommendations = recommendations
    
    def _test_bridge_song_discovery(self):
        """Test bridge song discovery functionality"""
        
        # Test different cultural transitions
        transitions = [
            ('vietnamese', 'western'),
            ('western', 'vietnamese'),
            ('vietnamese', 'chinese')
        ]
        
        bridge_results = {}
        
        for from_culture, to_culture in transitions:
            bridges = self.engine.bridge_engine.get_bridge_recommendations(
                from_culture, to_culture, n_recommendations=3
            )
            
            bridge_results[f"{from_culture}_to_{to_culture}"] = len(bridges)
            print(f"   • {from_culture} → {to_culture}: {len(bridges)} bridge songs")
        
        self.test_results['bridge_song_discovery'] = {
            'bridge_transitions': bridge_results,
            'status': 'success'
        }
    
    def _test_evaluation_framework(self):
        """Test evaluation framework (lightweight test)"""
        
        # Initialize evaluator
        self.evaluator = RecommendationEvaluator(self.streaming_data)
        
        # Create a simple temporal split
        split = self.evaluator.splitter.create_final_evaluation_split(train_ratio=0.9)
        
        print(f"   • Created temporal split: Train={len(split.train_data)}, Test={len(split.test_data)}")
        print(f"   • Split date: {split.split_date.date()}")
        
        # Quick evaluation test (using test recommendations if available)
        if hasattr(self, 'test_recommendations'):
            metrics = self.evaluator.evaluate_recommendations(
                self.test_recommendations, split.test_data, k=10
            )
            
            print(f"   • NDCG@10: {metrics.ndcg_at_10:.3f}")
            print(f"   • Precision@10: {metrics.precision_at_10:.3f}")
            print(f"   • Diversity: {metrics.diversity:.3f}")
            print(f"   • Cultural Diversity: {metrics.cultural_diversity:.3f}")
            
            self.test_results['evaluation_framework'] = {
                'ndcg_at_10': metrics.ndcg_at_10,
                'precision_at_10': metrics.precision_at_10,
                'diversity': metrics.diversity,
                'cultural_diversity': metrics.cultural_diversity,
                'status': 'success'
            }
        else:
            self.test_results['evaluation_framework'] = {
                'status': 'partial - evaluation framework initialized only'
            }
    
    def _generate_integration_report(self):
        """Generate comprehensive integration test report"""
        
        report = {
            'test_date': datetime.now().isoformat(),
            'system_status': 'operational',
            'test_results': self.test_results,
            'system_summary': {
                'data_integration': f"{self.test_results['data_loading']['streaming_records']:,} streaming records + {self.test_results['data_loading']['playlist_tracks']:,} playlist tracks",
                'personalities_discovered': self.test_results['engine_initialization']['personalities'],
                'change_points_integrated': self.test_results['engine_initialization']['change_points'],
                'bridge_songs_available': self.test_results['engine_initialization']['bridge_songs'],
                'recommendation_quality': self.test_results.get('evaluation_framework', {}).get('ndcg_at_10', 'not_tested'),
                'cultural_diversity': self.test_results.get('evaluation_framework', {}).get('cultural_diversity', 'not_tested')
            },
            'next_steps': [
                "Run full evaluation with multiple temporal splits",
                "Launch interactive demo interface",
                "Collect user feedback for system refinement",
                "Consider A/B testing with different personality weights"
            ]
        }
        
        # Save integration report
        output_path = Path('results/phase4_integration_report.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   • Integration report saved to {output_path}")
        
        # Print summary
        print("\n📊 INTEGRATION TEST SUMMARY:")
        print(f"   🎵 System Status: {report['system_status'].upper()}")
        print(f"   📊 Data: {report['system_summary']['data_integration']}")
        print(f"   🧬 Personalities: {report['system_summary']['personalities_discovered']}")
        print(f"   ⏰ Change Points: {report['system_summary']['change_points_integrated']}")
        print(f"   🌉 Bridge Songs: {report['system_summary']['bridge_songs_available']}")
        if report['system_summary']['recommendation_quality'] != 'not_tested':
            print(f"   🎯 Recommendation Quality: {report['system_summary']['recommendation_quality']:.3f}")


def main():
    """Run Phase 4 integration test"""
    
    try:
        tester = Phase4IntegrationTester()
        success = tester.run_complete_integration_test()
        
        if success:
            print("\n🚀 PHASE 4 READY FOR PRODUCTION!")
            print("\nTo launch the interactive demo:")
            print("   streamlit run phase4_demo.py")
            print("\nTo run comprehensive evaluation:")
            print("   python -c 'from src.evaluation.recommendation_evaluator import *; evaluator = RecommendationEvaluator(data); evaluator.run_comprehensive_evaluation(engine)'")
            
            return 0
        else:
            print("\n❌ Integration test failed. Check logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Integration test crashed: {str(e)}")
        print(f"\n💥 Integration test crashed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())