"""
Phase 4: Recommendation System Evaluation Framework

Temporal train/test splits respecting chronological order.
Evaluates recommendation quality using your 4+ years of listening history.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from sklearn.metrics import ndcg_score, precision_recall_curve
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for recommendation system"""
    ndcg_at_10: float
    precision_at_10: float
    recall_at_10: float
    coverage: float
    diversity: float
    novelty: float
    cultural_diversity: float
    serendipity: float
    temporal_consistency: float


@dataclass
class TemporalSplit:
    """Temporal train/test split maintaining chronological order"""
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    split_date: datetime
    train_period: str
    test_period: str


class TemporalSplitter:
    """
    Creates temporal train/test splits that respect chronological order.
    
    Prevents temporal data leakage by ensuring test data comes after train data.
    """
    
    def __init__(self, streaming_data: pd.DataFrame):
        self.streaming_data = streaming_data.copy()
        self.streaming_data['played_at'] = pd.to_datetime(self.streaming_data['played_at'])
        self.streaming_data = self.streaming_data.sort_values('played_at')
        
    def create_temporal_splits(
        self, 
        test_months: int = 6,
        min_train_months: int = 12
    ) -> List[TemporalSplit]:
        """Create multiple temporal splits for robust evaluation"""
        
        splits = []
        
        # Get date range
        min_date = self.streaming_data['played_at'].min()
        max_date = self.streaming_data['played_at'].max()
        
        logger.info(f"Creating temporal splits from {min_date.date()} to {max_date.date()}")
        
        # Create splits moving forward in time
        current_date = min_date + timedelta(days=min_train_months * 30)
        
        while current_date + timedelta(days=test_months * 30) < max_date:
            # Define train period (all data before current_date)
            train_data = self.streaming_data[
                self.streaming_data['played_at'] < current_date
            ]
            
            # Define test period (next test_months of data)
            test_start = current_date
            test_end = current_date + timedelta(days=test_months * 30)
            
            test_data = self.streaming_data[
                (self.streaming_data['played_at'] >= test_start) &
                (self.streaming_data['played_at'] < test_end)
            ]
            
            # Only create split if we have sufficient data
            if len(train_data) >= 1000 and len(test_data) >= 100:
                split = TemporalSplit(
                    train_data=train_data,
                    test_data=test_data,
                    split_date=current_date,
                    train_period=f"{min_date.strftime('%Y-%m')} to {current_date.strftime('%Y-%m')}",
                    test_period=f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}"
                )
                splits.append(split)
                logger.info(f"Created split: Train={len(train_data)}, Test={len(test_data)}")
            
            # Move forward by 3 months for next split
            current_date += timedelta(days=90)
            
        return splits
    
    def create_final_evaluation_split(self, train_ratio: float = 0.8) -> TemporalSplit:
        """Create final train/test split for evaluation"""
        
        # Sort by time
        sorted_data = self.streaming_data.sort_values('played_at')
        
        # Split chronologically
        split_idx = int(len(sorted_data) * train_ratio)
        train_data = sorted_data.iloc[:split_idx]
        test_data = sorted_data.iloc[split_idx:]
        
        split_date = test_data['played_at'].min()
        
        return TemporalSplit(
            train_data=train_data,
            test_data=test_data,
            split_date=split_date,
            train_period=f"{train_data['played_at'].min().strftime('%Y-%m')} to {train_data['played_at'].max().strftime('%Y-%m')}",
            test_period=f"{test_data['played_at'].min().strftime('%Y-%m')} to {test_data['played_at'].max().strftime('%Y-%m')}"
        )


class RecommendationEvaluator:
    """
    Comprehensive evaluation of recommendation system performance.
    
    Evaluates accuracy, diversity, novelty, and cultural aspects.
    """
    
    def __init__(self, streaming_data: pd.DataFrame):
        self.streaming_data = streaming_data
        self.splitter = TemporalSplitter(streaming_data)
        
    def evaluate_recommendations(
        self,
        recommendations: List,
        test_data: pd.DataFrame,
        k: int = 10
    ) -> EvaluationMetrics:
        """Evaluate recommendation quality using multiple metrics"""
        
        # Get top-k recommendations
        top_k_recs = recommendations[:k]
        rec_track_ids = [rec.track_id for rec in top_k_recs]
        
        # Get ground truth (actually played tracks in test period)
        actual_tracks = set(test_data['track_id'].tolist())
        
        # Precision@K and Recall@K
        hits = len(set(rec_track_ids) & actual_tracks)
        precision_at_k = hits / k if k > 0 else 0
        recall_at_k = hits / len(actual_tracks) if len(actual_tracks) > 0 else 0
        
        # NDCG@K calculation
        ndcg_at_k = self._calculate_ndcg(rec_track_ids, actual_tracks, k)
        
        # Coverage (fraction of all possible items recommended)
        all_possible_tracks = set(self.streaming_data['track_id'].tolist())
        coverage = len(set(rec_track_ids)) / len(all_possible_tracks) if len(all_possible_tracks) > 0 else 0
        
        # Diversity (intra-list diversity)
        diversity = self._calculate_diversity(top_k_recs)
        
        # Novelty (how rare are the recommended items)
        novelty = self._calculate_novelty(top_k_recs, self.streaming_data)
        
        # Cultural diversity
        cultural_diversity = self._calculate_cultural_diversity(top_k_recs)
        
        # Serendipity (unexpected relevant items)
        serendipity = self._calculate_serendipity(top_k_recs, test_data)
        
        # Temporal consistency (recommendations align with recent preferences)
        temporal_consistency = self._calculate_temporal_consistency(top_k_recs, test_data)
        
        return EvaluationMetrics(
            ndcg_at_10=ndcg_at_k,
            precision_at_10=precision_at_k,
            recall_at_10=recall_at_k,
            coverage=coverage,
            diversity=diversity,
            novelty=novelty,
            cultural_diversity=cultural_diversity,
            serendipity=serendipity,
            temporal_consistency=temporal_consistency
        )
    
    def _calculate_ndcg(self, recommendations: List[str], actual: set, k: int) -> float:
        """Calculate NDCG@K"""
        
        # Create relevance scores (1 for actual, 0 for not)
        relevance_scores = [1 if track_id in actual else 0 for track_id in recommendations[:k]]
        
        if sum(relevance_scores) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = ideal_relevance[0]
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_diversity(self, recommendations: List) -> float:
        """Calculate intra-list diversity based on audio features"""
        
        if len(recommendations) < 2:
            return 0.0
        
        # Extract audio features
        features = []
        for rec in recommendations:
            feature_vector = [
                getattr(rec, 'energy', 0.5),
                getattr(rec, 'valence', 0.5), 
                getattr(rec, 'danceability', 0.5)
            ]
            features.append(feature_vector)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                dist = np.linalg.norm(np.array(features[i]) - np.array(features[j]))
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_novelty(self, recommendations: List, historical_data: pd.DataFrame) -> float:
        """Calculate novelty (popularity-based)"""
        
        # Calculate track popularity from historical data
        track_popularity = historical_data['track_id'].value_counts()
        total_plays = len(historical_data)
        
        novelty_scores = []
        for rec in recommendations:
            track_plays = track_popularity.get(rec.track_id, 0)
            popularity = track_plays / total_plays if total_plays > 0 else 0
            novelty = 1 - popularity  # More novel = less popular
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def _calculate_cultural_diversity(self, recommendations: List) -> float:
        """Calculate cultural diversity in recommendations"""
        
        cultures = [rec.cultural_classification for rec in recommendations]
        culture_counts = pd.Series(cultures).value_counts()
        
        # Shannon entropy for cultural diversity
        if len(culture_counts) <= 1:
            return 0.0
        
        proportions = culture_counts / len(cultures)
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(culture_counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_serendipity(self, recommendations: List, test_data: pd.DataFrame) -> float:
        """Calculate serendipity (unexpected but relevant recommendations)"""
        
        # Get user's historical preferences (genres, artists)
        historical_artists = set(self.streaming_data['artist_name'].tolist())
        
        serendipity_scores = []
        for rec in recommendations:
            # Check if artist is new to user
            is_new_artist = rec.artist_name not in historical_artists
            
            # Check if it was actually played in test period (relevant)
            was_played = rec.track_id in test_data['track_id'].tolist()
            
            # Serendipitous = new artist + actually played
            if is_new_artist and was_played:
                serendipity_scores.append(1.0)
            elif is_new_artist:
                serendipity_scores.append(0.5)  # New but not validated
            else:
                serendipity_scores.append(0.0)
        
        return np.mean(serendipity_scores) if serendipity_scores else 0.0
    
    def _calculate_temporal_consistency(self, recommendations: List, test_data: pd.DataFrame) -> float:
        """Calculate how well recommendations align with actual test period preferences"""
        
        if len(test_data) == 0:
            return 0.0
        
        # Get actual preferences from test period
        test_features = {
            'avg_energy': test_data.get('audio_energy', pd.Series([0.5])).mean(),
            'avg_valence': test_data.get('audio_valence', pd.Series([0.5])).mean(),
            'avg_danceability': test_data.get('audio_danceability', pd.Series([0.6])).mean()
        }
        
        # Calculate recommendation features
        rec_features = {
            'avg_energy': np.mean([getattr(rec, 'energy', 0.5) for rec in recommendations]),
            'avg_valence': np.mean([getattr(rec, 'valence', 0.5) for rec in recommendations]),
            'avg_danceability': np.mean([getattr(rec, 'danceability', 0.6) for rec in recommendations])
        }
        
        # Calculate similarity
        similarities = []
        for feature in ['avg_energy', 'avg_valence', 'avg_danceability']:
            test_val = test_features.get(feature, 0.5)
            rec_val = rec_features.get(feature, 0.5)
            similarity = 1 - abs(test_val - rec_val)
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def run_comprehensive_evaluation(
        self,
        recommendation_engine,
        n_splits: int = 3
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across multiple temporal splits"""
        
        logger.info("Starting comprehensive evaluation...")
        
        # Create temporal splits
        splits = self.splitter.create_temporal_splits()
        if len(splits) > n_splits:
            splits = splits[-n_splits:]  # Use most recent splits
            
        evaluation_results = []
        
        for i, split in enumerate(splits):
            logger.info(f"Evaluating split {i+1}/{len(splits)}: {split.test_period}")
            
            try:
                # Create user profile from training data
                user_profile = recommendation_engine.create_user_profile(split.train_data)
                
                # Get candidate tracks (tracks not in training set but available)
                all_tracks = self.streaming_data[
                    ['track_id', 'track_name', 'artist_name', 'audio_energy', 
                     'audio_valence', 'audio_danceability', 'audio_acousticness', 'dominant_culture']
                ].drop_duplicates('track_id')
                
                train_track_ids = set(split.train_data['track_id'].tolist())
                candidate_tracks = all_tracks[~all_tracks['track_id'].isin(train_track_ids)]
                
                # Generate recommendations
                recommendations = recommendation_engine.generate_recommendations(
                    user_profile=user_profile,
                    candidate_tracks=candidate_tracks,
                    n_recommendations=50,  # Generate more for evaluation
                    include_bridges=True
                )
                
                # Evaluate recommendations
                metrics = self.evaluate_recommendations(recommendations, split.test_data)
                
                evaluation_results.append({
                    'split_id': i,
                    'train_period': split.train_period,
                    'test_period': split.test_period,
                    'train_size': len(split.train_data),
                    'test_size': len(split.test_data),
                    'metrics': metrics
                })
                
                logger.info(f"Split {i+1} results: NDCG@10={metrics.ndcg_at_10:.3f}, "
                          f"Precision@10={metrics.precision_at_10:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating split {i+1}: {str(e)}")
                continue
        
        # Aggregate results
        if not evaluation_results:
            raise ValueError("No successful evaluations completed")
            
        aggregated_metrics = self._aggregate_evaluation_results(evaluation_results)
        
        # Create comprehensive report
        comprehensive_report = {
            'evaluation_date': datetime.now().isoformat(),
            'n_splits': len(evaluation_results),
            'individual_results': evaluation_results,
            'aggregated_metrics': aggregated_metrics,
            'evaluation_summary': self._generate_evaluation_summary(aggregated_metrics)
        }
        
        # Save evaluation report
        output_path = Path('results/phase4_evaluation_report.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
            
        logger.info(f"Comprehensive evaluation saved to {output_path}")
        
        return comprehensive_report
    
    def _aggregate_evaluation_results(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate evaluation metrics across splits"""
        
        metrics_to_aggregate = [
            'ndcg_at_10', 'precision_at_10', 'recall_at_10', 'coverage',
            'diversity', 'novelty', 'cultural_diversity', 'serendipity', 'temporal_consistency'
        ]
        
        aggregated = {}
        for metric in metrics_to_aggregate:
            values = [result['metrics'].__dict__[metric] for result in results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        
        return aggregated
    
    def _generate_evaluation_summary(self, aggregated_metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate human-readable evaluation summary"""
        
        summary = {}
        
        # Overall performance
        ndcg_mean = aggregated_metrics['ndcg_at_10_mean']
        precision_mean = aggregated_metrics['precision_at_10_mean']
        
        if ndcg_mean > 0.3 and precision_mean > 0.1:
            summary['overall_performance'] = 'Good'
        elif ndcg_mean > 0.2 and precision_mean > 0.05:
            summary['overall_performance'] = 'Moderate'
        else:
            summary['overall_performance'] = 'Needs Improvement'
        
        # Diversity assessment
        diversity_mean = aggregated_metrics['diversity_mean']
        cultural_diversity_mean = aggregated_metrics['cultural_diversity_mean']
        
        if diversity_mean > 0.7 and cultural_diversity_mean > 0.6:
            summary['diversity'] = 'High diversity across audio and cultural dimensions'
        elif diversity_mean > 0.5 and cultural_diversity_mean > 0.4:
            summary['diversity'] = 'Moderate diversity'
        else:
            summary['diversity'] = 'Low diversity - may be over-personalizing'
        
        # Novelty and serendipity
        novelty_mean = aggregated_metrics['novelty_mean']
        serendipity_mean = aggregated_metrics['serendipity_mean']
        
        if novelty_mean > 0.6 and serendipity_mean > 0.2:
            summary['discovery'] = 'Good balance of novelty and serendipity'
        elif novelty_mean > 0.4:
            summary['discovery'] = 'Moderate discovery potential'
        else:
            summary['discovery'] = 'Low discovery - recommendations may be too safe'
        
        # Temporal consistency
        temporal_mean = aggregated_metrics['temporal_consistency_mean']
        if temporal_mean > 0.7:
            summary['temporal_alignment'] = 'Strong alignment with temporal preferences'
        elif temporal_mean > 0.5:
            summary['temporal_alignment'] = 'Moderate temporal alignment'
        else:
            summary['temporal_alignment'] = 'Poor temporal alignment - may not adapt to preference changes'
        
        return summary