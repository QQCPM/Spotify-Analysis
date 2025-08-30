#!/usr/bin/env python3
"""
Spotify Extended Data Processing Script

Processes your complete Spotify Extended Streaming History for Phase 2 analysis.
This script serves as the entry point for comprehensive data extraction and validation.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from src.data_processing.extended_spotify_processor import process_spotify_extended_data
from src.data_processing.cultural_categorizer import categorize_cultural_data
from src.data_processing.feature_engineer import engineer_features
from src.evaluation.reproducibility import setup_reproducible_experiment


def main():
    """Main processing pipeline for Spotify Extended data"""
    
    # Set up paths
    data_path = Path("data/raw/spotify_extended")
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    
    print("🎵 Phase 2: Processing Spotify Extended Streaming History")
    print("=" * 60)
    
    # Step 1: Set up reproducible experiment
    print("\n📊 Step 1: Setting up reproducible experiment...")
    
    experiment_manager = setup_reproducible_experiment(
        name="Phase2_SpotifyExtendedAnalysis",
        description="Processing and analyzing Spotify Extended Streaming History for cross-cultural music preference research",
        hypothesis="Individual music preferences can be decomposed into 3-7 stable latent factors with identifiable temporal evolution patterns and cultural bridge behaviors",
        parameters={
            "min_listening_time_ms": 30000,
            "session_gap_minutes": 30,
            "exclude_podcasts": True,
            "cultural_classification": True,
            "temporal_analysis": True
        },
        researcher_info={
            "name": "Research Team",
            "institution": "Cross-Cultural Music Research",
            "contact": "research@example.com"
        }
    )
    
    print(f"✅ Experiment created: {experiment_manager.current_experiment.experiment_id}")
    
    # Step 2: Process raw Spotify data
    print("\n🔄 Step 2: Processing raw Spotify streaming data...")
    
    try:
        streaming_data, processing_stats = process_spotify_extended_data(
            data_path=data_path,
            output_path=processed_path / "streaming_data_processed.parquet"
        )
        
        print(f"✅ Processed {processing_stats.total_records:,} streaming records")
        print(f"   📅 Date range: {processing_stats.date_range[0].strftime('%Y-%m-%d')} to {processing_stats.date_range[1].strftime('%Y-%m-%d')}")
        print(f"   🎵 Unique tracks: {processing_stats.unique_tracks:,}")
        print(f"   👤 Unique artists: {processing_stats.unique_artists:,}")
        print(f"   ⏱️  Total listening time: {processing_stats.total_listening_time_hours:.1f} hours")
        print(f"   🌍 Countries: {', '.join(processing_stats.countries)}")
        print(f"   📱 Platforms: {', '.join(processing_stats.platforms)}")
        
    except Exception as e:
        print(f"❌ Error processing Spotify data: {str(e)}")
        return
    
    # Step 3: Cultural categorization
    print("\n🌏 Step 3: Analyzing cultural patterns...")
    
    try:
        # Create artist data for cultural analysis
        artist_data = streaming_data.groupby('artist_name').agg({
            'track_name': 'count',
            'ms_played': 'sum',
            'listening_location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()
        
        artist_data.columns = ['artist_name', 'play_count', 'total_ms_played', 'primary_location']
        artist_data['artist_id'] = artist_data['artist_name']  # Temporary ID
        artist_data['genres'] = ['unknown'] * len(artist_data)  # Placeholder
        
        # Apply cultural categorization
        categorized_data, cultural_stats = categorize_cultural_data(
            listening_data=streaming_data,
            artist_data=artist_data
        )
        
        print(f"✅ Cultural categorization complete:")
        print(f"   🇻🇳 Vietnamese tracks: {cultural_stats.get('vietnamese_tracks', 0):,} ({cultural_stats.get('vietnamese_percentage', 0):.1f}%)")
        print(f"   🌍 Western tracks: {cultural_stats.get('western_tracks', 0):,} ({cultural_stats.get('western_percentage', 0):.1f}%)")
        print(f"   🌉 Bridge tracks: {cultural_stats.get('bridge_tracks', 0):,} ({cultural_stats.get('bridge_percentage', 0):.1f}%)")
        print(f"   🎯 Average confidence: {cultural_stats.get('avg_confidence', 0):.3f}")
        
        # Save categorized data
        categorized_data.to_parquet(processed_path / "streaming_data_categorized.parquet", index=False)
        
        with open(processed_path / "cultural_stats.json", 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            cultural_stats_serializable = {}
            for key, value in cultural_stats.items():
                if hasattr(value, 'item'):  # numpy scalar
                    cultural_stats_serializable[key] = value.item()
                else:
                    cultural_stats_serializable[key] = value
            json.dump(cultural_stats_serializable, f, indent=2)
        
    except Exception as e:
        print(f"❌ Error in cultural categorization: {str(e)}")
        categorized_data = streaming_data
        cultural_stats = {}
    
    # Step 4: Feature engineering
    print("\n⚙️  Step 4: Engineering comprehensive features...")
    
    try:
        # Create dummy audio features (would normally be fetched from Spotify API)
        unique_tracks = categorized_data['track_id'].dropna().unique()
        audio_features = create_dummy_audio_features(unique_tracks)
        
        # Engineer features
        engineered_data, feature_groups = engineer_features(
            listening_data=categorized_data,
            audio_features=audio_features,
            cultural_data=categorized_data[['track_id'] + [col for col in categorized_data.columns if 'cultural' in col or 'vietnamese' in col or 'western' in col or 'bridge' in col]].drop_duplicates()
        )
        
        print(f"✅ Feature engineering complete:")
        print(f"   📊 Total features: {len(engineered_data.columns)}")
        print(f"   🕒 Temporal features: {len(feature_groups.get('temporal', []))}")
        print(f"   🎵 Audio features: {len(feature_groups.get('audio', []))}")
        print(f"   🌍 Cultural features: {len(feature_groups.get('cultural', []))}")
        print(f"   👤 Behavioral features: {len(feature_groups.get('behavioral', []))}")
        
        # Save engineered data
        engineered_data.to_parquet(processed_path / "streaming_data_engineered.parquet", index=False)
        
        with open(processed_path / "feature_groups.json", 'w') as f:
            json.dump(feature_groups, f, indent=2)
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {str(e)}")
        engineered_data = categorized_data
        feature_groups = {}
    
    # Step 5: Generate initial analysis report
    print("\n📈 Step 5: Generating initial analysis report...")
    
    try:
        report = generate_initial_analysis_report(
            streaming_data=streaming_data,
            processing_stats=processing_stats,
            cultural_stats=cultural_stats,
            feature_groups=feature_groups
        )
        
        with open(processed_path / "initial_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("✅ Initial analysis report generated")
        
        # Print key insights
        print("\n🔍 Key Insights from Your Data:")
        print("-" * 40)
        
        if 'listening_patterns' in report:
            patterns = report['listening_patterns']
            print(f"📅 Most active listening day: {patterns.get('most_active_day', 'Unknown')}")
            print(f"🕐 Peak listening hour: {patterns.get('peak_hour', 'Unknown')}:00")
            print(f"📱 Primary platform: {patterns.get('primary_platform', 'Unknown')}")
            print(f"🌍 Primary listening location: {patterns.get('primary_location', 'Unknown')}")
        
        if 'cultural_insights' in report:
            cultural = report['cultural_insights']
            print(f"🎵 Cultural balance: {cultural.get('cultural_balance', 'Unknown')}")
            print(f"🔄 Cultural exploration rate: {cultural.get('exploration_rate', 0):.1%}")
        
        if 'temporal_insights' in report:
            temporal = report['temporal_insights']
            print(f"📊 Listening sessions: {temporal.get('total_sessions', 0):,}")
            print(f"⏱️  Average session length: {temporal.get('avg_session_minutes', 0):.1f} minutes")
        
    except Exception as e:
        print(f"❌ Error generating analysis report: {str(e)}")
        report = {}
    
    # Step 6: Track experiment results
    print("\n💾 Step 6: Recording experiment results...")
    
    try:
        # Prepare results for tracking
        results = {
            'processing_completed': True,
            'data_quality': {
                'total_records': processing_stats.total_records,
                'date_range_days': (processing_stats.date_range[1] - processing_stats.date_range[0]).days,
                'unique_tracks': processing_stats.unique_tracks,
                'unique_artists': processing_stats.unique_artists
            },
            'cultural_analysis': cultural_stats,
            'feature_engineering': {
                'total_features': len(engineered_data.columns),
                'feature_groups': {k: len(v) for k, v in feature_groups.items()}
            },
            'analysis_report': report
        }
        
        metrics = {
            'listening_hours': processing_stats.total_listening_time_hours,
            'track_completion_rate': processing_stats.avg_track_completion_rate,
            'cultural_confidence': cultural_stats.get('avg_confidence', 0),
            'data_completeness': len(engineered_data) / processing_stats.total_records if processing_stats.total_records > 0 else 0
        }
        
        run_id = experiment_manager.track_experiment_results(
            results=results,
            metrics=metrics,
            execution_time=0.0  # Would be calculated in real implementation
        )
        
        print(f"✅ Experiment results tracked: {run_id}")
        
    except Exception as e:
        print(f"❌ Error tracking experiment results: {str(e)}")
    
    # Final summary
    print("\n🎉 Phase 2 Processing Complete!")
    print("=" * 60)
    print(f"📁 Processed data saved to: {processed_path}")
    print(f"🔬 Experiment ID: {experiment_manager.current_experiment.experiment_id}")
    print(f"📊 Ready for Phase 2 Analysis:")
    print(f"   • Study 1: Latent Factor Discovery")
    print(f"   • Study 2: Temporal Dynamics Analysis") 
    print(f"   • Study 3: Cultural Bridge Detection")
    print("\n🚀 Next Steps:")
    print("   1. Run latent factor analysis on engineered data")
    print("   2. Apply statistical hypothesis testing suite")
    print("   3. Analyze preference evolution patterns")
    print("   4. Detect cultural bridge mechanisms")


def create_dummy_audio_features(track_ids):
    """Create dummy audio features for tracks (placeholder for real Spotify API data)"""
    
    import numpy as np
    import pandas as pd
    
    # Set random seed for reproducible dummy data
    np.random.seed(42)
    
    audio_features = []
    
    for track_id in track_ids:
        if track_id and track_id != 'nan':  # Skip invalid track IDs
            features = {
                'track_id': track_id,
                'danceability': np.random.beta(2, 2),  # 0-1 range with realistic distribution
                'energy': np.random.beta(2, 2),
                'key': np.random.randint(0, 12),
                'loudness': np.random.normal(-8, 4),  # dB
                'mode': np.random.choice([0, 1]),
                'speechiness': np.random.beta(1, 5),  # Usually low for music
                'acousticness': np.random.beta(1, 3),
                'instrumentalness': np.random.beta(1, 5),
                'liveness': np.random.beta(1, 4),
                'valence': np.random.beta(2, 2),
                'tempo': np.random.normal(120, 30),  # BPM
                'duration_ms': np.random.normal(210000, 60000)  # Around 3.5 minutes
            }
            
            # Ensure realistic ranges
            features['loudness'] = max(-60, min(0, features['loudness']))
            features['tempo'] = max(60, min(200, features['tempo']))
            features['duration_ms'] = max(30000, min(600000, features['duration_ms']))
            
            audio_features.append(features)
    
    return pd.DataFrame(audio_features)


def generate_initial_analysis_report(streaming_data, processing_stats, cultural_stats, feature_groups):
    """Generate comprehensive initial analysis report"""
    
    report = {
        'processing_summary': {
            'total_records': processing_stats.total_records,
            'date_range': {
                'start': processing_stats.date_range[0],
                'end': processing_stats.date_range[1],
                'duration_days': (processing_stats.date_range[1] - processing_stats.date_range[0]).days
            },
            'data_richness': {
                'unique_tracks': processing_stats.unique_tracks,
                'unique_artists': processing_stats.unique_artists,
                'total_listening_hours': processing_stats.total_listening_time_hours,
                'countries': processing_stats.countries,
                'platforms': processing_stats.platforms
            }
        }
    }
    
    # Listening patterns analysis
    if len(streaming_data) > 0:
        try:
            # Temporal patterns
            hourly_counts = streaming_data.groupby('hour').size()
            daily_counts = streaming_data.groupby('day_of_week').size()
            platform_counts = streaming_data.groupby('platform_type').size()
            location_counts = streaming_data.groupby('listening_location').size()
            
            report['listening_patterns'] = {
                'peak_hour': int(hourly_counts.idxmax()),
                'most_active_day': int(daily_counts.idxmax()),
                'primary_platform': platform_counts.idxmax(),
                'primary_location': location_counts.idxmax(),
                'hourly_distribution': hourly_counts.to_dict(),
                'daily_distribution': daily_counts.to_dict()
            }
            
            # Session analysis
            if 'session_id' in streaming_data.columns:
                session_stats = streaming_data.groupby('session_id').agg({
                    'ms_played': 'sum',
                    'track_id': 'count'
                })
                
                report['temporal_insights'] = {
                    'total_sessions': len(session_stats),
                    'avg_session_minutes': session_stats['ms_played'].mean() / (1000 * 60),
                    'avg_tracks_per_session': session_stats['track_id'].mean(),
                    'session_length_distribution': session_stats['ms_played'].describe().to_dict()
                }
            
        except Exception as e:
            report['listening_patterns'] = {'error': str(e)}
    
    # Cultural insights
    if cultural_stats:
        vietnamese_pct = cultural_stats.get('vietnamese_percentage', 0)
        western_pct = cultural_stats.get('western_percentage', 0)
        bridge_pct = cultural_stats.get('bridge_percentage', 0)
        
        # Determine cultural balance
        if vietnamese_pct > western_pct * 1.5:
            balance = "Vietnamese-dominant"
        elif western_pct > vietnamese_pct * 1.5:
            balance = "Western-dominant"
        else:
            balance = "Balanced cross-cultural"
        
        report['cultural_insights'] = {
            'cultural_balance': balance,
            'vietnamese_percentage': vietnamese_pct,
            'western_percentage': western_pct,
            'bridge_percentage': bridge_pct,
            'exploration_rate': bridge_pct / 100,  # Convert to rate
            'cultural_confidence': cultural_stats.get('avg_confidence', 0)
        }
    
    # Feature engineering summary
    if feature_groups:
        report['feature_engineering'] = {
            'total_feature_groups': len(feature_groups),
            'feature_counts': {k: len(v) for k, v in feature_groups.items()},
            'engineering_success': True
        }
    
    return report


if __name__ == "__main__":
    main()