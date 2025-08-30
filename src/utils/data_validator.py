"""
Data Validation Utilities for Research Integrity

Validates that data meets research requirements and catches issues
that could invalidate research conclusions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result from data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, any]

class ResearchDataValidator:
    """Validates research data for integrity and quality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_streaming_data(self, data: pd.DataFrame) -> ValidationResult:
        """Comprehensive validation for streaming data"""
        errors = []
        warnings = []
        stats = {}
        
        # Basic structure checks
        if len(data) == 0:
            errors.append("Dataset is empty")
            return ValidationResult(False, errors, warnings, stats)
        
        # Required columns for research
        required_cols = ['track_id', 'played_at', 'artist_name', 'track_name']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Audio features validation
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness']
        for feature in audio_features:
            if feature in data.columns:
                # Check for valid range [0, 1]
                out_of_range = ((data[feature] < 0) | (data[feature] > 1)).sum()
                if out_of_range > 0:
                    warnings.append(f"{out_of_range} out-of-range values in {feature}")
                
                # Check for missing values
                missing = data[feature].isna().sum()
                if missing > 0:
                    warnings.append(f"{missing} missing values in {feature}")
                
                stats[f"{feature}_mean"] = data[feature].mean()
                stats[f"{feature}_std"] = data[feature].std()
        
        # Cultural classification validation
        if 'cultural_classification' in data.columns:
            valid_cultures = {'vietnamese', 'western', 'chinese', 'mixed'}
            unique_cultures = set(data['cultural_classification'].dropna().unique())
            invalid_cultures = unique_cultures - valid_cultures
            
            if invalid_cultures:
                warnings.append(f"Unexpected cultural classifications: {invalid_cultures}")
            
            # Check distribution
            culture_dist = data['cultural_classification'].value_counts(normalize=True)
            stats['cultural_distribution'] = culture_dist.to_dict()
            
            # Warn if heavily skewed (>90% one culture)
            max_culture_pct = culture_dist.max()
            if max_culture_pct > 0.9:
                warnings.append(f"Heavily skewed toward one culture: {max_culture_pct:.1%}")
        
        # Temporal validation
        if 'played_at' in data.columns:
            data_copy = data.copy()
            data_copy['played_at'] = pd.to_datetime(data_copy['played_at'], errors='coerce')
            
            invalid_dates = data_copy['played_at'].isna().sum()
            if invalid_dates > 0:
                warnings.append(f"{invalid_dates} invalid timestamps")
            
            # Check date range
            if not data_copy['played_at'].isna().all():
                min_date = data_copy['played_at'].min()
                max_date = data_copy['played_at'].max()
                date_span = (max_date - min_date).days
                
                stats['date_range'] = {
                    'start': str(min_date),
                    'end': str(max_date),
                    'span_days': date_span
                }
                
                # Warn if very short timespan for research
                if date_span < 30:
                    warnings.append(f"Short timespan: {date_span} days (research may need longer period)")
        
        # Duplicate detection
        if all(col in data.columns for col in ['track_id', 'played_at']):
            duplicates = data.duplicated(['track_id', 'played_at']).sum()
            if duplicates > 0:
                warnings.append(f"{duplicates} duplicate listening events")
                stats['duplicates'] = duplicates
        
        # Data quality metrics
        stats.update({
            'total_records': len(data),
            'total_unique_tracks': data['track_id'].nunique() if 'track_id' in data.columns else None,
            'total_unique_artists': data['artist_name'].nunique() if 'artist_name' in data.columns else None,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
        })
        
        # Overall assessment
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, stats)
    
    def validate_cultural_coherence(self, data: pd.DataFrame) -> ValidationResult:
        """Validate that cultural classifications make sense"""
        errors = []
        warnings = []
        stats = {}
        
        required_cols = ['vietnamese_score', 'western_score', 'cultural_classification']
        if not all(col in data.columns for col in required_cols):
            errors.append(f"Missing cultural analysis columns: {required_cols}")
            return ValidationResult(False, errors, warnings, stats)
        
        # Check score consistency with classification
        for culture in ['vietnamese', 'western']:
            culture_songs = data[data['cultural_classification'] == culture]
            if len(culture_songs) > 0:
                avg_score = culture_songs[f'{culture}_score'].mean()
                stats[f'{culture}_avg_score'] = avg_score
                
                # Cultural score should be higher for songs classified as that culture
                if avg_score < 0.5:
                    warnings.append(f"{culture} songs have low {culture}_score: {avg_score:.2f}")
        
        # Check for logical inconsistencies
        inconsistent = data[
            (data['cultural_classification'] == 'vietnamese') & 
            (data['vietnamese_score'] < data['western_score'])
        ]
        
        if len(inconsistent) > 0:
            warnings.append(f"{len(inconsistent)} Vietnamese songs with higher Western scores")
        
        return ValidationResult(True, errors, warnings, stats)
    
    def print_validation_report(self, result: ValidationResult, title: str = "Data Validation Report"):
        """Print a formatted validation report"""
        print(f"\n📊 {title}")
        print("=" * len(title))
        
        if result.is_valid:
            print("✅ Data validation PASSED")
        else:
            print("❌ Data validation FAILED")
        
        if result.errors:
            print(f"\n❌ ERRORS ({len(result.errors)}):")
            for error in result.errors:
                print(f"   - {error}")
        
        if result.warnings:
            print(f"\n⚠️ WARNINGS ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"   - {warning}")
        
        if result.stats:
            print(f"\n📈 STATISTICS:")
            for key, value in result.stats.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        print(f"     {k}: {v}")
                elif isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")

def validate_for_research(data: pd.DataFrame) -> bool:
    """Quick validation function for research use"""
    validator = ResearchDataValidator()
    
    result = validator.validate_streaming_data(data)
    validator.print_validation_report(result, "Research Data Validation")
    
    # Also validate cultural coherence if possible
    if all(col in data.columns for col in ['vietnamese_score', 'western_score', 'cultural_classification']):
        cultural_result = validator.validate_cultural_coherence(data)
        validator.print_validation_report(cultural_result, "Cultural Classification Validation")
        
        return result.is_valid and cultural_result.is_valid
    
    return result.is_valid

if __name__ == "__main__":
    # Test validation
    print("🔬 Testing Data Validator...")
    
    # Create test data
    test_data = pd.DataFrame({
        'track_id': [f'track_{i}' for i in range(100)],
        'played_at': pd.date_range('2023-01-01', periods=100, freq='H'),
        'artist_name': ['Test Artist'] * 100,
        'track_name': [f'Song {i}' for i in range(100)],
        'danceability': np.random.uniform(0, 1, 100),
        'energy': np.random.uniform(0, 1, 100),
        'valence': np.random.uniform(0, 1, 100),
        'vietnamese_score': np.random.uniform(0, 1, 100),
        'western_score': np.random.uniform(0, 1, 100),
        'cultural_classification': np.random.choice(['vietnamese', 'western', 'chinese'], 100)
    })
    
    is_valid = validate_for_research(test_data)
    print(f"\n✅ Validation test complete - Data valid: {is_valid}")