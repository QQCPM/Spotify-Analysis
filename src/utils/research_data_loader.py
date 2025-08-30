"""
Research Data Loader with Robust Error Handling

Centralized data loading for research integrity.
Handles common failure modes gracefully with informative error messages.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
import logging
from datetime import datetime
import json

class DataLoadError(Exception):
    """Custom exception for data loading issues"""
    pass

class ResearchDataLoader:
    """
    Robust data loader for research projects.
    
    Handles common failure modes:
    - Missing files
    - Corrupted data
    - Memory issues
    - Schema validation
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for data operations"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_streaming_data(self, 
                           path: str = "data/processed/streaming_data_processed.parquet",
                           validate: bool = True) -> Optional[pd.DataFrame]:
        """
        Load streaming data with comprehensive error handling.
        
        Args:
            path: Path to streaming data file
            validate: Whether to run data validation
            
        Returns:
            DataFrame if successful, None if failed
            
        Raises:
            DataLoadError: If critical data integrity issues found
        """
        file_path = self.base_path / path
        
        try:
            # Check file exists
            if not file_path.exists():
                self.logger.error(f"❌ Streaming data not found: {file_path}")
                self.logger.info("💡 Try running: python process_spotify_data.py")
                return None
            
            # Check file size (warn if too large)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 500:  # More than 500MB
                self.logger.warning(f"⚠️ Large file detected: {file_size_mb:.1f}MB - may cause memory issues")
            
            # Load data
            self.logger.info(f"📂 Loading streaming data from {file_path}...")
            data = pd.read_parquet(file_path)
            
            # Basic validation
            if len(data) == 0:
                raise DataLoadError("Empty streaming data file")
            
            # Convert timestamp if needed
            if 'played_at' in data.columns:
                data['played_at'] = pd.to_datetime(data['played_at'], errors='coerce')
                invalid_timestamps = data['played_at'].isna().sum()
                if invalid_timestamps > 0:
                    self.logger.warning(f"⚠️ {invalid_timestamps} invalid timestamps found")
            
            self.logger.info(f"✅ Loaded {len(data):,} streaming records")
            
            # Run validation if requested
            if validate:
                self._validate_streaming_data(data)
            
            return data
            
        except pd.errors.ParquetError as e:
            self.logger.error(f"❌ Parquet file corrupted: {e}")
            self.logger.info("💡 Try re-processing the data")
            return None
            
        except MemoryError:
            self.logger.error("❌ Not enough memory to load data")
            self.logger.info("💡 Try increasing system memory or using data sampling")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Unexpected error loading streaming data: {e}")
            return None
    
    def load_playlist_data(self, 
                          playlist_dir: str = "/Users/quangnguyen/Downloads/spotify_playlists",
                          max_playlists: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Load playlist data with error handling.
        
        Args:
            playlist_dir: Directory containing CSV playlist files
            max_playlists: Limit number of playlists to load (for testing)
            
        Returns:
            Tuple of (combined_data, individual_playlists)
        """
        playlist_path = Path(playlist_dir)
        playlists = {}
        
        try:
            if not playlist_path.exists():
                self.logger.warning(f"⚠️ Playlist directory not found: {playlist_path}")
                return None, {}
            
            csv_files = list(playlist_path.glob('*.csv'))
            if not csv_files:
                self.logger.warning(f"⚠️ No CSV files found in {playlist_path}")
                return None, {}
            
            if max_playlists:
                csv_files = csv_files[:max_playlists]
                
            self.logger.info(f"📂 Loading {len(csv_files)} playlist files...")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['playlist_name'] = csv_file.stem
                    
                    # Convert timestamp if exists
                    if 'Added At' in df.columns:
                        df['Added At'] = pd.to_datetime(df['Added At'], errors='coerce')
                    
                    playlists[csv_file.stem] = df
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {csv_file.name}: {e}")
                    continue
            
            if playlists:
                combined_data = pd.concat(playlists.values(), ignore_index=True)
                self.logger.info(f"✅ Loaded {len(combined_data):,} playlist tracks from {len(playlists)} playlists")
                return combined_data, playlists
            else:
                self.logger.error("❌ No playlist data loaded successfully")
                return None, {}
                
        except Exception as e:
            self.logger.error(f"❌ Error loading playlist data: {e}")
            return None, {}
    
    def load_phase3_results(self, 
                           results_dir: str = "results/phase3") -> Optional[Dict]:
        """
        Load Phase 3 research results with error handling.
        
        Args:
            results_dir: Directory containing phase 3 results
            
        Returns:
            Dictionary of results or None if failed
        """
        results_path = self.base_path / results_dir
        
        try:
            if not results_path.exists():
                self.logger.error(f"❌ Phase 3 results not found: {results_path}")
                self.logger.info("💡 Try running: python phase3_deep_analysis.py")
                return None
            
            results = {}
            
            # Load JSON files
            json_files = list(results_path.glob('*.json'))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        results[json_file.stem] = json.load(f)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {json_file.name}: {e}")
            
            if results:
                self.logger.info(f"✅ Loaded Phase 3 results: {list(results.keys())}")
                return results
            else:
                self.logger.error("❌ No Phase 3 results found")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading Phase 3 results: {e}")
            return None
    
    def _validate_streaming_data(self, data: pd.DataFrame) -> None:
        """Validate streaming data for research integrity"""
        issues = []
        
        # Check required columns
        required_cols = ['track_id', 'played_at']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check audio features
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness']
        for feature in audio_features:
            if feature in data.columns:
                out_of_range = ((data[feature] < 0) | (data[feature] > 1)).sum()
                if out_of_range > 0:
                    issues.append(f"{out_of_range} out-of-range values in {feature}")
        
        # Check for duplicates
        if 'track_id' in data.columns:
            duplicates = data.duplicated(['track_id', 'played_at']).sum()
            if duplicates > 0:
                issues.append(f"{duplicates} duplicate listening events")
        
        # Check cultural classification
        if 'cultural_classification' in data.columns:
            valid_cultures = {'vietnamese', 'western', 'chinese', 'mixed'}
            invalid_cultures = set(data['cultural_classification'].unique()) - valid_cultures
            if invalid_cultures:
                issues.append(f"Invalid cultural classifications: {invalid_cultures}")
        
        # Report issues
        if issues:
            self.logger.warning("⚠️ Data validation issues found:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
            
            # Critical issues that should stop research
            critical_issues = [issue for issue in issues if 'Missing required' in issue]
            if critical_issues:
                raise DataLoadError(f"Critical data issues: {critical_issues}")
        else:
            self.logger.info("✅ Data validation passed")
    
    def get_data_info(self, data: pd.DataFrame) -> Dict:
        """Get comprehensive data information for research"""
        info = {
            'total_records': len(data),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'columns': list(data.columns),
            'date_range': None,
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        # Add date range if timestamp column exists
        if 'played_at' in data.columns:
            info['date_range'] = {
                'start': str(data['played_at'].min()),
                'end': str(data['played_at'].max()),
                'span_days': (data['played_at'].max() - data['played_at'].min()).days
            }
        
        return info

# Convenience function for quick loading
def load_research_data(base_path: Optional[str] = None, 
                      validate: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict]]:
    """
    Quick load function for common research data.
    
    Returns:
        Tuple of (streaming_data, playlist_data, phase3_results)
    """
    loader = ResearchDataLoader(base_path)
    
    streaming_data = loader.load_streaming_data(validate=validate)
    playlist_data, _ = loader.load_playlist_data()
    phase3_results = loader.load_phase3_results()
    
    return streaming_data, playlist_data, phase3_results

if __name__ == "__main__":
    # Test data loading
    print("🔬 Testing Research Data Loader...")
    loader = ResearchDataLoader()
    
    streaming_data = loader.load_streaming_data()
    if streaming_data is not None:
        info = loader.get_data_info(streaming_data)
        print(f"📊 Data Info: {info['total_records']} records, {info['memory_usage_mb']:.1f}MB")
    
    print("✅ Data loader test complete")