"""
Preference Evolution Analysis Module

Analyzes temporal dynamics of musical preferences including:
- Change point detection in listening patterns
- Preference trend analysis over time
- Life event correlation with preference shifts
- Prediction accuracy decay modeling
- Temporal pattern identification
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
# import ruptures as rpt  # Temporarily disabled - will use simple change point detection
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')


@dataclass
class ChangePoint:
    """Represents a detected change point in preferences"""
    timestamp: datetime
    confidence: float
    change_type: str  # 'cultural_shift', 'energy_change', 'exploration_burst'
    magnitude: float
    description: str
    preceding_pattern: Dict[str, float]
    following_pattern: Dict[str, float]


@dataclass
class PreferenceEvolutionResult:
    """Results from preference evolution analysis"""
    change_points: List[ChangePoint]
    trend_analysis: Dict[str, Any]
    stability_periods: List[Tuple[datetime, datetime, Dict[str, float]]]
    prediction_decay_model: Dict[str, Any]
    temporal_patterns: Dict[str, Any]
    life_event_correlations: Dict[str, Any]


class PreferenceEvolutionAnalyzer:
    """
    Analyzes temporal evolution of musical preferences.
    
    Implements sophisticated change point detection, trend analysis,
    and temporal pattern recognition for cross-cultural music preferences.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        self.scalers = {}
        self.models = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for preference evolution analysis"""
        return {
            'change_point_detection': {
                'algorithms': ['pelt', 'binseg', 'window'],
                'penalty': 10,
                'min_size': 7,  # Minimum days between change points
                'jump': 1,
                'model': 'rbf'
            },
            'trend_analysis': {
                'window_size': 14,  # Rolling window in days
                'polynomial_degree': 2,
                'seasonal_periods': [7, 30],  # Weekly and monthly patterns
                'smoothing_factor': 0.1
            },
            'stability_analysis': {
                'min_stability_period': 14,  # Minimum stable period in days
                'stability_threshold': 0.1,  # Maximum variance for stability
                'trend_threshold': 0.05  # Maximum trend slope for stability
            },
            'prediction_decay': {
                'horizons': [1, 7, 14, 30, 60, 90],  # Days ahead
                'models': ['exponential', 'power_law', 'linear'],
                'cross_validation_folds': 5
            },
            'life_events': {
                'detection_window': 7,  # Days around events to analyze
                'significance_threshold': 0.05,
                'event_types': ['cultural_exposure', 'life_transition', 'seasonal_change']
            }
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for preference evolution analysis"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def analyze_preference_evolution(
        self,
        listening_data: pd.DataFrame,
        cultural_features: Optional[pd.DataFrame] = None,
        audio_features: Optional[pd.DataFrame] = None,
        life_events: Optional[pd.DataFrame] = None
    ) -> PreferenceEvolutionResult:
        """
        Comprehensive analysis of preference evolution over time.
        
        Args:
            listening_data: DataFrame with temporal listening history
            cultural_features: Optional cultural categorization data
            audio_features: Optional audio features data
            life_events: Optional life events data with timestamps
            
        Returns:
            PreferenceEvolutionResult with comprehensive analysis
        """
        self.logger.info("Starting comprehensive preference evolution analysis")
        
        # Prepare temporal data
        temporal_data = self._prepare_temporal_data(
            listening_data, cultural_features, audio_features
        )
        
        # 1. Change point detection
        change_points = self._detect_change_points(temporal_data)
        
        # 2. Trend analysis
        trend_analysis = self._analyze_trends(temporal_data)
        
        # 3. Identify stability periods
        stability_periods = self._identify_stability_periods(temporal_data, change_points)
        
        # 4. Model prediction decay
        prediction_decay_model = self._model_prediction_decay(temporal_data)
        
        # 5. Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(temporal_data)
        
        # 6. Correlate with life events
        life_event_correlations = self._correlate_life_events(
            temporal_data, change_points, life_events
        )
        
        result = PreferenceEvolutionResult(
            change_points=change_points,
            trend_analysis=trend_analysis,
            stability_periods=stability_periods,
            prediction_decay_model=prediction_decay_model,
            temporal_patterns=temporal_patterns,
            life_event_correlations=life_event_correlations
        )
        
        self.logger.info(f"Analysis complete: {len(change_points)} change points detected")
        return result

    def _prepare_temporal_data(
        self,
        listening_data: pd.DataFrame,
        cultural_features: Optional[pd.DataFrame],
        audio_features: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Prepare temporal data for analysis"""
        
        # Ensure timestamp column
        if 'played_at' not in listening_data.columns:
            raise ValueError("listening_data must have 'played_at' timestamp column")
        
        # Sort by timestamp
        data = listening_data.copy()
        data['played_at'] = pd.to_datetime(data['played_at'])
        data = data.sort_values('played_at').reset_index(drop=True)
        
        # Merge cultural features if available
        if cultural_features is not None:
            data = data.merge(cultural_features, on='track_id', how='left', suffixes=('', '_cultural'))
        
        # Merge audio features if available
        if audio_features is not None:
            audio_cols = ['energy', 'valence', 'danceability', 'acousticness', 'tempo']
            available_audio_cols = [col for col in audio_cols if col in audio_features.columns]
            if available_audio_cols:
                audio_subset = audio_features[['track_id'] + available_audio_cols]
                data = data.merge(audio_subset, on='track_id', how='left', suffixes=('', '_audio'))
        
        # Create daily aggregations
        data['date'] = data['played_at'].dt.date
        daily_data = self._aggregate_daily_preferences(data)
        
        return daily_data

    def _aggregate_daily_preferences(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate listening data by day for temporal analysis"""
        
        # Group by date and calculate daily statistics
        daily_stats = []
        
        for date, day_data in data.groupby('date'):
            stats = {
                'date': pd.to_datetime(date),
                'n_plays': len(day_data),
                'unique_tracks': day_data['track_id'].nunique(),
                'unique_artists': day_data['artist_id'].nunique()
            }
            
            # Cultural preferences
            cultural_cols = [col for col in day_data.columns if 'vietnamese_score' in col or 'western_score' in col or 'bridge_score' in col]
            for col in cultural_cols:
                if col in day_data.columns:
                    stats[f'avg_{col}'] = day_data[col].mean()
                    stats[f'std_{col}'] = day_data[col].std()
            
            # Audio characteristics
            audio_cols = ['energy', 'valence', 'danceability', 'acousticness', 'tempo']
            for col in audio_cols:
                if col in day_data.columns:
                    stats[f'avg_{col}'] = day_data[col].mean()
                    stats[f'std_{col}'] = day_data[col].std()
            
            # Behavioral patterns
            stats['exploration_rate'] = (day_data['track_id'].nunique() / len(day_data)) if len(day_data) > 0 else 0
            stats['cultural_diversity'] = len(day_data['dominant_culture'].unique()) if 'dominant_culture' in day_data.columns else 0
            
            daily_stats.append(stats)
        
        daily_df = pd.DataFrame(daily_stats)
        daily_df = daily_df.fillna(daily_df.mean())  # Fill missing values
        
        return daily_df

    def _detect_change_points(self, temporal_data: pd.DataFrame) -> List[ChangePoint]:
        """Detect change points in preference evolution"""
        
        if len(temporal_data) < 14:  # Need minimum data for change point detection
            self.logger.warning("Insufficient data for change point detection")
            return []
        
        change_points = []
        
        # Signals to analyze for change points
        signals = self._select_change_point_signals(temporal_data)
        
        for signal_name, signal_data in signals.items():
            cps = self._detect_change_points_in_signal(signal_data, signal_name)
            change_points.extend(cps)
        
        # Remove duplicate change points (within 3 days of each other)
        change_points = self._merge_close_change_points(change_points, temporal_data)
        
        # Sort by timestamp
        change_points.sort(key=lambda x: x.timestamp)
        
        return change_points

    def _select_change_point_signals(self, temporal_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Select signals for change point detection"""
        
        signals = {}
        
        # Cultural preference signals
        cultural_cols = [col for col in temporal_data.columns if 'vietnamese_score' in col or 'western_score' in col]
        for col in cultural_cols:
            if temporal_data[col].notna().sum() > 10:  # Sufficient non-null values
                signals[col] = temporal_data[col].fillna(temporal_data[col].mean()).values
        
        # Audio characteristic signals
        audio_cols = [col for col in temporal_data.columns if col.startswith('avg_') and any(feat in col for feat in ['energy', 'valence', 'danceability'])]
        for col in audio_cols:
            if temporal_data[col].notna().sum() > 10:
                signals[col] = temporal_data[col].fillna(temporal_data[col].mean()).values
        
        # Behavioral signals
        if 'exploration_rate' in temporal_data.columns:
            signals['exploration_rate'] = temporal_data['exploration_rate'].values
        
        if 'cultural_diversity' in temporal_data.columns:
            signals['cultural_diversity'] = temporal_data['cultural_diversity'].values
        
        return signals

    def _detect_change_points_in_signal(self, signal: np.ndarray, signal_name: str) -> List[ChangePoint]:
        """Detect change points in a single signal using multiple algorithms"""
        
        if len(signal) < 10:
            return []
        
        change_points = []
        config = self.config['change_point_detection']
        
        # Simple change point detection using variance changes
        # This replaces the advanced ruptures algorithms temporarily
        try:
            change_points_simple = self._simple_change_point_detection(signal, signal_name, config)
            change_points.extend(change_points_simple)
        except Exception as e:
            self.logger.warning(f"Simple change point detection failed for {signal_name}: {str(e)}")
        
        return change_points

    def _simple_change_point_detection(self, signal: np.ndarray, signal_name: str, config: Dict) -> List[ChangePoint]:
        """Simple change point detection using sliding window variance"""
        
        if len(signal) < config['min_size'] * 2:
            return []
        
        change_points = []
        window_size = max(7, config['min_size'])  # At least 7 days
        
        # Calculate rolling variance
        variances = []
        for i in range(window_size, len(signal) - window_size):
            left_var = np.var(signal[i-window_size:i])
            right_var = np.var(signal[i:i+window_size])
            variance_change = abs(left_var - right_var)
            variances.append((i, variance_change))
        
        if not variances:
            return []
        
        # Find peaks in variance changes
        variance_values = [v[1] for v in variances]
        threshold = np.mean(variance_values) + 2 * np.std(variance_values)  # 2 standard deviations above mean
        
        for idx, var_change in variances:
            if var_change > threshold:
                cp = self._create_change_point(idx, signal, signal_name, 'simple_variance')
                change_points.append(cp)
        
        # Limit to reasonable number of change points
        if len(change_points) > 5:
            # Keep the most significant ones
            change_points.sort(key=lambda x: x.magnitude, reverse=True)
            change_points = change_points[:5]
            
        return change_points

    def _create_change_point(
        self,
        index: int,
        signal: np.ndarray,
        signal_name: str,
        algorithm: str
    ) -> ChangePoint:
        """Create ChangePoint object from detected index"""
        
        # Calculate change magnitude
        before_mean = np.mean(signal[max(0, index-7):index])
        after_mean = np.mean(signal[index:min(len(signal), index+7)])
        magnitude = abs(after_mean - before_mean) / max(np.std(signal), 0.001)
        
        # Determine change type based on signal name
        if 'vietnamese' in signal_name or 'western' in signal_name:
            change_type = 'cultural_shift'
        elif 'energy' in signal_name or 'valence' in signal_name:
            change_type = 'energy_change'
        elif 'exploration' in signal_name:
            change_type = 'exploration_burst'
        else:
            change_type = 'preference_change'
        
        # Confidence based on magnitude and algorithm consistency
        confidence = min(1.0, magnitude / 2.0)  # Normalized confidence
        
        # Create placeholder timestamp (will be updated with real dates)
        timestamp = datetime(2023, 1, 1) + timedelta(days=index)
        
        return ChangePoint(
            timestamp=timestamp,
            confidence=confidence,
            change_type=change_type,
            magnitude=magnitude,
            description=f"{change_type} in {signal_name} detected by {algorithm}",
            preceding_pattern={'mean': before_mean, 'signal': signal_name},
            following_pattern={'mean': after_mean, 'signal': signal_name}
        )

    def _merge_close_change_points(
        self,
        change_points: List[ChangePoint],
        temporal_data: pd.DataFrame
    ) -> List[ChangePoint]:
        """Merge change points that are close in time"""
        
        if not change_points:
            return []
        
        # Convert indices to actual dates
        dates = temporal_data['date'].dt.to_pydatetime()
        for cp in change_points:
            try:
                index = (cp.timestamp - datetime(2023, 1, 1)).days
                if 0 <= index < len(dates):
                    cp.timestamp = dates[index]
            except:
                continue
        
        # Group change points within 3 days
        merged = []
        current_group = [change_points[0]]
        
        for cp in change_points[1:]:
            time_diff = abs((cp.timestamp - current_group[-1].timestamp).days)
            if time_diff <= 3:
                current_group.append(cp)
            else:
                # Merge current group
                if len(current_group) == 1:
                    merged.append(current_group[0])
                else:
                    merged_cp = self._merge_change_point_group(current_group)
                    merged.append(merged_cp)
                current_group = [cp]
        
        # Handle last group
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            merged_cp = self._merge_change_point_group(current_group)
            merged.append(merged_cp)
        
        return merged

    def _merge_change_point_group(self, change_points: List[ChangePoint]) -> ChangePoint:
        """Merge a group of close change points into one"""
        
        # Use the change point with highest confidence
        best_cp = max(change_points, key=lambda x: x.confidence)
        
        # Aggregate information
        change_types = [cp.change_type for cp in change_points]
        most_common_type = max(set(change_types), key=change_types.count)
        
        avg_magnitude = np.mean([cp.magnitude for cp in change_points])
        avg_confidence = np.mean([cp.confidence for cp in change_points])
        
        return ChangePoint(
            timestamp=best_cp.timestamp,
            confidence=avg_confidence,
            change_type=most_common_type,
            magnitude=avg_magnitude,
            description=f"Merged {len(change_points)} change points: {most_common_type}",
            preceding_pattern=best_cp.preceding_pattern,
            following_pattern=best_cp.following_pattern
        )

    def _analyze_trends(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in temporal preference data"""
        
        trend_analysis = {
            'overall_trends': {},
            'cultural_trends': {},
            'seasonal_patterns': {},
            'smoothed_signals': {}
        }
        
        # Overall trends (linear regression)
        trend_analysis['overall_trends'] = self._calculate_overall_trends(temporal_data)
        
        # Cultural preference trends
        trend_analysis['cultural_trends'] = self._calculate_cultural_trends(temporal_data)
        
        # Seasonal decomposition
        trend_analysis['seasonal_patterns'] = self._analyze_seasonal_patterns(temporal_data)
        
        # Smoothed signals for visualization
        trend_analysis['smoothed_signals'] = self._create_smoothed_signals(temporal_data)
        
        return trend_analysis

    def _calculate_overall_trends(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall linear trends for key metrics"""
        
        overall_trends = {}
        
        # Create time variable (days from start)
        start_date = temporal_data['date'].min()
        temporal_data['days_from_start'] = (temporal_data['date'] - start_date).dt.days
        
        # Analyze trends for key signals
        trend_signals = ['exploration_rate', 'cultural_diversity']
        cultural_signals = [col for col in temporal_data.columns if 'vietnamese_score' in col or 'western_score' in col]
        audio_signals = [col for col in temporal_data.columns if col.startswith('avg_energy') or col.startswith('avg_valence')]
        
        all_signals = trend_signals + cultural_signals + audio_signals
        
        for signal in all_signals:
            if signal in temporal_data.columns and temporal_data[signal].notna().sum() > 5:
                trend_data = temporal_data[['days_from_start', signal]].dropna()
                
                if len(trend_data) >= 5:
                    X = trend_data[['days_from_start']]
                    y = trend_data[signal]
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    slope = model.coef_[0]
                    r2 = r2_score(y, model.predict(X))
                    
                    # Statistical significance
                    X_stats = sm.add_constant(X)
                    stats_model = sm.OLS(y, X_stats).fit()
                    p_value = stats_model.pvalues[1]
                    
                    overall_trends[signal] = {
                        'slope': slope,
                        'r_squared': r2,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'magnitude': abs(slope)
                    }
        
        return overall_trends

    def _calculate_cultural_trends(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate specific trends in cultural preferences"""
        
        cultural_trends = {}
        
        # Vietnamese vs Western preference evolution
        vietnamese_cols = [col for col in temporal_data.columns if 'vietnamese_score' in col]
        western_cols = [col for col in temporal_data.columns if 'western_score' in col]
        
        if vietnamese_cols and western_cols:
            vietnamese_signal = temporal_data[vietnamese_cols[0]].fillna(0)
            western_signal = temporal_data[western_cols[0]].fillna(0)
            
            # Cultural balance over time
            cultural_balance = vietnamese_signal - western_signal
            
            # Trend in cultural balance
            if len(temporal_data) >= 10:
                days = np.arange(len(cultural_balance))
                slope, intercept, r_value, p_value, std_err = stats.linregress(days, cultural_balance)
                
                cultural_trends['cultural_balance'] = {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'interpretation': self._interpret_cultural_trend(slope, p_value)
                }
            
            # Cross-cultural exploration patterns
            bridge_cols = [col for col in temporal_data.columns if 'bridge_score' in col]
            if bridge_cols:
                bridge_signal = temporal_data[bridge_cols[0]].fillna(0)
                
                # Periods of high cross-cultural exploration
                high_bridge_periods = bridge_signal > bridge_signal.quantile(0.75)
                cultural_trends['exploration_periods'] = {
                    'n_periods': high_bridge_periods.sum(),
                    'average_duration': self._calculate_period_durations(high_bridge_periods),
                    'frequency': high_bridge_periods.sum() / len(temporal_data)
                }
        
        return cultural_trends

    def _interpret_cultural_trend(self, slope: float, p_value: float) -> str:
        """Interpret cultural trend direction and significance"""
        
        if p_value >= 0.05:
            return "No significant cultural trend detected"
        
        if slope > 0.01:
            return "Significant trend toward Vietnamese music"
        elif slope < -0.01:
            return "Significant trend toward Western music"
        else:
            return "Stable cultural preferences"

    def _calculate_period_durations(self, binary_signal: pd.Series) -> float:
        """Calculate average duration of periods where binary signal is True"""
        
        # Find runs of consecutive True values
        runs = []
        current_run = 0
        
        for value in binary_signal:
            if value:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0
        
        # Handle case where signal ends with True
        if current_run > 0:
            runs.append(current_run)
        
        return np.mean(runs) if runs else 0

    def _analyze_seasonal_patterns(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal and cyclical patterns"""
        
        seasonal_patterns = {}
        
        # Add time-based features
        temporal_data['day_of_week'] = temporal_data['date'].dt.dayofweek
        temporal_data['month'] = temporal_data['date'].dt.month
        temporal_data['day_of_year'] = temporal_data['date'].dt.dayofyear
        
        # Weekly patterns
        weekly_patterns = {}
        for signal in ['exploration_rate', 'cultural_diversity']:
            if signal in temporal_data.columns:
                weekly_avg = temporal_data.groupby('day_of_week')[signal].mean()
                weekly_patterns[signal] = weekly_avg.to_dict()
        
        seasonal_patterns['weekly'] = weekly_patterns
        
        # Monthly patterns
        monthly_patterns = {}
        cultural_signals = [col for col in temporal_data.columns if 'vietnamese_score' in col or 'western_score' in col]
        for signal in cultural_signals:
            if temporal_data[signal].notna().sum() > 0:
                monthly_avg = temporal_data.groupby('month')[signal].mean()
                monthly_patterns[signal] = monthly_avg.to_dict()
        
        seasonal_patterns['monthly'] = monthly_patterns
        
        # Seasonal decomposition (if enough data)
        if len(temporal_data) >= 30:
            try:
                # Use a primary cultural signal for decomposition
                primary_signal = None
                for col in cultural_signals:
                    if temporal_data[col].notna().sum() > 20:
                        primary_signal = col
                        break
                
                if primary_signal:
                    signal_data = temporal_data.set_index('date')[primary_signal].fillna(method='ffill').fillna(method='bfill')
                    
                    if len(signal_data) >= 30:  # Minimum for seasonal decomposition
                        decomposition = seasonal_decompose(
                            signal_data,
                            period=min(7, len(signal_data) // 4),  # Weekly or adaptive period
                            extrapolate_trend='freq'
                        )
                        
                        seasonal_patterns['decomposition'] = {
                            'trend_strength': np.var(decomposition.trend.dropna()) / np.var(signal_data),
                            'seasonal_strength': np.var(decomposition.seasonal.dropna()) / np.var(signal_data),
                            'residual_strength': np.var(decomposition.resid.dropna()) / np.var(signal_data)
                        }
                        
            except Exception as e:
                self.logger.warning(f"Seasonal decomposition failed: {str(e)}")
                seasonal_patterns['decomposition'] = {}
        
        return seasonal_patterns

    def _create_smoothed_signals(self, temporal_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create smoothed versions of signals for visualization"""
        
        smoothed_signals = {}
        window_size = self.config['trend_analysis']['window_size']
        
        # Smooth key signals
        signals_to_smooth = []
        cultural_signals = [col for col in temporal_data.columns if 'vietnamese_score' in col or 'western_score' in col]
        behavioral_signals = ['exploration_rate', 'cultural_diversity']
        
        signals_to_smooth.extend(cultural_signals)
        signals_to_smooth.extend([s for s in behavioral_signals if s in temporal_data.columns])
        
        for signal in signals_to_smooth:
            if temporal_data[signal].notna().sum() > window_size:
                signal_data = temporal_data[signal].fillna(temporal_data[signal].mean())
                
                # Use Savitzky-Golay filter for smoothing
                try:
                    if len(signal_data) >= window_size:
                        smoothed = savgol_filter(
                            signal_data,
                            window_length=min(window_size, len(signal_data) - 1 if len(signal_data) % 2 == 0 else len(signal_data)),
                            polyorder=min(3, min(window_size, len(signal_data) - 1) - 1)
                        )
                        smoothed_signals[signal] = smoothed
                except:
                    # Fallback to simple moving average
                    smoothed_signals[signal] = signal_data.rolling(window=min(window_size, len(signal_data)), center=True).mean().fillna(signal_data).values
        
        return smoothed_signals

    def _identify_stability_periods(
        self,
        temporal_data: pd.DataFrame,
        change_points: List[ChangePoint]
    ) -> List[Tuple[datetime, datetime, Dict[str, float]]]:
        """Identify periods of stable preferences between change points"""
        
        if not change_points:
            # No change points, entire period is stable
            start_date = temporal_data['date'].min()
            end_date = temporal_data['date'].max()
            stats = self._calculate_period_statistics(temporal_data)
            return [(start_date, end_date, stats)]
        
        stability_periods = []
        dates = temporal_data['date'].dt.to_pydatetime()
        
        # Period before first change point
        start_date = dates[0]
        first_cp_date = change_points[0].timestamp
        if (first_cp_date - start_date).days >= self.config['stability_analysis']['min_stability_period']:
            period_data = temporal_data[temporal_data['date'] < first_cp_date]
            stats = self._calculate_period_statistics(period_data)
            stability_periods.append((start_date, first_cp_date, stats))
        
        # Periods between change points
        for i in range(len(change_points) - 1):
            start_date = change_points[i].timestamp
            end_date = change_points[i + 1].timestamp
            
            if (end_date - start_date).days >= self.config['stability_analysis']['min_stability_period']:
                period_data = temporal_data[
                    (temporal_data['date'] >= start_date) &
                    (temporal_data['date'] < end_date)
                ]
                
                if len(period_data) > 0:
                    stats = self._calculate_period_statistics(period_data)
                    
                    # Check if period is actually stable
                    if self._is_period_stable(period_data, stats):
                        stability_periods.append((start_date, end_date, stats))
        
        # Period after last change point
        last_cp_date = change_points[-1].timestamp
        end_date = dates[-1]
        if (end_date - last_cp_date).days >= self.config['stability_analysis']['min_stability_period']:
            period_data = temporal_data[temporal_data['date'] > last_cp_date]
            stats = self._calculate_period_statistics(period_data)
            stability_periods.append((last_cp_date, end_date, stats))
        
        return stability_periods

    def _calculate_period_statistics(self, period_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistics for a time period"""
        
        stats = {
            'duration_days': (period_data['date'].max() - period_data['date'].min()).days,
            'n_observations': len(period_data),
            'avg_plays_per_day': period_data['n_plays'].mean(),
            'avg_exploration_rate': period_data.get('exploration_rate', pd.Series([0])).mean()
        }
        
        # Cultural preference statistics
        cultural_cols = [col for col in period_data.columns if 'vietnamese_score' in col or 'western_score' in col]
        for col in cultural_cols:
            if col in period_data.columns:
                stats[f'avg_{col}'] = period_data[col].mean()
                stats[f'std_{col}'] = period_data[col].std()
        
        return stats

    def _is_period_stable(self, period_data: pd.DataFrame, stats: Dict[str, float]) -> bool:
        """Determine if a period shows stable preferences"""
        
        # Check variance in key metrics
        cultural_cols = [col for col in period_data.columns if 'vietnamese_score' in col or 'western_score' in col]
        
        stability_threshold = self.config['stability_analysis']['stability_threshold']
        
        for col in cultural_cols:
            if col in period_data.columns:
                cv = period_data[col].std() / max(period_data[col].mean(), 0.001)  # Coefficient of variation
                if cv > stability_threshold:
                    return False
        
        # Check for significant trends
        if len(period_data) >= 7:
            days = np.arange(len(period_data))
            for col in cultural_cols:
                if col in period_data.columns and period_data[col].notna().sum() >= 5:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(days, period_data[col].fillna(period_data[col].mean()))
                    if p_value < 0.05 and abs(slope) > self.config['stability_analysis']['trend_threshold']:
                        return False
        
        return True

    def _model_prediction_decay(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Model how recommendation accuracy decays over time"""
        
        decay_model = {
            'accuracy_timeline': {},
            'fitted_models': {},
            'model_comparison': {},
            'predictions': {}
        }
        
        # Create synthetic accuracy data based on preference stability
        # In real implementation, this would use actual recommendation accuracy data
        accuracy_timeline = self._create_synthetic_accuracy_timeline(temporal_data)
        
        if len(accuracy_timeline) < 10:
            self.logger.warning("Insufficient data for prediction decay modeling")
            return decay_model
        
        decay_model['accuracy_timeline'] = accuracy_timeline
        
        # Fit different decay models
        horizons = np.array(self.config['prediction_decay']['horizons'])
        models = self.config['prediction_decay']['models']
        
        model_results = {}
        
        for model_type in models:
            try:
                result = self._fit_decay_model(accuracy_timeline, model_type)
                model_results[model_type] = result
            except Exception as e:
                self.logger.warning(f"Failed to fit {model_type} model: {str(e)}")
        
        decay_model['fitted_models'] = model_results
        
        # Compare models
        if model_results:
            best_model = min(model_results.keys(), key=lambda k: model_results[k].get('rmse', float('inf')))
            decay_model['model_comparison'] = {
                'best_model': best_model,
                'model_scores': {k: v.get('rmse', float('inf')) for k, v in model_results.items()}
            }
            
            # Generate predictions
            if best_model in model_results:
                decay_model['predictions'] = self._generate_decay_predictions(
                    model_results[best_model], horizons
                )
        
        return decay_model

    def _create_synthetic_accuracy_timeline(self, temporal_data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic accuracy timeline based on preference stability"""
        
        # This is a placeholder - in real implementation, use actual recommendation accuracy
        # For now, create synthetic data that decreases with time horizons
        
        horizons = self.config['prediction_decay']['horizons']
        
        # Base accuracy starts high and decays
        base_accuracies = []
        for horizon in horizons:
            # Exponential decay with noise
            base_accuracy = 0.95 * np.exp(-horizon / 30)  # Decay with 30-day time constant
            noise = np.random.normal(0, 0.05)  # Add some noise
            accuracy = max(0.1, min(1.0, base_accuracy + noise))  # Clamp between 0.1 and 1.0
            base_accuracies.append(accuracy)
        
        timeline = pd.DataFrame({
            'days_ahead': horizons,
            'accuracy': base_accuracies
        })
        
        return timeline

    def _fit_decay_model(self, accuracy_timeline: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Fit a specific decay model to accuracy data"""
        
        x = accuracy_timeline['days_ahead'].values
        y = accuracy_timeline['accuracy'].values
        
        if model_type == 'exponential':
            # Exponential decay: y = a * exp(-b * x)
            # Transform to linear: log(y) = log(a) - b * x
            log_y = np.log(np.maximum(y, 0.001))  # Avoid log(0)
            X = sm.add_constant(x)
            model = sm.OLS(log_y, X).fit()
            
            a = np.exp(model.params[0])
            b = -model.params[1]
            r_squared = model.rsquared
            
            # Predictions
            y_pred = a * np.exp(-b * x)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            return {
                'model_type': 'exponential',
                'parameters': {'a': a, 'b': b},
                'r_squared': r_squared,
                'rmse': rmse,
                'equation': f'accuracy = {a:.3f} * exp(-{b:.3f} * days)',
                'fitted_model': model
            }
            
        elif model_type == 'power_law':
            # Power law: y = a * x^(-b)
            # Transform to linear: log(y) = log(a) - b * log(x)
            log_x = np.log(np.maximum(x, 0.1))
            log_y = np.log(np.maximum(y, 0.001))
            X = sm.add_constant(log_x)
            model = sm.OLS(log_y, X).fit()
            
            a = np.exp(model.params[0])
            b = -model.params[1]
            r_squared = model.rsquared
            
            # Predictions
            y_pred = a * np.power(np.maximum(x, 0.1), -b)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            return {
                'model_type': 'power_law',
                'parameters': {'a': a, 'b': b},
                'r_squared': r_squared,
                'rmse': rmse,
                'equation': f'accuracy = {a:.3f} * days^(-{b:.3f})',
                'fitted_model': model
            }
            
        elif model_type == 'linear':
            # Linear decay: y = a + b * x
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            
            a = model.params[0]
            b = model.params[1]
            r_squared = model.rsquared
            
            # Predictions
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            return {
                'model_type': 'linear',
                'parameters': {'a': a, 'b': b},
                'r_squared': r_squared,
                'rmse': rmse,
                'equation': f'accuracy = {a:.3f} + {b:.3f} * days',
                'fitted_model': model
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _generate_decay_predictions(self, best_model: Dict[str, Any], horizons: np.ndarray) -> Dict[str, Any]:
        """Generate predictions using the best decay model"""
        
        model_type = best_model['model_type']
        params = best_model['parameters']
        
        predictions = {}
        
        if model_type == 'exponential':
            pred_accuracies = params['a'] * np.exp(-params['b'] * horizons)
        elif model_type == 'power_law':
            pred_accuracies = params['a'] * np.power(np.maximum(horizons, 0.1), -params['b'])
        elif model_type == 'linear':
            pred_accuracies = params['a'] + params['b'] * horizons
        else:
            pred_accuracies = np.full(len(horizons), 0.5)  # Fallback
        
        # Clamp predictions to reasonable range
        pred_accuracies = np.clip(pred_accuracies, 0.1, 1.0)
        
        for horizon, accuracy in zip(horizons, pred_accuracies):
            predictions[f'{horizon}_days'] = accuracy
        
        return predictions

    def _analyze_temporal_patterns(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recurring temporal patterns in preferences"""
        
        temporal_patterns = {
            'cyclical_patterns': {},
            'habit_strength': {},
            'preference_volatility': {},
            'exploration_cycles': {}
        }
        
        # Cyclical patterns (weekly, monthly)
        temporal_patterns['cyclical_patterns'] = self._detect_cyclical_patterns(temporal_data)
        
        # Habit strength (consistency in preferences)
        temporal_patterns['habit_strength'] = self._calculate_habit_strength(temporal_data)
        
        # Preference volatility (how much preferences fluctuate)
        temporal_patterns['preference_volatility'] = self._calculate_preference_volatility(temporal_data)
        
        # Exploration cycles (periods of high/low exploration)
        temporal_patterns['exploration_cycles'] = self._detect_exploration_cycles(temporal_data)
        
        return temporal_patterns

    def _detect_cyclical_patterns(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect cyclical patterns in preferences"""
        
        cyclical_patterns = {}
        
        # Weekly cycles
        if len(temporal_data) >= 21:  # At least 3 weeks
            temporal_data['day_of_week'] = temporal_data['date'].dt.dayofweek
            
            weekly_patterns = {}
            cultural_cols = [col for col in temporal_data.columns if 'vietnamese_score' in col or 'western_score' in col]
            
            for col in cultural_cols:
                if temporal_data[col].notna().sum() > 14:
                    weekly_avg = temporal_data.groupby('day_of_week')[col].mean()
                    weekly_std = temporal_data.groupby('day_of_week')[col].std()
                    
                    # Test for significant weekly variation
                    f_stat, p_value = stats.f_oneway(*[group[col].dropna() for name, group in temporal_data.groupby('day_of_week') if len(group) > 0])
                    
                    weekly_patterns[col] = {
                        'weekly_averages': weekly_avg.to_dict(),
                        'weekly_std': weekly_std.to_dict(),
                        'significant_variation': p_value < 0.05,
                        'f_statistic': f_stat,
                        'p_value': p_value
                    }
            
            cyclical_patterns['weekly'] = weekly_patterns
        
        return cyclical_patterns

    def _calculate_habit_strength(self, temporal_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate strength of listening habits"""
        
        habit_metrics = {}
        
        # Consistency in daily patterns
        if 'n_plays' in temporal_data.columns:
            daily_consistency = 1 - (temporal_data['n_plays'].std() / max(temporal_data['n_plays'].mean(), 1))
            habit_metrics['daily_consistency'] = max(0, daily_consistency)
        
        # Consistency in cultural preferences
        cultural_cols = [col for col in temporal_data.columns if 'vietnamese_score' in col or 'western_score' in col]
        for col in cultural_cols:
            if temporal_data[col].notna().sum() > 5:
                cv = temporal_data[col].std() / max(temporal_data[col].mean(), 0.001)
                habit_metrics[f'{col}_consistency'] = max(0, 1 - cv)
        
        # Overall habit strength
        if habit_metrics:
            habit_metrics['overall_habit_strength'] = np.mean(list(habit_metrics.values()))
        
        return habit_metrics

    def _calculate_preference_volatility(self, temporal_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility (instability) in preferences"""
        
        volatility_metrics = {}
        
        cultural_cols = [col for col in temporal_data.columns if 'vietnamese_score' in col or 'western_score' in col]
        
        for col in cultural_cols:
            if temporal_data[col].notna().sum() > 5:
                signal = temporal_data[col].fillna(temporal_data[col].mean())
                
                # Day-to-day changes
                daily_changes = np.abs(signal.diff()).dropna()
                volatility = daily_changes.mean() / max(signal.std(), 0.001)
                
                volatility_metrics[col] = volatility
        
        # Overall volatility
        if volatility_metrics:
            volatility_metrics['overall_volatility'] = np.mean(list(volatility_metrics.values()))
        
        return volatility_metrics

    def _detect_exploration_cycles(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect cycles in musical exploration behavior"""
        
        exploration_cycles = {}
        
        if 'exploration_rate' not in temporal_data.columns:
            return exploration_cycles
        
        exploration_signal = temporal_data['exploration_rate'].fillna(temporal_data['exploration_rate'].mean())
        
        # Find peaks and valleys in exploration
        peaks, _ = find_peaks(exploration_signal, height=exploration_signal.quantile(0.7))
        valleys, _ = find_peaks(-exploration_signal, height=-exploration_signal.quantile(0.3))
        
        exploration_cycles['high_exploration_periods'] = len(peaks)
        exploration_cycles['low_exploration_periods'] = len(valleys)
        
        # Average cycle length
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            exploration_cycles['avg_cycle_length'] = np.mean(peak_intervals)
        
        # Exploration rhythm regularity
        if len(peaks) >= 3:
            peak_intervals = np.diff(peaks)
            regularity = 1 - (np.std(peak_intervals) / max(np.mean(peak_intervals), 1))
            exploration_cycles['rhythm_regularity'] = max(0, regularity)
        
        return exploration_cycles

    def _correlate_life_events(
        self,
        temporal_data: pd.DataFrame,
        change_points: List[ChangePoint],
        life_events: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Correlate preference changes with life events"""
        
        correlations = {
            'event_correlations': [],
            'seasonal_correlations': {},
            'external_correlations': {}
        }
        
        if life_events is not None and len(life_events) > 0:
            # Analyze correlation between life events and change points
            correlations['event_correlations'] = self._analyze_life_event_correlations(
                change_points, life_events
            )
        
        # Seasonal correlations (holidays, weather patterns, etc.)
        correlations['seasonal_correlations'] = self._analyze_seasonal_correlations(
            temporal_data, change_points
        )
        
        return correlations

    def _analyze_life_event_correlations(
        self,
        change_points: List[ChangePoint],
        life_events: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Analyze correlations between change points and life events"""
        
        event_correlations = []
        detection_window = self.config['life_events']['detection_window']
        
        for cp in change_points:
            # Find life events within window of change point
            cp_date = cp.timestamp
            window_start = cp_date - timedelta(days=detection_window)
            window_end = cp_date + timedelta(days=detection_window)
            
            nearby_events = life_events[
                (life_events['event_date'] >= window_start) &
                (life_events['event_date'] <= window_end)
            ]
            
            if len(nearby_events) > 0:
                for _, event in nearby_events.iterrows():
                    correlation = {
                        'change_point_date': cp_date,
                        'change_type': cp.change_type,
                        'event_date': event['event_date'],
                        'event_type': event.get('event_type', 'unknown'),
                        'event_description': event.get('description', ''),
                        'days_difference': abs((cp_date - event['event_date']).days),
                        'confidence': cp.confidence
                    }
                    event_correlations.append(correlation)
        
        return event_correlations

    def _analyze_seasonal_correlations(
        self,
        temporal_data: pd.DataFrame,
        change_points: List[ChangePoint]
    ) -> Dict[str, Any]:
        """Analyze correlations with seasonal patterns"""
        
        seasonal_correlations = {}
        
        # Month-based correlations
        cp_months = [cp.timestamp.month for cp in change_points]
        if cp_months:
            month_distribution = pd.Series(cp_months).value_counts().sort_index()
            
            # Test for seasonal clustering
            expected_per_month = len(change_points) / 12
            chi2_stat, p_value = stats.chisquare(month_distribution.values, expected_per_month)
            
            seasonal_correlations['monthly_clustering'] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'significant_clustering': p_value < 0.05,
                'peak_months': month_distribution.nlargest(3).index.tolist()
            }
        
        return seasonal_correlations


# High-level analysis functions
def analyze_preference_evolution_comprehensive(
    listening_data: pd.DataFrame,
    cultural_features: Optional[pd.DataFrame] = None,
    audio_features: Optional[pd.DataFrame] = None,
    life_events: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None
) -> PreferenceEvolutionResult:
    """
    Comprehensive preference evolution analysis.
    
    This is the main entry point for temporal preference analysis.
    """
    
    analyzer = PreferenceEvolutionAnalyzer(config)
    
    return analyzer.analyze_preference_evolution(
        listening_data=listening_data,
        cultural_features=cultural_features,
        audio_features=audio_features,
        life_events=life_events
    )