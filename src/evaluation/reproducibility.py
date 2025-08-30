"""
Reproducibility Management Module

Ensures reproducible research experiments through:
- Experiment configuration tracking and versioning
- Random seed management across all components
- Data version control and deterministic splits
- Results validation and comparison
- Statistical power analysis
- Environment tracking
- Reproducibility verification
"""

import os
import json
import hashlib
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.utils import check_random_state
import joblib
import yaml
import git  # For git commit tracking

warnings.filterwarnings('ignore')


@dataclass
class ExperimentConfig:
    """Configuration for a reproducible experiment"""
    experiment_id: str
    experiment_name: str
    description: str
    hypothesis: str
    random_seeds: Dict[str, int]
    parameters: Dict[str, Any]
    data_version: str
    model_configs: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    timestamp: str
    git_commit: Optional[str]
    environment_info: Dict[str, str]
    researcher_info: Dict[str, str]


@dataclass
class DataSplit:
    """Represents a data split with versioning and validation"""
    split_id: str
    split_type: str  # 'train_test', 'cross_validation', 'time_series'
    train_indices: np.ndarray
    test_indices: np.ndarray
    validation_indices: Optional[np.ndarray]
    split_parameters: Dict[str, Any]
    data_hash: str
    random_seed: int
    timestamp: str


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    experiment_id: str
    run_id: str
    results: Dict[str, Any]
    metrics: Dict[str, float]
    statistical_tests: Dict[str, Any]
    execution_time: float
    timestamp: str
    reproducibility_hash: str


class ReproducibilityManager:
    """
    Manages all aspects of reproducible research experiments.
    
    Ensures that every experiment can be exactly reproduced by tracking
    all relevant parameters, data versions, and computational environment.
    """
    
    def __init__(self, base_path: str = "experiments", config: Optional[Dict] = None):
        self.base_path = Path(base_path)
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        
        # Create directory structure
        self._setup_directory_structure()
        
        # Current experiment tracking
        self.current_experiment = None
        self.experiment_history = []
        
    def _default_config(self) -> Dict:
        """Default configuration for reproducibility management"""
        return {
            'random_seeds': {
                'global': 42,
                'data_split': 1337,
                'model_training': 2023,
                'evaluation': 999,
                'statistical_tests': 555
            },
            'data_validation': {
                'enable_checksums': True,
                'enable_schema_validation': True,
                'min_sample_size': 100
            },
            'experiment_tracking': {
                'auto_save_configs': True,
                'save_intermediate_results': True,
                'track_git_changes': True,
                'validate_environment': True
            },
            'reproducibility_checks': {
                'tolerance': 1e-6,
                'retry_failed_reproductions': 3,
                'validate_statistical_significance': True
            },
            'power_analysis': {
                'min_power': 0.8,
                'effect_sizes': [0.2, 0.5, 0.8],  # Small, medium, large
                'significance_level': 0.05
            }
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for reproducibility management"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def _setup_directory_structure(self):
        """Create directory structure for experiment tracking"""
        directories = [
            'configs',
            'data_splits', 
            'results',
            'models',
            'logs',
            'reproducibility_reports',
            'environment_snapshots'
        ]
        
        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        description: str,
        hypothesis: str,
        parameters: Dict[str, Any],
        researcher_info: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a new reproducible experiment.
        
        Args:
            name: Experiment name
            description: Detailed description
            hypothesis: Research hypothesis being tested
            parameters: Experiment parameters
            researcher_info: Information about researcher(s)
            
        Returns:
            Unique experiment ID
        """
        self.logger.info(f"Creating new experiment: {name}")
        
        # Generate unique experiment ID
        experiment_id = self._generate_experiment_id(name)
        
        # Get git commit information
        git_commit = self._get_git_commit()
        
        # Get environment information
        environment_info = self._get_environment_info()
        
        # Set random seeds
        random_seeds = self.config['random_seeds'].copy()
        self._set_global_random_seeds(random_seeds)
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=name,
            description=description,
            hypothesis=hypothesis,
            random_seeds=random_seeds,
            parameters=parameters,
            data_version="",  # Will be set when data is loaded
            model_configs={},
            evaluation_config={},
            timestamp=datetime.now().isoformat(),
            git_commit=git_commit,
            environment_info=environment_info,
            researcher_info=researcher_info or {}
        )
        
        # Save configuration
        self._save_experiment_config(experiment_config)
        
        # Set as current experiment
        self.current_experiment = experiment_config
        
        self.logger.info(f"Experiment created with ID: {experiment_id}")
        return experiment_id

    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{timestamp}_{name_hash}"

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha
        except Exception:
            self.logger.warning("Could not retrieve git commit information")
            return None

    def _get_environment_info(self) -> Dict[str, str]:
        """Get current environment information"""
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'system': platform.system(),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__
        }
        
        try:
            import sklearn
            env_info['sklearn_version'] = sklearn.__version__
        except ImportError:
            pass
            
        try:
            import torch
            env_info['torch_version'] = torch.__version__
        except ImportError:
            pass
        
        return env_info

    def _set_global_random_seeds(self, seeds: Dict[str, int]):
        """Set random seeds for all random number generators"""
        
        # Set Python random seed
        import random
        random.seed(seeds['global'])
        
        # Set NumPy seed
        np.random.seed(seeds['global'])
        
        # Set PyTorch seeds if available
        try:
            import torch
            torch.manual_seed(seeds['global'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seeds['global'])
        except ImportError:
            pass
        
        # Set environment variables for reproducibility
        os.environ['PYTHONHASHSEED'] = str(seeds['global'])

    def _save_experiment_config(self, config: ExperimentConfig):
        """Save experiment configuration to disk"""
        config_path = self.base_path / 'configs' / f"{config.experiment_id}.yaml"
        
        # Convert dataclass to dictionary
        config_dict = asdict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def load_experiment(self, experiment_id: str) -> ExperimentConfig:
        """Load experiment configuration from disk"""
        config_path = self.base_path / 'configs' / f"{experiment_id}.yaml"
        
        if not config_path.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        # Convert dictionary back to dataclass
        config = ExperimentConfig(**config_dict)
        
        # Set as current experiment
        self.current_experiment = config
        
        # Restore random seeds
        self._set_global_random_seeds(config.random_seeds)
        
        return config

    def create_deterministic_split(
        self,
        data: pd.DataFrame,
        split_type: str = 'train_test',
        test_size: float = 0.2,
        validation_size: Optional[float] = None,
        stratify_column: Optional[str] = None,
        time_column: Optional[str] = None,
        **kwargs
    ) -> DataSplit:
        """
        Create deterministic, reproducible data splits.
        
        Args:
            data: Input DataFrame
            split_type: Type of split ('train_test', 'cross_validation', 'time_series')
            test_size: Proportion for test set
            validation_size: Optional proportion for validation set
            stratify_column: Column for stratified splitting
            time_column: Column for time-series splitting
            **kwargs: Additional parameters for specific split types
            
        Returns:
            DataSplit object with indices and metadata
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        self.logger.info(f"Creating deterministic {split_type} split")
        
        # Generate data hash for validation
        data_hash = self._calculate_data_hash(data)
        
        # Update experiment with data version
        self.current_experiment.data_version = data_hash[:16]
        
        # Get random seed for splits
        random_seed = self.current_experiment.random_seeds['data_split']
        
        # Create split based on type
        if split_type == 'train_test':
            train_idx, test_idx, val_idx = self._create_train_test_split(
                data, test_size, validation_size, stratify_column, random_seed, **kwargs
            )
        elif split_type == 'time_series':
            train_idx, test_idx, val_idx = self._create_time_series_split(
                data, test_size, validation_size, time_column, **kwargs
            )
        elif split_type == 'cross_validation':
            # For cross-validation, we'll store the fold generator parameters
            train_idx, test_idx, val_idx = self._create_cv_split(
                data, random_seed, **kwargs
            )
        else:
            raise ValueError(f"Unsupported split type: {split_type}")
        
        # Create split object
        split_id = f"{self.current_experiment.experiment_id}_{split_type}_{datetime.now().strftime('%H%M%S')}"
        
        split = DataSplit(
            split_id=split_id,
            split_type=split_type,
            train_indices=train_idx,
            test_indices=test_idx,
            validation_indices=val_idx,
            split_parameters={
                'test_size': test_size,
                'validation_size': validation_size,
                'stratify_column': stratify_column,
                'time_column': time_column,
                **kwargs
            },
            data_hash=data_hash,
            random_seed=random_seed,
            timestamp=datetime.now().isoformat()
        )
        
        # Save split to disk
        self._save_data_split(split)
        
        # Validate split
        self._validate_data_split(split, data)
        
        return split

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data for version tracking"""
        # Create a hash based on data structure and content
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': str(data.dtypes.to_dict()),
            'sample': data.head(5).to_string() if len(data) > 0 else ""
        }
        
        data_string = json.dumps(data_info, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()

    def _create_train_test_split(
        self,
        data: pd.DataFrame,
        test_size: float,
        validation_size: Optional[float],
        stratify_column: Optional[str],
        random_seed: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Create train-test split"""
        
        indices = np.arange(len(data))
        
        # Stratification
        stratify = data[stratify_column] if stratify_column else None
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=stratify,
            random_state=random_seed,
            **kwargs
        )
        
        # Optional second split: train vs validation
        val_idx = None
        if validation_size is not None:
            # Adjust validation size relative to remaining data
            val_size_adjusted = validation_size / (1 - test_size)
            
            if stratify_column:
                stratify_remaining = data.iloc[train_val_idx][stratify_column]
            else:
                stratify_remaining = None
                
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size_adjusted,
                stratify=stratify_remaining,
                random_state=random_seed + 1,  # Different seed for second split
                **kwargs
            )
        else:
            train_idx = train_val_idx
        
        return train_idx, test_idx, val_idx

    def _create_time_series_split(
        self,
        data: pd.DataFrame,
        test_size: float,
        validation_size: Optional[float],
        time_column: str,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Create time-series split (temporal order preserved)"""
        
        if time_column is None or time_column not in data.columns:
            raise ValueError("time_column must be specified for time_series split")
        
        # Sort by time
        sorted_data = data.sort_values(time_column)
        indices = sorted_data.index.values
        
        n_samples = len(indices)
        n_test = int(n_samples * test_size)
        
        if validation_size:
            n_val = int(n_samples * validation_size)
            n_train = n_samples - n_test - n_val
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
        else:
            n_train = n_samples - n_test
            train_idx = indices[:n_train]
            val_idx = None
            test_idx = indices[n_train:]
        
        return train_idx, test_idx, val_idx

    def _create_cv_split(
        self,
        data: pd.DataFrame,
        random_seed: int,
        n_splits: int = 5,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, None]:
        """Create cross-validation split (returns first fold as example)"""
        
        # For CV, we return the parameters and first fold as example
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        
        indices = np.arange(len(data))
        folds = list(kf.split(indices))
        
        # Return first fold as example
        train_idx, test_idx = folds[0]
        
        return train_idx, test_idx, None

    def _save_data_split(self, split: DataSplit):
        """Save data split to disk"""
        split_path = self.base_path / 'data_splits' / f"{split.split_id}.pkl"
        
        with open(split_path, 'wb') as f:
            pickle.dump(split, f)

    def _validate_data_split(self, split: DataSplit, data: pd.DataFrame):
        """Validate data split for correctness"""
        
        # Check no overlap between sets
        train_set = set(split.train_indices)
        test_set = set(split.test_indices)
        
        if len(train_set.intersection(test_set)) > 0:
            raise ValueError("Train and test sets overlap")
        
        if split.validation_indices is not None:
            val_set = set(split.validation_indices)
            if len(train_set.intersection(val_set)) > 0 or len(test_set.intersection(val_set)) > 0:
                raise ValueError("Validation set overlaps with train or test")
        
        # Check all indices are valid
        all_indices = np.concatenate([
            split.train_indices,
            split.test_indices,
            split.validation_indices if split.validation_indices is not None else []
        ])
        
        if np.max(all_indices) >= len(data) or np.min(all_indices) < 0:
            raise ValueError("Invalid indices in data split")
        
        self.logger.info("Data split validation passed")

    def track_experiment_results(
        self,
        results: Dict[str, Any],
        metrics: Dict[str, float],
        statistical_tests: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> str:
        """
        Track results from experiment run.
        
        Args:
            results: Detailed results from experiment
            metrics: Key performance metrics
            statistical_tests: Statistical test results
            execution_time: Execution time in seconds
            
        Returns:
            Unique run ID
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        # Generate run ID
        run_id = f"{self.current_experiment.experiment_id}_run_{datetime.now().strftime('%H%M%S')}"
        
        # Calculate reproducibility hash
        reproducibility_hash = self._calculate_reproducibility_hash(results, metrics)
        
        # Create result object
        result = ExperimentResult(
            experiment_id=self.current_experiment.experiment_id,
            run_id=run_id,
            results=results,
            metrics=metrics,
            statistical_tests=statistical_tests or {},
            execution_time=execution_time or 0.0,
            timestamp=datetime.now().isoformat(),
            reproducibility_hash=reproducibility_hash
        )
        
        # Save results
        self._save_experiment_results(result)
        
        # Add to experiment history
        self.experiment_history.append(result)
        
        self.logger.info(f"Results tracked with run ID: {run_id}")
        return run_id

    def _calculate_reproducibility_hash(self, results: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """Calculate hash for reproducibility verification"""
        
        # Create reproducible representation
        repro_data = {
            'metrics': metrics,
            'key_results': self._extract_key_results(results)
        }
        
        repro_string = json.dumps(repro_data, sort_keys=True)
        return hashlib.md5(repro_string.encode()).hexdigest()

    def _extract_key_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key results for reproducibility hashing"""
        
        # Extract only deterministic, numerical results
        key_results = {}
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                key_results[key] = value
            elif isinstance(value, np.ndarray) and value.dtype in [np.float64, np.int64]:
                # Include array summary statistics
                key_results[f"{key}_mean"] = float(np.mean(value))
                key_results[f"{key}_std"] = float(np.std(value))
            elif isinstance(value, dict):
                # Recursively extract from nested dictionaries
                nested_results = self._extract_key_results(value)
                for nested_key, nested_value in nested_results.items():
                    key_results[f"{key}_{nested_key}"] = nested_value
        
        return key_results

    def _save_experiment_results(self, result: ExperimentResult):
        """Save experiment results to disk"""
        results_path = self.base_path / 'results' / f"{result.run_id}.pkl"
        
        with open(results_path, 'wb') as f:
            pickle.dump(result, f)

    def verify_reproducibility(
        self,
        experiment_id: str,
        run_function,
        tolerance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Verify that experiment results can be reproduced.
        
        Args:
            experiment_id: Experiment to reproduce
            run_function: Function that runs the experiment
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Reproducibility verification report
        """
        tolerance = tolerance or self.config['reproducibility_checks']['tolerance']
        
        self.logger.info(f"Verifying reproducibility for experiment {experiment_id}")
        
        # Load original experiment
        original_config = self.load_experiment(experiment_id)
        
        # Find original results
        original_results = self._load_experiment_results(experiment_id)
        if not original_results:
            raise ValueError(f"No results found for experiment {experiment_id}")
        
        # Re-run experiment
        verification_results = []
        n_retries = self.config['reproducibility_checks']['retry_failed_reproductions']
        
        for i in range(n_retries):
            self.logger.info(f"Reproducibility run {i+1}/{n_retries}")
            
            # Reset random seeds
            self._set_global_random_seeds(original_config.random_seeds)
            
            # Run experiment
            try:
                new_results = run_function(original_config)
                verification_results.append(new_results)
            except Exception as e:
                self.logger.error(f"Reproducibility run {i+1} failed: {str(e)}")
                continue
        
        # Compare results
        verification_report = self._compare_reproducibility_results(
            original_results[0],  # Use first original result
            verification_results,
            tolerance
        )
        
        # Save verification report
        self._save_reproducibility_report(experiment_id, verification_report)
        
        return verification_report

    def _load_experiment_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Load all results for an experiment"""
        results = []
        results_dir = self.base_path / 'results'
        
        for results_file in results_dir.glob(f"{experiment_id}_run_*.pkl"):
            with open(results_file, 'rb') as f:
                result = pickle.load(f)
                results.append(result)
        
        return results

    def _compare_reproducibility_results(
        self,
        original: ExperimentResult,
        reproduced: List[ExperimentResult],
        tolerance: float
    ) -> Dict[str, Any]:
        """Compare original and reproduced results"""
        
        if not reproduced:
            return {
                'reproducible': False,
                'reason': 'No successful reproduction runs',
                'successful_runs': 0
            }
        
        # Compare metrics
        metric_comparisons = {}
        for metric_name, original_value in original.metrics.items():
            reproduced_values = [r.metrics.get(metric_name) for r in reproduced if metric_name in r.metrics]
            
            if not reproduced_values:
                metric_comparisons[metric_name] = {
                    'original': original_value,
                    'reproduced': None,
                    'match': False,
                    'reason': 'Metric not found in reproduced results'
                }
                continue
            
            # Check if values are within tolerance
            matches = [abs(original_value - rv) <= tolerance for rv in reproduced_values]
            
            metric_comparisons[metric_name] = {
                'original': original_value,
                'reproduced': reproduced_values,
                'match': all(matches),
                'max_difference': max(abs(original_value - rv) for rv in reproduced_values),
                'tolerance': tolerance
            }
        
        # Overall reproducibility assessment
        all_metrics_match = all(comp['match'] for comp in metric_comparisons.values())
        successful_runs = len(reproduced)
        
        return {
            'reproducible': all_metrics_match,
            'successful_runs': successful_runs,
            'total_attempts': len(reproduced),
            'metric_comparisons': metric_comparisons,
            'hash_matches': [r.reproducibility_hash == original.reproducibility_hash for r in reproduced],
            'tolerance_used': tolerance,
            'verification_timestamp': datetime.now().isoformat()
        }

    def _save_reproducibility_report(self, experiment_id: str, report: Dict[str, Any]):
        """Save reproducibility verification report"""
        report_path = self.base_path / 'reproducibility_reports' / f"{experiment_id}_verification.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def conduct_power_analysis(
        self,
        effect_sizes: Optional[List[float]] = None,
        sample_sizes: Optional[List[int]] = None,
        significance_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Conduct statistical power analysis for experiment planning.
        
        Args:
            effect_sizes: Effect sizes to analyze
            sample_sizes: Sample sizes to test
            significance_level: Significance level
            
        Returns:
            Power analysis results
        """
        from statsmodels.stats.power import ttest_power, chisquare_power
        
        effect_sizes = effect_sizes or self.config['power_analysis']['effect_sizes']
        sample_sizes = sample_sizes or [50, 100, 200, 500, 1000, 2000]
        significance_level = significance_level or self.config['power_analysis']['significance_level']
        min_power = self.config['power_analysis']['min_power']
        
        power_analysis = {
            'parameters': {
                'effect_sizes': effect_sizes,
                'sample_sizes': sample_sizes,
                'significance_level': significance_level,
                'minimum_power': min_power
            },
            'ttest_power': {},
            'recommendations': {}
        }
        
        # T-test power analysis
        for effect_size in effect_sizes:
            power_analysis['ttest_power'][f'effect_{effect_size}'] = {}
            
            for sample_size in sample_sizes:
                power = ttest_power(effect_size, sample_size, significance_level)
                power_analysis['ttest_power'][f'effect_{effect_size}'][f'n_{sample_size}'] = power
        
        # Recommendations
        for effect_size in effect_sizes:
            # Find minimum sample size for adequate power
            for sample_size in sample_sizes:
                power = ttest_power(effect_size, sample_size, significance_level)
                if power >= min_power:
                    power_analysis['recommendations'][f'effect_{effect_size}'] = {
                        'min_sample_size': sample_size,
                        'achieved_power': power
                    }
                    break
            else:
                power_analysis['recommendations'][f'effect_{effect_size}'] = {
                    'min_sample_size': f'>{max(sample_sizes)}',
                    'achieved_power': ttest_power(effect_size, max(sample_sizes), significance_level)
                }
        
        return power_analysis

    def generate_reproducibility_report(self, experiment_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive reproducibility report"""
        
        if experiment_ids is None:
            # Get all experiments
            config_files = list((self.base_path / 'configs').glob('*.yaml'))
            experiment_ids = [f.stem for f in config_files]
        
        report = {
            'summary': {
                'total_experiments': len(experiment_ids),
                'verified_experiments': 0,
                'reproducible_experiments': 0,
                'report_timestamp': datetime.now().isoformat()
            },
            'experiments': {}
        }
        
        for exp_id in experiment_ids:
            exp_report = {
                'experiment_id': exp_id,
                'config_exists': (self.base_path / 'configs' / f"{exp_id}.yaml").exists(),
                'results_exist': len(list((self.base_path / 'results').glob(f"{exp_id}_run_*.pkl"))) > 0,
                'verification_exists': (self.base_path / 'reproducibility_reports' / f"{exp_id}_verification.json").exists()
            }
            
            # Load verification if exists
            if exp_report['verification_exists']:
                verification_path = self.base_path / 'reproducibility_reports' / f"{exp_id}_verification.json"
                with open(verification_path, 'r') as f:
                    verification = json.load(f)
                exp_report['reproducible'] = verification.get('reproducible', False)
                exp_report['verification_summary'] = {
                    'successful_runs': verification.get('successful_runs', 0),
                    'all_metrics_match': verification.get('reproducible', False)
                }
                
                if verification.get('reproducible', False):
                    report['summary']['reproducible_experiments'] += 1
                report['summary']['verified_experiments'] += 1
            
            report['experiments'][exp_id] = exp_report
        
        # Save report
        report_path = self.base_path / 'reproducibility_summary.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report


# High-level reproducibility functions
def setup_reproducible_experiment(
    name: str,
    description: str,
    hypothesis: str,
    parameters: Dict[str, Any],
    base_path: str = "experiments",
    researcher_info: Optional[Dict[str, str]] = None
) -> ReproducibilityManager:
    """
    Set up a reproducible experiment environment.
    
    This is the main entry point for creating reproducible experiments.
    """
    
    manager = ReproducibilityManager(base_path=base_path)
    experiment_id = manager.create_experiment(
        name=name,
        description=description,
        hypothesis=hypothesis,
        parameters=parameters,
        researcher_info=researcher_info
    )
    
    return manager


def verify_experiment_reproducibility(
    experiment_id: str,
    run_function,
    base_path: str = "experiments",
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Verify that an experiment can be reproduced.
    
    Args:
        experiment_id: ID of experiment to verify
        run_function: Function that runs the experiment
        base_path: Base path for experiment storage
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Reproducibility verification report
    """
    
    manager = ReproducibilityManager(base_path=base_path)
    return manager.verify_reproducibility(experiment_id, run_function, tolerance)