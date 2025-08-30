"""Model Evaluation Module

Contains comprehensive evaluation frameworks:
- Recommendation accuracy and ranking metrics
- Temporal consistency evaluation
- Cross-cultural performance metrics
- Statistical significance testing
- Results generation for research publication
"""

try:
    from .statistical_tests import StatisticalTestSuite, run_comprehensive_hypothesis_testing
    _has_statistical_tests = True
except ImportError:
    # Optional import - statistical tests not available
    StatisticalTestSuite = None
    run_comprehensive_hypothesis_testing = None
    _has_statistical_tests = False

try:
    from .reproducibility import ReproducibilityManager, setup_reproducible_experiment
    _has_reproducibility = True
except ImportError:
    # Optional import - reproducibility not available
    ReproducibilityManager = None
    setup_reproducible_experiment = None
    _has_reproducibility = False

__all__ = []
if _has_statistical_tests:
    __all__.extend(["StatisticalTestSuite", "run_comprehensive_hypothesis_testing"])
if _has_reproducibility:
    __all__.extend(["ReproducibilityManager", "setup_reproducible_experiment"])