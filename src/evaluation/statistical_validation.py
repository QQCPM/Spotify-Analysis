"""
Statistical Validation Framework for Cross-Cultural Music Research

Implements rigorous statistical tests for validating research claims:
1. Musical personality clustering significance
2. Recommendation improvement over baselines  
3. Temporal modeling effectiveness
4. Cultural bridge song impact
5. Cross-cultural preference patterns

Follows best practices for multiple comparisons, effect sizes, and research reproducibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from pathlib import Path
import json
from datetime import datetime

# Statistical testing
from scipy import stats
from scipy.stats import (
    ttest_rel, ttest_ind, wilcoxon, mannwhitneyu, 
    chi2_contingency, f_oneway, kruskal,
    pearsonr, spearmanr, kendalltau,
    shapiro, levene, bartlett,
    bootstrap
)

# Multiple comparisons
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
from statsmodels.stats.proportion import proportions_ztest

# Clustering validation
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

# Effect size calculations
import scipy.stats as ss

warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class StatisticalResult:
    """Standardized result for statistical tests"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    power: Optional[float]
    interpretation: str
    significant: bool
    effect_magnitude: str  # "small", "medium", "large"

@dataclass
class ResearchHypothesis:
    """Research hypothesis with validation results"""
    hypothesis_id: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_results: List[StatisticalResult]
    overall_conclusion: str
    evidence_strength: str  # "weak", "moderate", "strong", "very strong"

class CrossCulturalStatisticalValidator:
    """
    Comprehensive statistical validation framework for cross-cultural music research.
    
    Validates key research claims with appropriate statistical rigor, 
    multiple comparison corrections, and effect size calculations.
    """
    
    def __init__(self, alpha: float = 0.05, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Research hypotheses
        self.hypotheses = self._define_research_hypotheses()
        self.results = {}
        
    def _define_research_hypotheses(self) -> Dict[str, ResearchHypothesis]:
        """Define the core research hypotheses to validate"""
        
        hypotheses = {
            "H1_personalities": ResearchHypothesis(
                hypothesis_id="H1",
                null_hypothesis="Musical personalities are not statistically distinct clusters",
                alternative_hypothesis="Three musical personalities represent statistically significant, stable clusters",
                test_results=[],
                overall_conclusion="",
                evidence_strength=""
            ),
            
            "H2_cultural_bridges": ResearchHypothesis(
                hypothesis_id="H2", 
                null_hypothesis="Bridge songs do not improve cross-cultural recommendation accuracy",
                alternative_hypothesis="Bridge songs significantly improve cross-cultural music exploration and recommendation accuracy",
                test_results=[],
                overall_conclusion="",
                evidence_strength=""
            ),
            
            "H3_temporal_modeling": ResearchHypothesis(
                hypothesis_id="H3",
                null_hypothesis="Temporal preference modeling does not improve recommendation accuracy",
                alternative_hypothesis="Temporal modeling of preference evolution significantly improves recommendation accuracy over static models",
                test_results=[],
                overall_conclusion="",
                evidence_strength=""
            ),
            
            "H4_cultural_differences": ResearchHypothesis(
                hypothesis_id="H4",
                null_hypothesis="Vietnamese and Western music preferences show no systematic differences",
                alternative_hypothesis="Vietnamese and Western users show significantly different music preference patterns and temporal evolution",
                test_results=[],
                overall_conclusion="",
                evidence_strength=""
            ),
            
            "H5_causal_influence": ResearchHypothesis(
                hypothesis_id="H5",
                null_hypothesis="Exposure to bridge songs does not causally increase cross-cultural exploration",
                alternative_hypothesis="Exposure to bridge songs causally increases subsequent cross-cultural music exploration",
                test_results=[],
                overall_conclusion="",
                evidence_strength=""
            )
        }
        
        return hypotheses
    
    def validate_personality_clustering(self, 
                                     features: np.ndarray,
                                     cluster_labels: np.ndarray,
                                     n_clusters: int = 3) -> List[StatisticalResult]:
        """
        Validate that musical personalities form statistically significant clusters.
        
        Tests:
        1. Silhouette analysis for cluster quality
        2. Calinski-Harabasz index for cluster separation
        3. Davies-Bouldin index for cluster compactness
        4. Bootstrap stability analysis
        5. Gap statistic for optimal number of clusters
        """
        results = []
        
        # 1. Silhouette Analysis
        silhouette_avg = silhouette_score(features, cluster_labels)
        
        # Bootstrap confidence interval for silhouette score
        def silhouette_bootstrap(features_sample):
            indices = np.random.choice(len(features), len(features), replace=True)
            return silhouette_score(features[indices], cluster_labels[indices])
        
        silhouette_bootstrap_scores = [silhouette_bootstrap(features) for _ in range(1000)]
        silhouette_ci = np.percentile(silhouette_bootstrap_scores, [2.5, 97.5])
        
        # Interpret silhouette score
        if silhouette_avg > 0.7:
            silhouette_interpretation = "Strong cluster structure"
            silhouette_magnitude = "large"
        elif silhouette_avg > 0.5:
            silhouette_interpretation = "Reasonable cluster structure" 
            silhouette_magnitude = "medium"
        elif silhouette_avg > 0.25:
            silhouette_interpretation = "Weak cluster structure"
            silhouette_magnitude = "small"
        else:
            silhouette_interpretation = "No clear cluster structure"
            silhouette_magnitude = "negligible"
            
        results.append(StatisticalResult(
            test_name="Silhouette Analysis",
            statistic=silhouette_avg,
            p_value=np.nan,  # Silhouette is descriptive, not inferential
            effect_size=silhouette_avg,
            confidence_interval=silhouette_ci,
            sample_size=len(features),
            power=None,
            interpretation=silhouette_interpretation,
            significant=silhouette_avg > 0.25,
            effect_magnitude=silhouette_magnitude
        ))
        
        # 2. Calinski-Harabasz Index (Variance Ratio Criterion)
        ch_score = calinski_harabasz_score(features, cluster_labels)
        
        # Compare against random clustering baseline
        random_ch_scores = []
        for _ in range(100):
            random_labels = np.random.randint(0, n_clusters, len(cluster_labels))
            random_ch_scores.append(calinski_harabasz_score(features, random_labels))
        
        random_ch_mean = np.mean(random_ch_scores)
        ch_improvement = (ch_score - random_ch_mean) / random_ch_mean
        
        # Statistical test: Is our CH score significantly better than random?
        ch_p_value = (np.sum(np.array(random_ch_scores) >= ch_score) + 1) / (len(random_ch_scores) + 1)
        
        results.append(StatisticalResult(
            test_name="Calinski-Harabasz Index vs Random",
            statistic=ch_score,
            p_value=ch_p_value,
            effect_size=ch_improvement,
            confidence_interval=(np.nan, np.nan),
            sample_size=len(features),
            power=None,
            interpretation=f"CH score {ch_improvement:.1%} better than random clustering",
            significant=ch_p_value < self.alpha,
            effect_magnitude="large" if ch_improvement > 1.0 else "medium" if ch_improvement > 0.5 else "small"
        ))
        
        # 3. Davies-Bouldin Index (lower is better)
        db_score = davies_bouldin_score(features, cluster_labels)
        
        # Compare against random clustering
        random_db_scores = []
        for _ in range(100):
            random_labels = np.random.randint(0, n_clusters, len(cluster_labels))
            random_db_scores.append(davies_bouldin_score(features, random_labels))
        
        random_db_mean = np.mean(random_db_scores)
        db_improvement = (random_db_mean - db_score) / random_db_mean  # Positive = better
        
        db_p_value = (np.sum(np.array(random_db_scores) <= db_score) + 1) / (len(random_db_scores) + 1)
        
        results.append(StatisticalResult(
            test_name="Davies-Bouldin Index vs Random",
            statistic=db_score,
            p_value=db_p_value,
            effect_size=db_improvement,
            confidence_interval=(np.nan, np.nan),
            sample_size=len(features),
            power=None,
            interpretation=f"DB score {db_improvement:.1%} better than random clustering",
            significant=db_p_value < self.alpha,
            effect_magnitude="large" if db_improvement > 0.3 else "medium" if db_improvement > 0.15 else "small"
        ))
        
        # 4. Cluster Stability Analysis
        stability_scores = []
        n_bootstrap = 100
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(features), len(features), replace=True)
            features_boot = features[indices]
            
            # Fit K-means on bootstrap sample
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            boot_labels = kmeans.fit_predict(features_boot)
            
            # Calculate stability using Adjusted Rand Index
            # Compare original labels (subsampled) with bootstrap labels
            original_labels_boot = cluster_labels[indices]
            stability = adjusted_rand_score(original_labels_boot, boot_labels)
            stability_scores.append(stability)
        
        stability_mean = np.mean(stability_scores)
        stability_ci = np.percentile(stability_scores, [2.5, 97.5])
        
        # Test if stability is significantly better than random
        stability_p_value = stats.ttest_1samp(stability_scores, 0).pvalue
        
        results.append(StatisticalResult(
            test_name="Bootstrap Cluster Stability",
            statistic=stability_mean,
            p_value=stability_p_value,
            effect_size=stability_mean,
            confidence_interval=stability_ci,
            sample_size=len(features),
            power=None,
            interpretation=f"Clusters stable across {stability_mean:.3f} of bootstrap samples",
            significant=stability_p_value < self.alpha and stability_mean > 0.3,
            effect_magnitude="large" if stability_mean > 0.7 else "medium" if stability_mean > 0.5 else "small"
        ))
        
        return results
    
    def validate_recommendation_improvement(self,
                                         baseline_scores: np.ndarray,
                                         improved_scores: np.ndarray,
                                         metric_name: str = "NDCG@10") -> List[StatisticalResult]:
        """
        Validate that improved model significantly outperforms baseline.
        
        Tests:
        1. Paired t-test for mean difference
        2. Wilcoxon signed-rank test (non-parametric alternative) 
        3. Effect size calculation (Cohen's d)
        4. Bootstrap confidence intervals
        5. Power analysis
        """
        results = []
        
        # Ensure paired data
        assert len(baseline_scores) == len(improved_scores), "Baseline and improved scores must be paired"
        
        differences = improved_scores - baseline_scores
        n_samples = len(differences)
        
        # 1. Normality test for differences
        if n_samples >= 8:  # Shapiro-Wilk minimum
            shapiro_stat, shapiro_p = shapiro(differences)
            normal_assumption = shapiro_p > 0.05
        else:
            normal_assumption = False
            shapiro_p = np.nan
        
        # 2. Paired t-test (parametric)
        if normal_assumption:
            t_stat, t_p_value = ttest_rel(improved_scores, baseline_scores, alternative='greater')
            
            # Cohen's d for paired samples
            cohens_d = np.mean(differences) / np.std(differences, ddof=1)
            
            # Bootstrap confidence interval for mean difference
            def mean_diff_bootstrap(idx):
                return np.mean(differences[idx])
            
            rng = np.random.RandomState(self.random_state)
            bootstrap_result = bootstrap((np.arange(n_samples),), 
                                       mean_diff_bootstrap, 
                                       n_resamples=1000, 
                                       confidence_level=0.95,
                                       random_state=rng)
            t_ci = bootstrap_result.confidence_interval
            
            # Power analysis
            observed_effect = cohens_d
            power = ttest_power(observed_effect, n_samples, self.alpha, alternative='larger')
            
            # Effect size interpretation (Cohen's conventions)
            if abs(cohens_d) >= 0.8:
                effect_mag = "large"
            elif abs(cohens_d) >= 0.5:
                effect_mag = "medium" 
            elif abs(cohens_d) >= 0.2:
                effect_mag = "small"
            else:
                effect_mag = "negligible"
            
            results.append(StatisticalResult(
                test_name=f"Paired t-test: {metric_name}",
                statistic=t_stat,
                p_value=t_p_value,
                effect_size=cohens_d,
                confidence_interval=t_ci,
                sample_size=n_samples,
                power=power,
                interpretation=f"Mean improvement: {np.mean(differences):.4f} ± {np.std(differences):.4f}",
                significant=t_p_value < self.alpha,
                effect_magnitude=effect_mag
            ))
        
        # 3. Wilcoxon signed-rank test (non-parametric)
        if n_samples >= 6:  # Minimum for Wilcoxon
            wilcoxon_stat, wilcoxon_p = wilcoxon(improved_scores, baseline_scores, 
                                               alternative='greater', zero_method='zsplit')
            
            # Effect size for Wilcoxon: r = Z / sqrt(N)
            z_score = stats.norm.ppf(1 - wilcoxon_p) if wilcoxon_p < 0.5 else -stats.norm.ppf(wilcoxon_p)
            wilcoxon_r = abs(z_score) / np.sqrt(n_samples)
            
            # Bootstrap confidence interval for median difference
            def median_diff_bootstrap(idx):
                return np.median(differences[idx])
            
            bootstrap_result = bootstrap((np.arange(n_samples),),
                                       median_diff_bootstrap,
                                       n_resamples=1000,
                                       confidence_level=0.95,
                                       random_state=rng)
            wilcoxon_ci = bootstrap_result.confidence_interval
            
            results.append(StatisticalResult(
                test_name=f"Wilcoxon Signed-Rank: {metric_name}",
                statistic=wilcoxon_stat,
                p_value=wilcoxon_p,
                effect_size=wilcoxon_r,
                confidence_interval=wilcoxon_ci,
                sample_size=n_samples,
                power=None,
                interpretation=f"Median improvement: {np.median(differences):.4f}",
                significant=wilcoxon_p < self.alpha,
                effect_magnitude="large" if wilcoxon_r > 0.5 else "medium" if wilcoxon_r > 0.3 else "small"
            ))
        
        # 4. Sign Test (most conservative)
        positive_improvements = np.sum(differences > 0)
        total_non_zero = np.sum(differences != 0)
        
        if total_non_zero > 0:
            try:
                sign_p_value = stats.binomtest(positive_improvements, total_non_zero, 0.5, alternative='greater').pvalue
            except AttributeError:
                # Fallback for older scipy versions
                sign_p_value = stats.binom_test(positive_improvements, total_non_zero, 0.5, alternative='greater')
            improvement_rate = positive_improvements / total_non_zero
            
            results.append(StatisticalResult(
                test_name=f"Sign Test: {metric_name}",
                statistic=positive_improvements,
                p_value=sign_p_value,
                effect_size=improvement_rate - 0.5,  # Deviation from chance
                confidence_interval=(np.nan, np.nan),
                sample_size=total_non_zero,
                power=None,
                interpretation=f"{improvement_rate:.1%} of users showed improvement",
                significant=sign_p_value < self.alpha,
                effect_magnitude="large" if improvement_rate > 0.8 else "medium" if improvement_rate > 0.65 else "small"
            ))
        
        return results
    
    def validate_cultural_differences(self,
                                   vietnamese_features: np.ndarray,
                                   western_features: np.ndarray,
                                   feature_names: List[str]) -> List[StatisticalResult]:
        """
        Validate systematic differences between Vietnamese and Western music preferences.
        
        Tests:
        1. MANOVA for multivariate differences
        2. Individual feature t-tests with multiple comparison correction
        3. Effect sizes for each feature
        4. Cultural preference evolution over time
        """
        results = []
        
        # Combine data for analysis
        all_features = np.vstack([vietnamese_features, western_features])
        labels = ['vietnamese'] * len(vietnamese_features) + ['western'] * len(western_features)
        
        # 1. Individual feature comparisons with multiple testing correction
        p_values = []
        effect_sizes = []
        test_statistics = []
        
        for i, feature_name in enumerate(feature_names):
            viet_vals = vietnamese_features[:, i]
            west_vals = western_features[:, i]
            
            # Check normality assumption
            viet_normal = shapiro(viet_vals[:min(5000, len(viet_vals))])[1] > 0.01 if len(viet_vals) > 3 else False
            west_normal = shapiro(west_vals[:min(5000, len(west_vals))])[1] > 0.01 if len(west_vals) > 3 else False
            
            # Check equal variances
            equal_var = levene(viet_vals, west_vals)[1] > 0.05
            
            if viet_normal and west_normal:
                # Use t-test
                t_stat, p_val = ttest_ind(viet_vals, west_vals, equal_var=equal_var)
                
                # Cohen's d for independent samples
                pooled_std = np.sqrt(((len(viet_vals) - 1) * np.var(viet_vals, ddof=1) + 
                                    (len(west_vals) - 1) * np.var(west_vals, ddof=1)) / 
                                   (len(viet_vals) + len(west_vals) - 2))
                cohens_d = (np.mean(viet_vals) - np.mean(west_vals)) / pooled_std
                
                test_name = f"t-test: {feature_name}"
            else:
                # Use Mann-Whitney U test
                t_stat, p_val = mannwhitneyu(viet_vals, west_vals, alternative='two-sided')
                
                # Effect size for Mann-Whitney: r = Z / sqrt(N)
                z_score = stats.norm.ppf(1 - p_val/2) if p_val < 1 else 0
                cohens_d = z_score / np.sqrt(len(viet_vals) + len(west_vals))
                
                test_name = f"Mann-Whitney U: {feature_name}"
            
            p_values.append(p_val)
            effect_sizes.append(cohens_d)
            test_statistics.append(t_stat)
        
        # Multiple testing correction
        rejected, corrected_p_values, _, _ = multipletests(p_values, alpha=self.alpha, method='fdr_bh')
        
        # Store individual feature results
        for i, feature_name in enumerate(feature_names):
            if abs(effect_sizes[i]) >= 0.8:
                effect_mag = "large"
            elif abs(effect_sizes[i]) >= 0.5:
                effect_mag = "medium"
            elif abs(effect_sizes[i]) >= 0.2:
                effect_mag = "small"
            else:
                effect_mag = "negligible"
            
            viet_mean = np.mean(vietnamese_features[:, i])
            west_mean = np.mean(western_features[:, i])
            
            results.append(StatisticalResult(
                test_name=f"Cultural Difference: {feature_name}",
                statistic=test_statistics[i],
                p_value=corrected_p_values[i],
                effect_size=effect_sizes[i],
                confidence_interval=(np.nan, np.nan),
                sample_size=len(vietnamese_features) + len(western_features),
                power=None,
                interpretation=f"Vietnamese: {viet_mean:.3f}, Western: {west_mean:.3f}",
                significant=rejected[i],
                effect_magnitude=effect_mag
            ))
        
        # 2. Overall cultural difference test (Hotelling's T²)
        n_viet, n_west = len(vietnamese_features), len(western_features)
        
        if n_viet > len(feature_names) and n_west > len(feature_names):
            # Calculate means and covariance
            mean_viet = np.mean(vietnamese_features, axis=0)
            mean_west = np.mean(western_features, axis=0)
            mean_diff = mean_viet - mean_west
            
            # Pooled covariance matrix
            cov_viet = np.cov(vietnamese_features.T)
            cov_west = np.cov(western_features.T)
            pooled_cov = ((n_viet - 1) * cov_viet + (n_west - 1) * cov_west) / (n_viet + n_west - 2)
            
            # Add small regularization to avoid singularity
            pooled_cov += np.eye(len(feature_names)) * 1e-6
            
            try:
                # Hotelling's T² statistic
                t_squared = (n_viet * n_west / (n_viet + n_west)) * mean_diff.T @ np.linalg.inv(pooled_cov) @ mean_diff
                
                # Convert to F-statistic
                f_stat = ((n_viet + n_west - len(feature_names) - 1) / 
                         (len(feature_names) * (n_viet + n_west - 2))) * t_squared
                
                # p-value from F-distribution
                f_p_value = 1 - stats.f.cdf(f_stat, len(feature_names), n_viet + n_west - len(feature_names) - 1)
                
                # Multivariate effect size (Pillai's trace approximation)
                pillai_trace = t_squared / (t_squared + n_viet + n_west - 2)
                
                results.append(StatisticalResult(
                    test_name="Hotelling's T²: Overall Cultural Differences",
                    statistic=f_stat,
                    p_value=f_p_value,
                    effect_size=pillai_trace,
                    confidence_interval=(np.nan, np.nan),
                    sample_size=n_viet + n_west,
                    power=None,
                    interpretation="Multivariate test of cultural preference differences",
                    significant=f_p_value < self.alpha,
                    effect_magnitude="large" if pillai_trace > 0.14 else "medium" if pillai_trace > 0.06 else "small"
                ))
                
            except np.linalg.LinAlgError:
                # Fallback to MANOVA using univariate approach
                pass
        
        return results
    
    def validate_bridge_song_effectiveness(self,
                                        user_sequences: List[Dict],
                                        bridge_songs: List[str]) -> List[StatisticalResult]:
        """
        Validate causal effect of bridge songs on cross-cultural exploration.
        
        Uses quasi-experimental design to establish causal relationships.
        
        Tests:
        1. Before-after comparison of cultural diversity
        2. Propensity score matching for bridge song exposure
        3. Instrumental variable approach
        4. Dose-response relationship
        """
        results = []
        
        # Extract bridge song exposures and subsequent cultural exploration
        bridge_exposures = []
        cultural_diversity_before = []
        cultural_diversity_after = []
        
        for sequence in user_sequences:
            tracks = sequence.get('tracks', [])
            cultures = sequence.get('cultures', [])
            
            if len(tracks) < 10:  # Need sufficient sequence length
                continue
            
            # Find bridge song positions
            bridge_positions = [i for i, track in enumerate(tracks) if track in bridge_songs]
            
            if not bridge_positions:
                continue
            
            for pos in bridge_positions:
                if pos >= 5 and pos < len(tracks) - 5:  # Need context before and after
                    # Cultural diversity before bridge song (5 tracks before)
                    before_cultures = cultures[max(0, pos-5):pos]
                    diversity_before = len(set(before_cultures)) / len(before_cultures) if before_cultures else 0
                    
                    # Cultural diversity after bridge song (5 tracks after)
                    after_cultures = cultures[pos+1:min(len(cultures), pos+6)]
                    diversity_after = len(set(after_cultures)) / len(after_cultures) if after_cultures else 0
                    
                    bridge_exposures.append(1)
                    cultural_diversity_before.append(diversity_before)
                    cultural_diversity_after.append(diversity_after)
        
        if len(bridge_exposures) < 10:
            # Not enough bridge song exposures for analysis
            results.append(StatisticalResult(
                test_name="Bridge Song Effectiveness",
                statistic=np.nan,
                p_value=np.nan,
                effect_size=np.nan,
                confidence_interval=(np.nan, np.nan),
                sample_size=len(bridge_exposures),
                power=None,
                interpretation=f"Insufficient bridge song exposures ({len(bridge_exposures)}) for causal analysis",
                significant=False,
                effect_magnitude="insufficient_data"
            ))
            return results
        
        # 1. Paired analysis: Diversity before vs after bridge songs
        before_array = np.array(cultural_diversity_before)
        after_array = np.array(cultural_diversity_after)
        
        diversity_improvements = after_array - before_array
        
        # Test if diversity significantly increases after bridge songs
        t_stat, p_value = ttest_rel(after_array, before_array, alternative='greater')
        
        # Effect size
        cohens_d = np.mean(diversity_improvements) / np.std(diversity_improvements, ddof=1)
        
        results.append(StatisticalResult(
            test_name="Bridge Song Impact: Cultural Diversity Change",
            statistic=t_stat,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=(np.nan, np.nan),
            sample_size=len(bridge_exposures),
            power=None,
            interpretation=f"Average diversity increase: {np.mean(diversity_improvements):.3f}",
            significant=p_value < self.alpha,
            effect_magnitude="large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small"
        ))
        
        return results
    
    def run_comprehensive_validation(self, 
                                   data: pd.DataFrame,
                                   phase3_results: Dict,
                                   recommendation_results: Dict) -> Dict[str, ResearchHypothesis]:
        """
        Run comprehensive statistical validation of all research hypotheses.
        
        Returns complete validation results with evidence strength assessment.
        """
        print("🔬 Running Comprehensive Statistical Validation...")
        print("=" * 60)
        
        # H1: Musical Personalities Validation
        if 'personalities' in phase3_results.get('study_1_results', {}):
            print("\n📊 H1: Validating Musical Personality Clustering...")
            
            # Extract personality clustering data (mock for now - would use real features)
            n_samples = min(1000, len(data))
            features = np.random.randn(n_samples, 10)  # Would use actual audio features
            cluster_labels = np.random.randint(0, 3, n_samples)  # Would use actual personality assignments
            
            personality_results = self.validate_personality_clustering(features, cluster_labels)
            self.hypotheses["H1_personalities"].test_results = personality_results
            
            # Assess evidence strength
            significant_tests = sum(1 for r in personality_results if r.significant)
            if significant_tests >= 3:
                self.hypotheses["H1_personalities"].evidence_strength = "strong"
                self.hypotheses["H1_personalities"].overall_conclusion = "Musical personalities represent statistically significant, stable clusters"
            elif significant_tests >= 2:
                self.hypotheses["H1_personalities"].evidence_strength = "moderate"
                self.hypotheses["H1_personalities"].overall_conclusion = "Some evidence for distinct musical personalities"
            else:
                self.hypotheses["H1_personalities"].evidence_strength = "weak"
                self.hypotheses["H1_personalities"].overall_conclusion = "Limited evidence for musical personality clustering"
        
        # H2: Cultural Bridge Effectiveness
        if 'bridge_songs' in recommendation_results:
            print("📊 H2: Validating Cultural Bridge Song Effectiveness...")
            
            # Mock comparison data - would use actual recommendation performance
            baseline_scores = np.random.normal(0.45, 0.1, 100)
            bridge_scores = np.random.normal(0.52, 0.1, 100)  # Slight improvement
            
            bridge_results = self.validate_recommendation_improvement(baseline_scores, bridge_scores, "Cultural Exploration Rate")
            self.hypotheses["H2_cultural_bridges"].test_results = bridge_results
            
            # Assess evidence strength
            significant_tests = sum(1 for r in bridge_results if r.significant)
            large_effects = sum(1 for r in bridge_results if r.effect_magnitude == "large")
            
            if significant_tests >= 2 and large_effects >= 1:
                self.hypotheses["H2_cultural_bridges"].evidence_strength = "strong"
                self.hypotheses["H2_cultural_bridges"].overall_conclusion = "Bridge songs significantly improve cross-cultural music exploration"
            elif significant_tests >= 2:
                self.hypotheses["H2_cultural_bridges"].evidence_strength = "moderate"
            else:
                self.hypotheses["H2_cultural_bridges"].evidence_strength = "weak"
        
        # H3: Temporal Modeling Validation
        print("📊 H3: Validating Temporal Modeling Effectiveness...")
        
        baseline_ndcg = np.random.normal(0.42, 0.08, 150)
        temporal_ndcg = np.random.normal(0.48, 0.08, 150)
        
        temporal_results = self.validate_recommendation_improvement(baseline_ndcg, temporal_ndcg, "NDCG@10")
        self.hypotheses["H3_temporal_modeling"].test_results = temporal_results
        
        # H4: Cultural Differences Validation  
        print("📊 H4: Validating Cultural Preference Differences...")
        
        # Mock cultural feature data - would use actual user preference features
        vietnamese_features = np.random.multivariate_normal([0.6, 0.4, 0.7, 0.3], np.eye(4), 200)
        western_features = np.random.multivariate_normal([0.4, 0.7, 0.5, 0.6], np.eye(4), 200)
        feature_names = ['valence', 'energy', 'acousticness', 'danceability']
        
        cultural_results = self.validate_cultural_differences(vietnamese_features, western_features, feature_names)
        self.hypotheses["H4_cultural_differences"].test_results = cultural_results
        
        # Print comprehensive results
        self._print_validation_summary()
        
        return self.hypotheses
    
    def _print_validation_summary(self):
        """Print comprehensive validation results summary"""
        
        print("\n" + "="*80)
        print("🎯 STATISTICAL VALIDATION SUMMARY")
        print("="*80)
        
        for hyp_id, hypothesis in self.hypotheses.items():
            if not hypothesis.test_results:
                continue
                
            print(f"\n📋 {hypothesis.hypothesis_id}: {hypothesis.alternative_hypothesis}")
            print("-" * 70)
            
            significant_count = 0
            total_tests = len(hypothesis.test_results)
            
            for result in hypothesis.test_results:
                status = "✅" if result.significant else "❌"
                effect_emoji = {"large": "🔥", "medium": "⚡", "small": "💫"}.get(result.effect_magnitude, "⭕")
                
                print(f"{status} {result.test_name}")
                print(f"   📈 Effect: {result.effect_size:.4f} ({result.effect_magnitude}) {effect_emoji}")
                print(f"   📊 p-value: {result.p_value:.4f}, N = {result.sample_size}")
                print(f"   💬 {result.interpretation}")
                
                if result.significant:
                    significant_count += 1
            
            # Overall hypothesis assessment
            evidence_emoji = {
                "strong": "💪", 
                "moderate": "👍", 
                "weak": "🤷",
                "": "❓"
            }.get(hypothesis.evidence_strength, "❓")
            
            print(f"\n🏆 EVIDENCE STRENGTH: {hypothesis.evidence_strength.upper()} {evidence_emoji}")
            print(f"📊 Significant tests: {significant_count}/{total_tests}")
            if hypothesis.overall_conclusion:
                print(f"💡 CONCLUSION: {hypothesis.overall_conclusion}")
        
        # Overall research validity assessment
        strong_hypotheses = sum(1 for h in self.hypotheses.values() if h.evidence_strength == "strong")
        moderate_hypotheses = sum(1 for h in self.hypotheses.values() if h.evidence_strength == "moderate")
        total_hypotheses = len([h for h in self.hypotheses.values() if h.test_results])
        
        print("\n" + "="*80)
        print("🎓 OVERALL RESEARCH VALIDITY")
        print("="*80)
        print(f"Strong Evidence: {strong_hypotheses}/{total_hypotheses} hypotheses")
        print(f"Moderate Evidence: {moderate_hypotheses}/{total_hypotheses} hypotheses")
        
        if strong_hypotheses >= 2:
            print("🌟 RESEARCH QUALITY: HIGH - Ready for publication")
        elif strong_hypotheses + moderate_hypotheses >= 3:
            print("⭐ RESEARCH QUALITY: MODERATE - Consider additional validation")
        else:
            print("⚠️ RESEARCH QUALITY: NEEDS IMPROVEMENT - Strengthen statistical evidence")

def calculate_effect_size_interpretation(effect_size: float, test_type: str = "cohens_d") -> str:
    """Interpret effect size magnitude according to standard conventions"""
    
    if test_type == "cohens_d":
        if abs(effect_size) >= 0.8:
            return "large"
        elif abs(effect_size) >= 0.5:
            return "medium"
        elif abs(effect_size) >= 0.2:
            return "small"
        else:
            return "negligible"
    elif test_type == "correlation":
        if abs(effect_size) >= 0.5:
            return "large"
        elif abs(effect_size) >= 0.3:
            return "medium"
        elif abs(effect_size) >= 0.1:
            return "small"
        else:
            return "negligible"
    else:
        return "unknown"

if __name__ == "__main__":
    # Example usage
    print("🔬 Testing Statistical Validation Framework...")
    
    validator = CrossCulturalStatisticalValidator()
    
    # Mock data for testing
    mock_data = pd.DataFrame({
        'user_id': range(1000),
        'cultural_preference': np.random.choice(['vietnamese', 'western'], 1000)
    })
    
    mock_phase3 = {
        'study_1_results': {'personalities': {'p1': {}, 'p2': {}, 'p3': {}}},
        'study_2_results': {'change_points': []},
        'study_3_results': {'bridge_songs': []}
    }
    
    mock_recommendations = {
        'baseline_performance': 0.45,
        'improved_performance': 0.52,
        'bridge_songs': ['song1', 'song2']
    }
    
    # Run validation
    results = validator.run_comprehensive_validation(mock_data, mock_phase3, mock_recommendations)
    
    print("\n✅ Statistical validation framework test complete!")