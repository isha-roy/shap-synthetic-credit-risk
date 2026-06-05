import numpy as np
from scipy.stats import wilcoxon

def compute_bootstrap_ci(data, confidence_level=0.95, n_resamples=1000, seed=42):
    """
    Computes 95% bootstrap confidence interval of the mean of data using the percentile method.
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)
    if n == 0:
        return 0.0, 0.0
        
    bootstrap_means = []
    for _ in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
        
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_means, (alpha / 2.0) * 100.0))
    upper = float(np.percentile(bootstrap_means, (1.0 - alpha / 2.0) * 100.0))
    
    return lower, upper

def compute_paired_cohens_d(group1, group2):
    """
    Computes Cohen's d effect size for paired samples.
    d = mean(differences) / std(differences)
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if len(group1) != len(group2) or len(group1) == 0:
        return 0.0
        
    diff = group1 - group2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    if std_diff > 0:
        d = mean_diff / std_diff
    else:
        d = 0.0
        
    return float(d)

def run_paired_wilcoxon(group1, group2):
    """
    Runs Wilcoxon signed-rank test for paired samples.
    Safely handles edge cases like zero differences.
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if len(group1) != len(group2) or len(group1) == 0:
        return 0.0, 1.0
        
    diff = group1 - group2
    # If all differences are zero, Wilcoxon test fails. Return p-value = 1.0
    if np.all(diff == 0):
        return 0.0, 1.0
        
    try:
        stat, p_val = wilcoxon(group1, group2, alternative='two-sided')
        return float(stat), float(p_val)
    except Exception as e:
        print(f"Wilcoxon signed-rank test warning: {e}. Returning p-value=1.0.")
        return 0.0, 1.0
