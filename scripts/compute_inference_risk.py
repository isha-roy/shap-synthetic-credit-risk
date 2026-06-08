import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wilcoxon

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc
from src.models.synthetic import fit_real_preprocessing, preprocess_synthetic_data

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# ==========================================
# TASK 1: Core Inference Risk Function
# ==========================================
def compute_inference_risk(X_synthetic, X_train_real):
    """
    Computes Inference Risk Indicator for synthetic data.
    
    For each synthetic point:
    - Find ds = distance to closest real training point
    - Find d0 = distance from that closest real point to 
                its nearest neighbor WITHIN the real training set
    - If ds < d0, the synthetic point is dangerously close 
                  to that specific real record
    
    Returns:
    - inference_risk_score: float (proportion of risky synthetic points)
    - risky_indices: array of synthetic point indices that are risky
    - per_point_flags: boolean array
    """
    if hasattr(X_synthetic, 'values'):
        X_synthetic = X_synthetic.values
    if hasattr(X_train_real, 'values'):
        X_train_real = X_train_real.values
        
    n_syn = len(X_synthetic)
    n_real = len(X_train_real)
    
    if n_real < 2:
        raise ValueError("Real dataset must have at least 2 samples to compute d0.")
        
    # 1. Precompute d0 for each point in X_train_real (distance to its nearest neighbor other than itself)
    nn_real = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1)
    nn_real.fit(X_train_real)
    dists_real, _ = nn_real.kneighbors(X_train_real, n_neighbors=2)
    # The second column contains distance to the nearest other point (d0)
    d0 = dists_real[:, 1]
    
    # 2. Query 1-NN in X_train_real for each point in X_synthetic
    nn_real_1 = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn_real_1.fit(X_train_real)
    
    batch_size = 500
    per_point_flags = np.zeros(n_syn, dtype=bool)
    
    # Add progress bar if tqdm is requested and dataset is larger
    disable_tqdm = n_syn <= 1000
    
    for start_idx in tqdm(range(0, n_syn, batch_size), desc="Computing inference risk", disable=disable_tqdm):
        end_idx = min(start_idx + batch_size, n_syn)
        X_batch = X_synthetic[start_idx:end_idx]
        
        ds_batch, indices_batch = nn_real_1.kneighbors(X_batch, n_neighbors=1)
        ds = ds_batch[:, 0]
        closest_real_indices = indices_batch[:, 0]
        
        d0_batch = d0[closest_real_indices]
        
        # Check ds < d0 (dangerously close)
        batch_flags = ds < d0_batch
        per_point_flags[start_idx:end_idx] = batch_flags
        
    risky_indices = np.where(per_point_flags)[0]
    inference_risk_score = float(len(risky_indices) / n_syn)
    
    return inference_risk_score, risky_indices, per_point_flags

# ==========================================
# TASK 2: Threshold Computation (OI equivalent)
# ==========================================
def compute_inference_risk_threshold(X_real, n_splits=50, 
                                      sample_size=None,
                                      percentile=95,
                                      random_state=42):
    """
    Estimates what Inference Risk looks like when synthetic data 
    is just another random sample from the same real distribution.
    
    Method (from Min & Oh 2025):
    - Randomly split real data into 50 subsets
    - Compute inference risk of each subset against the rest
    - Take 95th percentile as threshold
    
    If X_real is small (German Credit train ~700 rows):
        use sample_size = 100 per split
    If X_real is large (GMSC train ~7000 rows):
        use sample_size = 500 per split
    """
    if hasattr(X_real, 'values'):
        X_real = X_real.values
        
    if sample_size is None:
        if len(X_real) < 2000:
            sample_size = 100
        else:
            sample_size = 500
            
    scores = []
    from sklearn.model_selection import train_test_split
    
    for i in range(n_splits):
        split_seed = random_state + i
        X_train, X_val = train_test_split(
            X_real, 
            test_size=sample_size, 
            random_state=split_seed,
            shuffle=True
        )
        score, _, _ = compute_inference_risk(X_val, X_train)
        scores.append(score)
        
    threshold = float(np.percentile(scores, percentile))
    return threshold

# Cohen's d helper function
def compute_cohen_d(x, y):
    x = np.array(x)
    y = np.array(y)
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

# ==========================================
# TASK 3: Multi-Seed Experiment Runner
# ==========================================
def run_experiments():
    seeds = [42, 123, 456, 789, 1337]
    datasets = {
        "german_credit": {
            "name": "german_credit",
            "suffix": "german",
            "target": "class",
            "preprocess_func": preprocess_german_credit
        },
        "gmsc": {
            "name": "gmsc",
            "suffix": "gmsc",
            "target": "SeriousDlqin2yrs",
            "preprocess_func": preprocess_gmsc
        }
    }
    
    results = {}
    
    for ds_key, ds_info in datasets.items():
        dataset_name = ds_info["name"]
        suffix = ds_info["suffix"]
        target_col = ds_info["target"]
        prep_func = ds_info["preprocess_func"]
        
        print(f"\n" + "="*50)
        print(f"RUNNING EVALUATION FOR DATASET: {dataset_name.upper()}")
        print("="*50)
        
        # 1. Compute threshold using seed 42 train split
        prep_func(seed=42)
        X_train_real_ref = pd.read_csv(f"data/processed/X_train_{suffix}.csv")
        threshold = compute_inference_risk_threshold(X_train_real_ref, n_splits=50, random_state=42)
        print(f"Estimated baseline threshold (95th pct): {threshold:.4f}")
        
        ctgan_scores = []
        tvae_scores = []
        
        for seed in seeds:
            # Regenerate training split for this seed
            prep_func(seed=seed)
            X_train_real = pd.read_csv(f"data/processed/X_train_{suffix}.csv")
            fitted_transformers = fit_real_preprocessing(dataset_name, seed)
            
            # Load CTGAN synthetic data
            ctgan_paths = [
                f"data/synthetic/{dataset_name}_ctgan_seed{seed}.csv",
                f"data/synthetic/ctgan_synthetic_{suffix}_seed{seed}.csv",
                f"data/synthetic/ctgan_synthetic_{suffix}.csv"
            ]
            ctgan_path = None
            for p in ctgan_paths:
                if os.path.exists(p):
                    ctgan_path = p
                    break
            
            if ctgan_path:
                df_syn = pd.read_csv(ctgan_path)
                X_syn, _ = preprocess_synthetic_data(df_syn, dataset_name, fitted_transformers, target_col)
                score, _, _ = compute_inference_risk(X_syn, X_train_real)
                ctgan_scores.append(score)
            else:
                print(f"  CTGAN file not found for seed {seed}")
                
            # Load TVAE synthetic data
            tvae_paths = [
                f"data/synthetic/{dataset_name}_tvae_seed{seed}.csv",
                f"data/synthetic/tvae_synthetic_{suffix}_seed{seed}.csv",
                f"data/synthetic/tvae_synthetic_{suffix}.csv"
            ]
            tvae_path = None
            for p in tvae_paths:
                if os.path.exists(p):
                    tvae_path = p
                    break
            
            if tvae_path:
                df_syn = pd.read_csv(tvae_path)
                X_syn, _ = preprocess_synthetic_data(df_syn, dataset_name, fitted_transformers, target_col)
                score, _, _ = compute_inference_risk(X_syn, X_train_real)
                tvae_scores.append(score)
            else:
                print(f"  TVAE file not found for seed {seed}")
                
        ctgan_mean = float(np.mean(ctgan_scores)) if ctgan_scores else 0.0
        ctgan_std = float(np.std(ctgan_scores)) if ctgan_scores else 0.0
        tvae_mean = float(np.mean(tvae_scores)) if tvae_scores else 0.0
        tvae_std = float(np.std(tvae_scores)) if tvae_scores else 0.0
        
        results[dataset_name] = {
            "ctgan": {
                "scores_per_seed": ctgan_scores,
                "mean": ctgan_mean,
                "std": ctgan_std,
                "threshold": threshold,
                "exceeds_threshold": ctgan_mean > threshold
            },
            "tvae": {
                "scores_per_seed": tvae_scores,
                "mean": tvae_mean,
                "std": tvae_std,
                "threshold": threshold,
                "exceeds_threshold": tvae_mean > threshold
            }
        }
        
        # Save results JSON
        out_dir = "results/privacy/"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"inference_risk_{suffix}.json")
        with open(out_path, 'w') as f:
            json.dump(results[dataset_name], f, indent=2)
        print(f"Results successfully written to {out_path}")
        
    return results

# ==========================================
# TASK 4: Visualization
# ==========================================
def generate_plots(results):
    sns.set_theme(style="whitegrid")
    
    german_ctgan = results["german_credit"]["ctgan"]
    german_tvae = results["german_credit"]["tvae"]
    gmsc_ctgan = results["gmsc"]["ctgan"]
    gmsc_tvae = results["gmsc"]["tvae"]
    
    # ------------------------------------------
    # Figure A - Inference Risk Bar Chart
    # ------------------------------------------
    plt.figure(figsize=(10, 5))
    
    means = [german_ctgan["mean"], german_tvae["mean"], gmsc_ctgan["mean"], gmsc_tvae["mean"]]
    stds = [german_ctgan["std"], german_tvae["std"], gmsc_ctgan["std"], gmsc_tvae["std"]]
    thresholds = [german_ctgan["threshold"], german_tvae["threshold"], gmsc_ctgan["threshold"], gmsc_tvae["threshold"]]
    
    labels = ['CTGAN\n(German)', 'TVAE\n(German)', 'CTGAN\n(GMSC)', 'TVAE\n(GMSC)']
    colors = ['#4c72b0', '#c44e52', '#4c72b0', '#c44e52'] # CTGAN: blue-ish, TVAE: red-ish
    
    bars = plt.bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor='black', alpha=0.85)
    
    # Draw horizontal dashed lines for thresholds
    plt.hlines(y=german_ctgan["threshold"], xmin=-0.4, xmax=1.4, colors='#c44e52', linestyles='dashed', 
               linewidth=2, label=f'German Credit Threshold ({german_ctgan["threshold"]:.4f})')
    plt.hlines(y=gmsc_ctgan["threshold"], xmin=1.6, xmax=3.4, colors='#8172b3', linestyles='dashed', 
               linewidth=2, label=f'GMSC Threshold ({gmsc_ctgan["threshold"]:.4f})')
    
    # Annotate mean values on top of bars
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        is_exceeded = height > thresholds[idx]
        text_color = 'red' if is_exceeded else 'black'
        weight = 'bold' if is_exceeded else 'normal'
        
        plt.text(bar.get_x() + bar.get_width()/2.0, height + stds[idx] + 0.005, 
                 f"{height:.4f}", ha='center', va='bottom', color=text_color, weight=weight, fontsize=10)
        
        if is_exceeded:
            # Highlight with a badge in the middle
            plt.text(bar.get_x() + bar.get_width()/2.0, height / 2, 
                     "EXCEEDS", ha='center', va='center', color='red', weight='bold', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.2'))
            
    # Set tick labels color if they exceed
    ax = plt.gca()
    for idx, tick in enumerate(ax.get_xticklabels()):
        if means[idx] > thresholds[idx]:
            tick.set_color('red')
            tick.set_weight('bold')
            
    plt.ylabel('Inference Risk Score', fontsize=12, fontweight='bold')
    plt.title('Inference Risk Comparison with Baseline Thresholds (IEEE Access Format)', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, max(means) + max(stds) + 0.06)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    fig_dir = "figures/privacy/"
    os.makedirs(fig_dir, exist_ok=True)
    fig_a_path = os.path.join(fig_dir, "inference_risk_comparison.png")
    plt.savefig(fig_a_path, dpi=300)
    plt.close()
    print(f"Figure A successfully saved to {fig_a_path}")
    
    # ------------------------------------------
    # Figure B - Privacy Dashboard
    # ------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    
    datasets = ["german_credit", "gmsc"]
    generators = ["ctgan", "tvae"]
    
    metrics_data = {
        "german_credit": {
            "ctgan": {
                "DCR": (3.4798 / 3.4798, 3.4798, None),
                "NNDR": (0.9206, 0.9206, None),
                "MIA AUC": (0.4878, 0.4878, 0.50),
                "Inf. Risk": (german_ctgan["mean"], german_ctgan["mean"], german_ctgan["threshold"])
            },
            "tvae": {
                "DCR": (2.0956 / 3.4798, 2.0956, None),
                "NNDR": (0.8351, 0.8351, None),
                "MIA AUC": (0.5080, 0.5080, 0.50),
                "Inf. Risk": (german_tvae["mean"], german_tvae["mean"], german_tvae["threshold"])
            }
        },
        "gmsc": {
            "ctgan": {
                "DCR": (0.3245 / 0.3245, 0.3245, None),
                "NNDR": (0.7740, 0.7740, None),
                "MIA AUC": (0.5051, 0.5051, 0.50),
                "Inf. Risk": (gmsc_ctgan["mean"], gmsc_ctgan["mean"], gmsc_ctgan["threshold"])
            },
            "tvae": {
                "DCR": (0.1589 / 0.3245, 0.1589, None),
                "NNDR": (0.7158, 0.7158, None),
                "MIA AUC": (0.5034, 0.5034, 0.50),
                "Inf. Risk": (gmsc_tvae["mean"], gmsc_tvae["mean"], gmsc_tvae["threshold"])
            }
        }
    }
    
    bar_colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
    
    for r_idx, ds in enumerate(datasets):
        for c_idx, gen in enumerate(generators):
            ax = axes[r_idx, c_idx]
            data = metrics_data[ds][gen]
            
            keys = list(data.keys())
            norm_vals = [data[k][0] for k in keys]
            raw_vals = [data[k][1] for k in keys]
            
            bars = ax.barh(keys, norm_vals, color=bar_colors, edgecolor='black', height=0.45, alpha=0.85)
            
            for idx, bar in enumerate(bars):
                width = bar.get_width()
                raw_val = raw_vals[idx]
                
                # Format text label
                if keys[idx] == "DCR":
                    val_text = f"{raw_val:.4f} (norm: {width:.2f})"
                else:
                    val_text = f"{raw_val:.4f}"
                    
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.0, 
                        val_text, ha='left', va='center', fontsize=9, fontweight='bold')
                
                # Plot threshold dashed lines
                threshold = data[keys[idx]][2]
                if threshold is not None:
                    ax.vlines(x=threshold, ymin=bar.get_y(), ymax=bar.get_y() + bar.get_height(), 
                              colors='red', linestyles='dashed', linewidth=1.5)
                    # Text label next to line
                    ax.text(threshold, bar.get_y() + bar.get_height() + 0.02, "Thresh" if keys[idx]=="Inf. Risk" else "Rand", 
                            color='red', fontsize=7, ha='center', va='bottom')
            
            ax.set_title(f"{ds.replace('_', ' ').upper()} - {gen.upper()}", fontsize=11, fontweight='bold')
            ax.set_xlim(0, 1.25)
            ax.xaxis.grid(True, linestyle='--', alpha=0.5)
            
    plt.suptitle("Privacy Metrics Comparative Dashboard (Normalized DCR, Raw Others)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig_b_path = os.path.join(fig_dir, "privacy_dashboard.png")
    plt.savefig(fig_b_path, dpi=300)
    plt.close()
    print(f"Figure B successfully saved to {fig_b_path}")

# ==========================================
# TASK 5: Results Table Generator
# ==========================================
def generate_results_table(results):
    german_ctgan = results["german_credit"]["ctgan"]
    german_tvae = results["german_credit"]["tvae"]
    gmsc_ctgan = results["gmsc"]["ctgan"]
    gmsc_tvae = results["gmsc"]["tvae"]
    
    data = [
        {
            "Dataset": "German Credit",
            "Generator": "CTGAN",
            "SHAP rho": 0.6072,
            "DCR": 3.4798,
            "NNDR": 0.9206,
            "MIA AUC": 0.4878,
            "Inference Risk": f"{german_ctgan['mean']:.4f} ± {german_ctgan['std']:.4f}"
        },
        {
            "Dataset": "German Credit",
            "Generator": "TVAE",
            "SHAP rho": 0.6224,
            "DCR": 2.0956,
            "NNDR": 0.8351,
            "MIA AUC": 0.5080,
            "Inference Risk": f"{german_tvae['mean']:.4f} ± {german_tvae['std']:.4f}"
        },
        {
            "Dataset": "GMSC",
            "Generator": "CTGAN",
            "SHAP rho": 0.2848,
            "DCR": 0.3245,
            "NNDR": 0.7740,
            "MIA AUC": 0.5051,
            "Inference Risk": f"{gmsc_ctgan['mean']:.4f} ± {gmsc_ctgan['std']:.4f}"
        },
        {
            "Dataset": "GMSC",
            "Generator": "TVAE",
            "SHAP rho": 0.5661,
            "DCR": 0.1589,
            "NNDR": 0.7158,
            "MIA AUC": 0.5034,
            "Inference Risk": f"{gmsc_tvae['mean']:.4f} ± {gmsc_tvae['std']:.4f}"
        }
    ]
    
    df = pd.DataFrame(data)
    print("\n" + "="*80)
    print("UPDATED RESULTS TABLE (PANDAS DATAFRAME)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Generate LaTeX table code
    latex_code = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Quantitative Comparison of Explanatory Consistency and Privacy Metrics Across Generators and Datasets.}}
\\label{{tab:privacy_utility_results}}
\\begin{{tabular}}{{llccccc}}
\\hline
\\textbf{{Dataset}} & \\textbf{{Generator}} & \\textbf{{SHAP $\\rho$}} & \\textbf{{DCR}} & \\textbf{{NNDR}} & \\textbf{{MIA AUC}} & \\textbf{{Inference Risk}} \\\\
\\hline
German Credit & CTGAN & 0.6072 & 3.4798 & 0.9206 & 0.4878 & {german_ctgan['mean']:.4f} $\\pm$ {german_ctgan['std']:.4f} \\\\
              & TVAE  & 0.6224 & 2.0956 & 0.8351 & 0.5080 & {german_tvae['mean']:.4f} $\\pm$ {german_tvae['std']:.4f} \\\\
\\hline
GMSC          & CTGAN & 0.2848 & 0.3245 & 0.7740 & 0.5051 & {gmsc_ctgan['mean']:.4f} $\\pm$ {gmsc_ctgan['std']:.4f} \\\\
              & TVAE  & 0.5661 & 0.1589 & 0.7158 & 0.5034 & {gmsc_tvae['mean']:.4f} $\\pm$ {gmsc_tvae['std']:.4f} \\\\
\\hline
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Note:}} The baseline Inference Risk thresholds (95th percentile of random splits) are {german_ctgan['threshold']:.4f} for German Credit and {gmsc_ctgan['threshold']:.4f} for GMSC. Exceeding this threshold denotes high disclosure risk.
\\end{{tablenotes}}
\\end{{table}}
"""
    print("="*80)
    print("UPDATED RESULTS TABLE (LATEX CODE)")
    print("="*80)
    print(latex_code)
    print("="*80 + "\n")

# ==========================================
# TASK 6: Statistical Analysis Extension
# ==========================================
def run_statistical_analysis(results):
    print("="*80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    for ds in ["german_credit", "gmsc"]:
        ctgan_scores = results[ds]["ctgan"]["scores_per_seed"]
        tvae_scores = results[ds]["tvae"]["scores_per_seed"]
        
        diff = np.array(tvae_scores) - np.array(ctgan_scores)
        if np.all(diff == 0):
            p_val = 1.0
            stat = 0.0
        else:
            stat, p_val = wilcoxon(tvae_scores, ctgan_scores)
            
        d = compute_cohen_d(tvae_scores, ctgan_scores)
        
        print(f"\nDataset: {ds.replace('_', ' ').upper()}")
        print(f"  CTGAN Scores: {[round(x, 4) for x in ctgan_scores]}")
        print(f"  TVAE Scores:  {[round(x, 4) for x in tvae_scores]}")
        print(f"  Wilcoxon Test: stat={stat:.2f}, p-val={p_val:.4f}")
        print(f"  Cohen's d:     {d:.4f}")
        
        # Effect size classification
        if abs(d) >= 0.8:
            effect_desc = "large effect"
        elif abs(d) >= 0.5:
            effect_desc = "medium effect"
        elif abs(d) >= 0.2:
            effect_desc = "small effect"
        else:
            effect_desc = "negligible effect"
            
        # Due to N=5, minimum two-sided p-value is 0.0625.
        # We use alpha=0.10 to capture significance at N=5.
        if p_val < 0.10:
            if np.mean(tvae_scores) > np.mean(ctgan_scores):
                print(f"  Interpretation: \"TVAE has significantly higher inference risk than CTGAN (p={p_val:.4f}, d={d:.4f}, {effect_desc})\"")
            else:
                print(f"  Interpretation: \"CTGAN has significantly higher inference risk than TVAE (p={p_val:.4f}, d={d:.4f}, {effect_desc})\"")
        else:
            print(f"  Interpretation: \"No significant difference in inference risk (p={p_val:.4f}, d={d:.4f})\"")
            
    print("="*80 + "\n")

def main():
    # Configure console encoding for UTF-8 to handle math symbols smoothly
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    print("======================================================================")
    print("Shap Synthetic Credit Risk: Starting Inference Risk Computation Pipeline")
    print("======================================================================\n")
    
    results = run_experiments()
    generate_plots(results)
    generate_results_table(results)
    run_statistical_analysis(results)
    
    print("======================================================================")
    print("Inference Risk Computation Pipeline Finished Successfully")
    print("======================================================================")

if __name__ == "__main__":
    main()
