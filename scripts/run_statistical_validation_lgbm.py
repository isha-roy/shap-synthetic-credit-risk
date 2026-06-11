import os
import sys
import json
import pandas as pd
import numpy as np

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.statistics.validation import compute_bootstrap_ci, compute_paired_cohens_d, run_paired_wilcoxon

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_group_stats(data_list):
    if not data_list:
        return {
            "raw_values": [],
            "mean": 0.0,
            "std": 0.0,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0
        }
    arr = np.array(data_list)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    ci_lower, ci_upper = compute_bootstrap_ci(arr)
    return {
        "raw_values": data_list,
        "mean": mean_val,
        "std": std_val,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper
    }

def run_paired_comparison(g1_name, g2_name, g1_data, g2_data):
    if not g1_data or not g2_data or len(g1_data) != len(g2_data):
        return {
            "comparison": f"{g1_name} vs {g2_name}",
            "wilcoxon_stat": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0
        }
    stat, p_val = run_paired_wilcoxon(g1_data, g2_data)
    d = compute_paired_cohens_d(g1_data, g2_data)
    return {
        "comparison": f"{g1_name} vs {g2_name}",
        "wilcoxon_stat": stat,
        "p_value": p_val,
        "cohens_d": d
    }

def analyze_dataset_statistics_lgbm(dataset_name):
    print("\n" + "="*80)
    print(f"Running Statistical Validation for Dataset: {dataset_name.upper()} (LightGBM)")
    print("="*80)
    
    # 1. Load summary files
    real_path = f"results/baseline/{dataset_name}_real_lgbm_summary.json"
    ctgan_path = f"results/ctgan/{dataset_name}_ctgan_lgbm_summary.json"
    tvae_path = f"results/tvae/{dataset_name}_tvae_lgbm_summary.json"
    dp10_path = f"results/tvae_dp_eps10.0/{dataset_name}_tvae_dp_eps10.0_lgbm_summary.json"
    dp5_path = f"results/tvae_dp_eps5.0/{dataset_name}_tvae_dp_eps5.0_lgbm_summary.json"
    dp1_path = f"results/tvae_dp_eps1.0/{dataset_name}_tvae_dp_eps1.0_lgbm_summary.json"
    shap_path = f"results/shap/{dataset_name}_shap_consistency_lgbm_summary.json"
    
    # Check baseline paths
    for path in [real_path, ctgan_path, tvae_path, shap_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required summary file {path} not found. Run baseline & synthetic modeling first.")
            
    real_sum = load_json(real_path)
    ctgan_sum = load_json(ctgan_path)
    tvae_sum = load_json(tvae_path)
    dp10_sum = load_json(dp10_path)
    dp5_sum = load_json(dp5_path)
    dp1_sum = load_json(dp1_path)
    shap_sum = load_json(shap_path)
    
    # 2. Extract arrays
    real_auc = [run["roc_auc"] for run in real_sum["runs"]]
    real_f1 = [run["f1"] for run in real_sum["runs"]]
    
    ctgan_auc = [run["roc_auc"] for run in ctgan_sum["runs"]]
    ctgan_f1 = [run["f1"] for run in ctgan_sum["runs"]]
    
    tvae_auc = [run["roc_auc"] for run in tvae_sum["runs"]]
    tvae_f1 = [run["f1"] for run in tvae_sum["runs"]]
    
    dp10_auc = [run["roc_auc"] for run in dp10_sum["runs"]] if dp10_sum else []
    dp10_f1 = [run["f1"] for run in dp10_sum["runs"]] if dp10_sum else []
    
    dp5_auc = [run["roc_auc"] for run in dp5_sum["runs"]] if dp5_sum else []
    dp5_f1 = [run["f1"] for run in dp5_sum["runs"]] if dp5_sum else []
    
    dp1_auc = [run["roc_auc"] for run in dp1_sum["runs"]] if dp1_sum else []
    dp1_f1 = [run["f1"] for run in dp1_sum["runs"]] if dp1_sum else []
    
    ctgan_shap = [run["spearman_rho"] for run in shap_sum["ctgan_runs"]]
    tvae_shap = [run["spearman_rho"] for run in shap_sum["tvae_runs"]]
    
    # 3. Compute stats with CIs
    stats = {
        "real_auc": compute_group_stats(real_auc),
        "real_f1": compute_group_stats(real_f1),
        "ctgan_auc": compute_group_stats(ctgan_auc),
        "ctgan_f1": compute_group_stats(ctgan_f1),
        "tvae_auc": compute_group_stats(tvae_auc),
        "tvae_f1": compute_group_stats(tvae_f1),
        "dp10_auc": compute_group_stats(dp10_auc),
        "dp10_f1": compute_group_stats(dp10_f1),
        "dp5_auc": compute_group_stats(dp5_auc),
        "dp5_f1": compute_group_stats(dp5_f1),
        "dp1_auc": compute_group_stats(dp1_auc),
        "dp1_f1": compute_group_stats(dp1_f1),
        "ctgan_shap": compute_group_stats(ctgan_shap),
        "tvae_shap": compute_group_stats(tvae_shap)
    }
    
    # 4. Run paired hypothesis tests
    tests = {
        # ROC-AUC Utility comparison
        "auc_tvae_vs_ctgan": run_paired_comparison("TVAE AUC", "CTGAN AUC", tvae_auc, ctgan_auc),
        "auc_tvae_vs_real": run_paired_comparison("TVAE AUC", "Real AUC", tvae_auc, real_auc),
        "auc_ctgan_vs_real": run_paired_comparison("CTGAN AUC", "Real AUC", ctgan_auc, real_auc),
        
        # F1 Utility comparison
        "f1_tvae_vs_ctgan": run_paired_comparison("TVAE F1", "CTGAN F1", tvae_f1, ctgan_f1),
        "f1_tvae_vs_real": run_paired_comparison("TVAE F1", "Real F1", tvae_f1, real_f1),
        "f1_ctgan_vs_real": run_paired_comparison("CTGAN F1", "Real F1", ctgan_f1, real_f1),
        
        # SHAP Consistency comparison
        "shap_tvae_vs_ctgan": run_paired_comparison("TVAE SHAP", "CTGAN SHAP", tvae_shap, ctgan_shap)
    }
    
    # 5. Compile results
    dataset_summary = {
        "dataset": dataset_name,
        "metrics_summary": stats,
        "paired_tests": tests
    }
    
    # Write to statistics summary file
    results_dir = "results/statistics/"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"{dataset_name}_statistical_validation_lgbm.json")
    with open(summary_path, 'w') as f:
        json.dump(dataset_summary, f, indent=2)
        
    print(f"Statistical Validation summary written to {summary_path}")
    
    # Print publication-ready output
    print(f"\n--- {dataset_name.upper()} EFFECT SIZES & SIGNIFICANCE (LightGBM) ---")
    for key, test in tests.items():
        sig = "Significant (*)" if test["p_value"] < 0.05 else "Not Significant"
        print(f"  * {test['comparison']}:")
        print(f"    - p-value: {test['p_value']:.4f} ({sig})")
        print(f"    - Cohen's d: {test['cohens_d']:.4f}")
        
    return dataset_summary

def main():
    print("======================================================================")
    print("Shap Synthetic Credit Risk: Starting LightGBM Statistical Validation")
    print("======================================================================\n")
    
    analyze_dataset_statistics_lgbm("german_credit")
    
    if os.path.exists("data/raw/cs-training.csv"):
        analyze_dataset_statistics_lgbm("gmsc")
    else:
        print("GMSC raw data missing. Skipping GMSC statistical validation.")
        
    print("\n======================================================================")
    print("LightGBM Statistical Validation Execution Finished Successfully")
    print("======================================================================")

if __name__ == "__main__":
    main()
