import os
import sys
import json
import pandas as pd
import numpy as np

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.statistics.validation import compute_bootstrap_ci, compute_paired_cohens_d, run_paired_wilcoxon

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_group_stats(data_list):
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
    stat, p_val = run_paired_wilcoxon(g1_data, g2_data)
    d = compute_paired_cohens_d(g1_data, g2_data)
    return {
        "comparison": f"{g1_name} vs {g2_name}",
        "wilcoxon_stat": stat,
        "p_value": p_val,
        "cohens_d": d
    }

def analyze_dataset_statistics(dataset_name):
    print("\n" + "="*80)
    print(f"Running Statistical Validation for Dataset: {dataset_name.upper()}")
    print("="*80)
    
    # 1. Load summary files
    real_path = f"results/baseline/{dataset_name}_real_xgboost_summary.json"
    ctgan_path = f"results/ctgan/{dataset_name}_ctgan_xgboost_summary.json"
    tvae_path = f"results/tvae/{dataset_name}_tvae_xgboost_summary.json"
    shap_path = f"results/shap/{dataset_name}_shap_consistency_summary.json"
    privacy_path = f"results/privacy/{dataset_name}_privacy_summary.json"
    
    for path in [real_path, ctgan_path, tvae_path, shap_path, privacy_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required summary file {path} not found. Run previous phases first.")
            
    real_sum = load_json(real_path)
    ctgan_sum = load_json(ctgan_path)
    tvae_sum = load_json(tvae_path)
    shap_sum = load_json(shap_path)
    privacy_sum = load_json(privacy_path)
    
    # 2. Extract arrays
    real_auc = [run["roc_auc"] for run in real_sum["runs"]]
    real_f1 = [run["f1"] for run in real_sum["runs"]]
    
    ctgan_auc = [run["roc_auc"] for run in ctgan_sum["runs"]]
    ctgan_f1 = [run["f1"] for run in ctgan_sum["runs"]]
    
    tvae_auc = [run["roc_auc"] for run in tvae_sum["runs"]]
    tvae_f1 = [run["f1"] for run in tvae_sum["runs"]]
    
    ctgan_shap = [run["spearman_rho"] for run in shap_sum["ctgan_runs"]]
    tvae_shap = [run["spearman_rho"] for run in shap_sum["tvae_runs"]]
    
    ctgan_dcr = [run["dcr_mean"] for run in privacy_sum["ctgan_runs"]]
    tvae_dcr = [run["dcr_mean"] for run in privacy_sum["tvae_runs"]]
    
    ctgan_nndr = [run["nndr_mean"] for run in privacy_sum["ctgan_runs"]]
    tvae_nndr = [run["nndr_mean"] for run in privacy_sum["tvae_runs"]]
    
    ctgan_mia = [run["mia_auc"] for run in privacy_sum["ctgan_runs"]]
    tvae_mia = [run["mia_auc"] for run in privacy_sum["tvae_runs"]]
    
    # 3. Compute stats with CIs
    stats = {
        "real_auc": compute_group_stats(real_auc),
        "real_f1": compute_group_stats(real_f1),
        "ctgan_auc": compute_group_stats(ctgan_auc),
        "ctgan_f1": compute_group_stats(ctgan_f1),
        "tvae_auc": compute_group_stats(tvae_auc),
        "tvae_f1": compute_group_stats(tvae_f1),
        "ctgan_shap": compute_group_stats(ctgan_shap),
        "tvae_shap": compute_group_stats(tvae_shap),
        "ctgan_dcr": compute_group_stats(ctgan_dcr),
        "tvae_dcr": compute_group_stats(tvae_dcr),
        "ctgan_nndr": compute_group_stats(ctgan_nndr),
        "tvae_nndr": compute_group_stats(tvae_nndr),
        "ctgan_mia": compute_group_stats(ctgan_mia),
        "tvae_mia": compute_group_stats(tvae_mia)
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
        "shap_tvae_vs_ctgan": run_paired_comparison("TVAE SHAP", "CTGAN SHAP", tvae_shap, ctgan_shap),
        
        # Privacy DCR / NNDR comparison
        "dcr_tvae_vs_ctgan": run_paired_comparison("TVAE DCR", "CTGAN DCR", tvae_dcr, ctgan_dcr),
        "nndr_tvae_vs_ctgan": run_paired_comparison("TVAE NNDR", "CTGAN NNDR", tvae_nndr, ctgan_nndr),
        "mia_tvae_vs_ctgan": run_paired_comparison("TVAE MIA", "CTGAN MIA", tvae_mia, ctgan_mia)
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
    summary_path = os.path.join(results_dir, f"{dataset_name}_statistical_validation.json")
    with open(summary_path, 'w') as f:
        json.dump(dataset_summary, f, indent=2)
        
    print(f"Statistical Validation summary written to {summary_path}")
    
    # Print publication-ready output
    print(f"\n--- {dataset_name.upper()} EFFECT SIZES & SIGNIFICANCE ---")
    for key, test in tests.items():
        sig = "Significant (*)" if test["p_value"] < 0.05 else "Not Significant"
        print(f"  * {test['comparison']}:")
        print(f"    - p-value: {test['p_value']:.4f} ({sig})")
        print(f"    - Cohen's d: {test['cohens_d']:.4f}")
        
    return dataset_summary

def main():
    print("======================================================================")
    print("Shap Synthetic Credit Risk: Starting Statistical Validation")
    print("======================================================================\n")
    
    analyze_dataset_statistics("german_credit")
    analyze_dataset_statistics("gmsc")
    
    print("\n======================================================================")
    print("Statistical Validation Execution Finished Successfully")
    print("======================================================================")

if __name__ == "__main__":
    main()
