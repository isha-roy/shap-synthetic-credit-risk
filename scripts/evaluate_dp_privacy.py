import torch
import os
import sys
import json
import pandas as pd
import numpy as np

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc
from src.models.synthetic import fit_real_preprocessing, preprocess_synthetic_data
from src.privacy.evaluation import compute_dcr_nndr, compute_mia_auc
from scripts.compute_inference_risk import compute_inference_risk, compute_inference_risk_threshold

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_dp_privacy_evaluation(dataset_name, epsilon_values):
    print("\n" + "="*80)
    print(f"Starting DP Privacy Evaluation for Dataset: {dataset_name.upper()}")
    print("="*80)
    
    seeds = load_json("configs/seeds/seeds.json")["seeds"]
    
    # Load dataset-specific config
    if dataset_name == "german_credit":
        ds_config = load_json("configs/datasets/german_credit.json")
        target_column = ds_config["target_column"]
        dataset_suffix = "german"
        preprocess_func = preprocess_german_credit
    elif dataset_name == "gmsc":
        ds_config = load_json("configs/datasets/gmsc.json")
        target_column = ds_config["target_column"]
        dataset_suffix = "gmsc"
        preprocess_func = preprocess_gmsc
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Get threshold for inference risk from seed 42 train split
    preprocess_func(seed=42)
    X_train_real_ref = pd.read_csv(f"data/processed/X_train_{dataset_suffix}.csv")
    threshold = compute_inference_risk_threshold(X_train_real_ref, n_splits=50, random_state=42)
    print(f"Estimated baseline threshold (95th pct): {threshold:.4f}")
    
    # Store runs grouped by epsilon
    results_by_eps = {str(eps): [] for eps in epsilon_values}
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        
        # Regenerate real splits for this seed
        preprocess_func(seed=seed)
        X_train_real = pd.read_csv(f"data/processed/X_train_{dataset_suffix}.csv")
        X_test_real = pd.read_csv(f"data/processed/X_test_{dataset_suffix}.csv")
        
        # Fit baseline preprocessing transformers
        fitted_transformers = fit_real_preprocessing(dataset_name, seed)
        
        for eps in epsilon_values:
            eps_str = str(eps)
            csv_path = f"data/synthetic/{dataset_name}_tvae_dp_eps{eps}_seed{seed}.csv"
            
            if not os.path.exists(csv_path):
                print(f"  * Epsilon {eps}: Synthetic file not found. Skipping.")
                continue
                
            df_syn = pd.read_csv(csv_path)
            X_syn, _ = preprocess_synthetic_data(
                df_synthetic=df_syn,
                dataset_name=dataset_name,
                fitted=fitted_transformers,
                target_column=target_column
            )
            
            # Compute privacy metrics
            priv_metrics = compute_dcr_nndr(X_syn, X_train_real)
            mia_auc = compute_mia_auc(X_train_real, X_test_real, X_syn)
            inf_risk, _, _ = compute_inference_risk(X_syn, X_train_real)
            
            run_result = {
                "seed": seed,
                "dcr_mean": priv_metrics["dcr_mean"],
                "dcr_median": priv_metrics["dcr_median"],
                "dcr_5th_percentile": priv_metrics["dcr_5th_percentile"],
                "nndr_mean": priv_metrics["nndr_mean"],
                "nndr_median": priv_metrics["nndr_median"],
                "mia_auc": mia_auc,
                "inference_risk": inf_risk
            }
            results_by_eps[eps_str].append(run_result)
            print(f"  * Epsilon {eps:2}: Mean DCR = {run_result['dcr_mean']:.4f}, MIA AUC = {run_result['mia_auc']:.4f}, Inference Risk = {run_result['inference_risk']:.4f}")
            
    # Save DP privacy summary to results
    results_dir = "results/privacy/"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"{dataset_name}_dp_privacy_summary.json")
    
    summary = {
        "dataset": dataset_name,
        "seeds": seeds,
        "threshold": threshold,
        "runs": results_by_eps
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nDP Privacy Summary written to {summary_path}")
    
    # Print aggregate comparison
    print(f"\nAggregated DP Privacy Metrics (Mean ± Std) for {dataset_name.upper()} (Threshold = {threshold:.4f}):")
    for eps in epsilon_values:
        eps_str = str(eps)
        runs = results_by_eps[eps_str]
        if not runs:
            continue
        df_runs = pd.DataFrame(runs)
        
        mean_dcr = df_runs["dcr_mean"].mean()
        std_dcr = df_runs["dcr_mean"].std()
        
        mean_nndr = df_runs["nndr_mean"].mean()
        std_nndr = df_runs["nndr_mean"].std()
        
        mean_mia = df_runs["mia_auc"].mean()
        std_mia = df_runs["mia_auc"].std()
        
        mean_ir = df_runs["inference_risk"].mean()
        std_ir = df_runs["inference_risk"].std()
        
        status = "PASS" if mean_ir < threshold else "FAIL"
        print(f"  * Epsilon = {eps:2}:")
        print(f"    - DCR (Mean):      {mean_dcr:.4f} ± {std_dcr:.4f}")
        print(f"    - NNDR (Mean):     {mean_nndr:.4f} ± {std_nndr:.4f}")
        print(f"    - MIA AUC:         {mean_mia:.4f} ± {std_mia:.4f}")
        print(f"    - Inference Risk:  {mean_ir:.4f} ± {std_ir:.4f}  [{status}]")

def main():
    # Configure console encoding for UTF-8
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    print("======================================================================")
    print("Shap Synthetic Credit Risk: Starting DP Privacy Evaluation")
    print("======================================================================\n")
    
    # Load epsilon values from config
    dp_config = load_json("configs/experiments/tvae_dp_default.json")
    epsilon_values = dp_config["dp_parameters"]["epsilon_values"]
    
    run_dp_privacy_evaluation("german_credit", epsilon_values)
    run_dp_privacy_evaluation("gmsc", epsilon_values)
    
    print("\n======================================================================")
    print("DP Privacy Evaluation Finished Successfully")
    print("======================================================================")

if __name__ == "__main__":
    main()
