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

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_dataset_privacy_pipeline(dataset_name):
    print("\n" + "="*80)
    print(f"Starting Privacy Evaluation for Dataset: {dataset_name.upper()}")
    print("="*80)
    
    seeds = load_json("configs/seeds/seeds.json")["seeds"]
    
    # Load dataset-specific config
    if dataset_name == "german_credit":
        ds_config = load_json("configs/datasets/german_credit.json")
        target_column = ds_config["target_column"]
        dataset_suffix = "german"
    elif dataset_name == "gmsc":
        ds_config = load_json("configs/datasets/gmsc.json")
        target_column = ds_config["target_column"]
        dataset_suffix = "gmsc"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    ctgan_runs = []
    tvae_runs = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        
        # 1. Regenerate split files on disk for this seed
        if dataset_name == "german_credit":
            preprocess_german_credit(seed=seed)
        else:
            preprocess_gmsc(seed=seed)
            
        X_train_real = pd.read_csv(f"data/processed/X_train_{dataset_suffix}.csv")
        X_test_real = pd.read_csv(f"data/processed/X_test_{dataset_suffix}.csv")
        
        # Fit baseline preprocessing transformers on raw training data
        fitted_transformers = fit_real_preprocessing(dataset_name, seed)
        
        # 2. Evaluate CTGAN
        ctgan_csv_path = f"data/synthetic/{dataset_name}_ctgan_seed{seed}.csv"
        if os.path.exists(ctgan_csv_path):
            df_syn_ctgan = pd.read_csv(ctgan_csv_path)
            X_syn_ctgan, _ = preprocess_synthetic_data(
                df_synthetic=df_syn_ctgan,
                dataset_name=dataset_name,
                fitted=fitted_transformers,
                target_column=target_column
            )
            
            # Compute DCR, NNDR, and MIA AUC
            ctgan_priv = compute_dcr_nndr(X_syn_ctgan, X_train_real)
            ctgan_mia = compute_mia_auc(X_train_real, X_test_real, X_syn_ctgan)
            
            run_result = {
                "seed": seed,
                "dcr_mean": ctgan_priv["dcr_mean"],
                "dcr_median": ctgan_priv["dcr_median"],
                "dcr_5th_percentile": ctgan_priv["dcr_5th_percentile"],
                "nndr_mean": ctgan_priv["nndr_mean"],
                "nndr_median": ctgan_priv["nndr_median"],
                "mia_auc": ctgan_mia
            }
            ctgan_runs.append(run_result)
            print(f"  * CTGAN: Mean DCR = {run_result['dcr_mean']:.4f}, Mean NNDR = {run_result['nndr_mean']:.4f}, MIA AUC = {run_result['mia_auc']:.4f}")
        else:
            print(f"CTGAN synthetic file {ctgan_csv_path} not found. Skipping seed {seed}.")
            
        # 3. Evaluate TVAE
        tvae_csv_path = f"data/synthetic/{dataset_name}_tvae_seed{seed}.csv"
        if os.path.exists(tvae_csv_path):
            df_syn_tvae = pd.read_csv(tvae_csv_path)
            X_syn_tvae, _ = preprocess_synthetic_data(
                df_synthetic=df_syn_tvae,
                dataset_name=dataset_name,
                fitted=fitted_transformers,
                target_column=target_column
            )
            
            # Compute DCR, NNDR, and MIA AUC
            tvae_priv = compute_dcr_nndr(X_syn_tvae, X_train_real)
            tvae_mia = compute_mia_auc(X_train_real, X_test_real, X_syn_tvae)
            
            run_result = {
                "seed": seed,
                "dcr_mean": tvae_priv["dcr_mean"],
                "dcr_median": tvae_priv["dcr_median"],
                "dcr_5th_percentile": tvae_priv["dcr_5th_percentile"],
                "nndr_mean": tvae_priv["nndr_mean"],
                "nndr_median": tvae_priv["nndr_median"],
                "mia_auc": tvae_mia
            }
            tvae_runs.append(run_result)
            print(f"  * TVAE:  Mean DCR = {run_result['dcr_mean']:.4f}, Mean NNDR = {run_result['nndr_mean']:.4f}, MIA AUC = {run_result['mia_auc']:.4f}")
        else:
            print(f"TVAE synthetic file {tvae_csv_path} not found. Skipping seed {seed}.")
            
    # Save privacy summary to results
    results_dir = "results/privacy/"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"{dataset_name}_privacy_summary.json")
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                existing_summary = json.load(f)
            existing_seeds = existing_summary.get("seeds", [])
            existing_ctgan = existing_summary.get("ctgan_runs", [])
            existing_tvae = existing_summary.get("tvae_runs", [])
            
            new_seeds = [s for s in seeds if s not in existing_seeds]
            existing_seeds.extend(new_seeds)
            
            # Map CTGAN runs by seed
            ctgan_by_seed = {r["seed"]: r for r in existing_ctgan}
            for r in ctgan_runs:
                ctgan_by_seed[r["seed"]] = r
            existing_ctgan = [ctgan_by_seed[s] for s in existing_seeds if s in ctgan_by_seed]
            
            # Map TVAE runs by seed
            tvae_by_seed = {r["seed"]: r for r in existing_tvae}
            for r in tvae_runs:
                tvae_by_seed[r["seed"]] = r
            existing_tvae = [tvae_by_seed[s] for s in existing_seeds if s in tvae_by_seed]
            
            summary = {
                "dataset": dataset_name,
                "seeds": existing_seeds,
                "ctgan_runs": existing_ctgan,
                "tvae_runs": existing_tvae
            }
            print(f"Appended results for new seeds: {new_seeds}")
        except Exception as e:
            print(f"Error loading existing summary, overwriting: {e}")
            summary = {
                "dataset": dataset_name,
                "seeds": seeds,
                "ctgan_runs": ctgan_runs,
                "tvae_runs": tvae_runs
            }
    else:
        summary = {
            "dataset": dataset_name,
            "seeds": seeds,
            "ctgan_runs": ctgan_runs,
            "tvae_runs": tvae_runs
        }
        
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nPrivacy Summary written to {summary_path}")
    
    # Print console aggregate comparison table
    print(f"\nAggregated Privacy Metrics (Mean ± Std) for {dataset_name.upper()}:")
    for gen_type, runs in [("CTGAN", ctgan_runs), ("TVAE", tvae_runs)]:
        if len(runs) == 0:
            continue
        df_runs = pd.DataFrame(runs)
        
        mean_dcr = df_runs["dcr_mean"].mean()
        std_dcr = df_runs["dcr_mean"].std()
        
        mean_nndr = df_runs["nndr_mean"].mean()
        std_nndr = df_runs["nndr_mean"].std()
        
        mean_mia = df_runs["mia_auc"].mean()
        std_mia = df_runs["mia_auc"].std()
        
        print(f"  * {gen_type}:")
        print(f"    - DCR (Mean):  {mean_dcr:.4f} ± {std_dcr:.4f}")
        print(f"    - NNDR (Mean): {mean_nndr:.4f} ± {std_nndr:.4f}")
        print(f"    - MIA AUC:     {mean_mia:.4f}  ± {std_mia:.4f}")

def main():
    print("======================================================================")
    print("Shap Synthetic Credit Risk: Starting Privacy Evaluation")
    print("======================================================================\n")
    
    run_dataset_privacy_pipeline("german_credit")
    run_dataset_privacy_pipeline("gmsc")
    
    print("\n======================================================================")
    print("Privacy Evaluation Finished Successfully")
    print("======================================================================")

if __name__ == "__main__":
    main()
