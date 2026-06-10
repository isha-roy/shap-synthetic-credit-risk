import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc
from src.shap_analysis.consistency import compute_model_shap, evaluate_ranking_similarity

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_dataset_shap_consistency(dataset_name):
    print("\n" + "="*80)
    print(f"Starting SHAP Consistency Analysis for Dataset: {dataset_name.upper()}")
    print("="*80)
    
    seeds = load_json("configs/seeds/seeds.json")["seeds"]
    
    if dataset_name == "german_credit":
        dataset_suffix = "german"
    elif dataset_name == "gmsc":
        dataset_suffix = "gmsc"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    ctgan_runs = []
    tvae_runs = []
    
    for seed in seeds:
        print(f"\n--- Processing Seed {seed} ---")
        
        # 1. Regenerate real test split for this seed
        if dataset_name == "german_credit":
            preprocess_german_credit(seed=seed)
        else:
            preprocess_gmsc(seed=seed)
            
        X_test_real = pd.read_csv(f"data/processed/X_test_{dataset_suffix}.csv")
        
        # 2. CTGAN SHAP computation & evaluation
        ctgan_model_path = f"models/ctgan/{dataset_name}_ctgan_xgboost_seed{seed}_v1.joblib"
        if os.path.exists(ctgan_model_path):
            ctgan_model = joblib.load(ctgan_model_path)
            # Compute SHAP values & plots
            compute_model_shap(
                model=ctgan_model, 
                X_test=X_test_real, 
                dataset_name=dataset_name, 
                generator_type="ctgan", 
                seed=seed
            )
            # Evaluate ranking similarity against baseline
            ctgan_similarity = evaluate_ranking_similarity(
                dataset_name=dataset_name, 
                generator_type="ctgan", 
                seed=seed
            )
            ctgan_runs.append(ctgan_similarity)
        else:
            print(f"CTGAN model not found at {ctgan_model_path}. Skipping.")
            
        # 3. TVAE SHAP computation & evaluation
        tvae_model_path = f"models/tvae/{dataset_name}_tvae_xgboost_seed{seed}_v1.joblib"
        if os.path.exists(tvae_model_path):
            tvae_model = joblib.load(tvae_model_path)
            # Compute SHAP values & plots
            compute_model_shap(
                model=tvae_model, 
                X_test=X_test_real, 
                dataset_name=dataset_name, 
                generator_type="tvae", 
                seed=seed
            )
            # Evaluate ranking similarity against baseline
            tvae_similarity = evaluate_ranking_similarity(
                dataset_name=dataset_name, 
                generator_type="tvae", 
                seed=seed
            )
            tvae_runs.append(tvae_similarity)
        else:
            print(f"TVAE model not found at {tvae_model_path}. Skipping.")
            
    # Save summaries
    summary_dir = "results/shap/"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{dataset_name}_shap_consistency_summary.json")
    
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
        
    print(f"\nSHAP Consistency Summary written to {summary_path}")
    
    # Render console comparison table
    print(f"\nAggregated Results (Mean ± Std) for {dataset_name.upper()}:")
    
    for gen_type, runs in [("CTGAN", ctgan_runs), ("TVAE", tvae_runs)]:
        if len(runs) == 0:
            continue
        df_runs = pd.DataFrame(runs)
        mean_rho = df_runs["spearman_rho"].mean()
        std_rho = df_runs["spearman_rho"].std()
        
        mean_top5 = df_runs["top5_overlap_count"].mean()
        std_top5 = df_runs["top5_overlap_count"].std()
        
        mean_top10 = df_runs["top10_overlap_count"].mean()
        std_top10 = df_runs["top10_overlap_count"].std()
        
        print(f"  * {gen_type}:")
        print(f"    - Spearman Rank Corr (rho): {mean_rho:.4f} ± {std_rho:.4f}")
        print(f"    - Top 5 Features Overlap:   {mean_top5:.1f} ± {std_top5:.2f} (out of 5)")
        print(f"    - Top 10 Features Overlap:  {mean_top10:.1f} ± {std_top10:.2f} (out of 10)")

def main():
    print("======================================================================")
    print("Shap Synthetic Credit Risk: Starting SHAP Consistency Evaluation")
    print("======================================================================\n")
    
    run_dataset_shap_consistency("german_credit")
    run_dataset_shap_consistency("gmsc")
    
    print("\n======================================================================")
    print("SHAP Consistency Evaluation Finished Successfully")
    print("======================================================================")

if __name__ == "__main__":
    main()
