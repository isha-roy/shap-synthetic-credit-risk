import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc
from src.shap_analysis.consistency_lgbm import compute_model_shap_lgbm, evaluate_ranking_similarity_lgbm

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_dataset_shap_consistency_lgbm(dataset_name):
    print("\n" + "="*80)
    print(f"Starting SHAP Consistency Analysis for Dataset: {dataset_name.upper()} (LightGBM)")
    print("="*80)
    
    seeds = load_json("configs/seeds/seeds.json")["seeds"]
    
    if dataset_name == "german_credit":
        dataset_suffix = "german"
    elif dataset_name == "gmsc":
        dataset_suffix = "gmsc"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    generators = ["ctgan", "tvae", "tvae_dp_eps10.0", "tvae_dp_eps5.0", "tvae_dp_eps1.0"]
    generator_results = {gen: [] for gen in generators}
    
    for seed in seeds:
        print(f"\n--- Processing Seed {seed} ---")
        
        # 1. Regenerate real test split for this seed
        if dataset_name == "german_credit":
            preprocess_german_credit(seed=seed)
        else:
            preprocess_gmsc(seed=seed)
            
        X_test_real = pd.read_csv(f"data/processed/X_test_{dataset_suffix}.csv")
        
        for gen in generators:
            model_path = f"models/{gen}/{dataset_name}_{gen}_lgbm_seed{seed}_v1.joblib"
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                # Compute SHAP values & plots
                compute_model_shap_lgbm(
                    model=model, 
                    X_test=X_test_real, 
                    dataset_name=dataset_name, 
                    generator_type=gen, 
                    seed=seed
                )
                # Evaluate ranking similarity against baseline
                similarity = evaluate_ranking_similarity_lgbm(
                    dataset_name=dataset_name, 
                    generator_type=gen, 
                    seed=seed
                )
                generator_results[gen].append(similarity)
            else:
                print(f"Model not found at {model_path}. Skipping.")
                
    # Save summaries
    summary_dir = "results/shap/"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{dataset_name}_shap_consistency_lgbm_summary.json")
    
    summary = {
        "dataset": dataset_name,
        "seeds": seeds,
    }
    for gen in generators:
        summary[f"{gen}_runs"] = generator_results[gen]
        
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSHAP Consistency Summary written to {summary_path}")
    
    # Render console comparison table
    print(f"\nAggregated LightGBM SHAP Results (Mean ± Std) for {dataset_name.upper()}:")
    
    for gen in generators:
        runs = generator_results[gen]
        if len(runs) == 0:
            continue
        df_runs = pd.DataFrame(runs)
        mean_rho = df_runs["spearman_rho"].mean()
        std_rho = df_runs["spearman_rho"].std()
        
        mean_top5 = df_runs["top5_overlap_count"].mean()
        std_top5 = df_runs["top5_overlap_count"].std()
        
        mean_top10 = df_runs["top10_overlap_count"].mean()
        std_top10 = df_runs["top10_overlap_count"].std()
        
        print(f"  * {gen}:")
        print(f"    - Spearman Rank Corr (rho): {mean_rho:.4f} ± {std_rho:.4f}")
        print(f"    - Top 5 Features Overlap:   {mean_top5:.1f} ± {std_top5:.2f} (out of 5)")
        print(f"    - Top 10 Features Overlap:  {mean_top10:.1f} ± {std_top10:.2f} (out of 10)")

def main():
    print("======================================================================")
    print("Shap Synthetic Credit Risk: Starting LightGBM SHAP Consistency Evaluation")
    print("======================================================================\n")
    
    run_dataset_shap_consistency_lgbm("german_credit")
    
    if os.path.exists("data/raw/cs-training.csv"):
        run_dataset_shap_consistency_lgbm("gmsc")
    else:
        print("GMSC raw data missing. Skipping GMSC SHAP consistency check.")
        
    print("\n======================================================================")
    print("LightGBM SHAP Consistency Evaluation Finished Successfully")
    print("======================================================================")

if __name__ == "__main__":
    main()
