import os
import sys
import json
import pandas as pd
import numpy as np

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc
from src.models.baseline_lgbm import tune_lightgbm, train_baseline_model_lgbm, compute_shap_reference_lgbm

def load_seeds(seeds_path="configs/seeds/seeds.json"):
    with open(seeds_path, 'r') as f:
        data = json.load(f)
    return data["seeds"]

def run_baseline_pipeline(dataset_name):
    print("\n" + "="*60)
    print(f"Starting Baseline LightGBM Pipeline for Dataset: {dataset_name}")
    print("="*60)
    
    seeds = load_seeds()
    first_seed = seeds[0]
    
    # 1. Generate splits for the first seed to perform hyperparameter tuning
    if dataset_name == "german_credit":
        preprocess_german_credit(seed=first_seed)
        X_train = pd.read_csv("data/processed/X_train_german.csv")
        y_train = pd.read_csv("data/processed/y_train_german.csv").values.ravel()
    else:
        preprocess_gmsc(seed=first_seed)
        X_train = pd.read_csv("data/processed/X_train_gmsc.csv")
        y_train = pd.read_csv("data/processed/y_train_gmsc.csv").values.ravel()
        
    # 2. Hyperparameter tuning on the first seed
    best_params = tune_lightgbm(X_train, y_train, dataset_name)
    print(f"Optimal parameters selected: {best_params}")
    
    all_metrics = []
    
    # 3. Loop over all seeds using the tuned hyperparameters
    for seed in seeds:
        print(f"\n--- Running Seed {seed} for {dataset_name} ---")
        
        # Regenerate dataset split for this seed
        if dataset_name == "german_credit":
            preprocess_german_credit(seed=seed)
            X_train_seed = pd.read_csv("data/processed/X_train_german.csv")
            y_train_seed = pd.read_csv("data/processed/y_train_german.csv").values.ravel()
            X_test_seed = pd.read_csv("data/processed/X_test_german.csv")
            y_test_seed = pd.read_csv("data/processed/y_test_german.csv").values.ravel()
        else:
            preprocess_gmsc(seed=seed)
            X_train_seed = pd.read_csv("data/processed/X_train_gmsc.csv")
            y_train_seed = pd.read_csv("data/processed/y_train_gmsc.csv").values.ravel()
            X_test_seed = pd.read_csv("data/processed/X_test_gmsc.csv")
            y_test_seed = pd.read_csv("data/processed/y_test_gmsc.csv").values.ravel()
            
        # Train model
        model, metrics = train_baseline_model_lgbm(
            X_train_seed, y_train_seed, X_test_seed, y_test_seed,
            best_params, dataset_name, seed
        )
        all_metrics.append(metrics)
        
        # Compute SHAP reference
        compute_shap_reference_lgbm(model, X_test_seed, dataset_name, seed)
        
    # 4. Save aggregated baseline metrics summary
    results_dir = "results/baseline/"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"{dataset_name}_real_lgbm_summary.json")
    
    summary = {
        "dataset": dataset_name,
        "best_parameters": best_params,
        "seeds": seeds,
        "runs": all_metrics
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nBaseline LightGBM Pipeline for {dataset_name} Completed!")
    print(f"Summary written to {summary_path}")

def main():
    print("==========================================================")
    print("Shap Synthetic Credit Risk: Starting Baseline LightGBM Training")
    print("==========================================================\n")
    
    # Run German Credit pipeline
    run_baseline_pipeline("german_credit")
    
    # Run GMSC pipeline if raw training file is available
    if os.path.exists("data/raw/cs-training.csv"):
        run_baseline_pipeline("gmsc")
    else:
        print("\nRaw GMSC file 'cs-training.csv' is missing. Skipping baseline runs for GMSC.")
        
    print("\n==========================================================")
    print("Baseline LightGBM Training Execution Finished Successfully")
    print("==========================================================")

if __name__ == "__main__":
    main()
