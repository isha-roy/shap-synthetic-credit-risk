import torch
import os
import sys
import json
import pandas as pd
import numpy as np

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc
from src.models.synthetic import fit_real_preprocessing, preprocess_synthetic_data, train_synthetic_model

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_dp_training_pipeline(dataset_name, epsilon_values):
    print("\n" + "="*80)
    print(f"Starting Downstream Training on DP-TVAE Data for Dataset: {dataset_name}")
    print("="*80)
    
    seeds = load_json("configs/seeds/seeds.json")["seeds"]
    
    # Load optimal hyperparameters from real baseline training
    baseline_summary_path = f"results/baseline/{dataset_name}_real_xgboost_summary.json"
    if not os.path.exists(baseline_summary_path):
        raise FileNotFoundError(
            f"Baseline summary file {baseline_summary_path} not found. "
            "Please run baseline model training first."
        )
    baseline_summary = load_json(baseline_summary_path)
    best_params = baseline_summary["best_parameters"]
    print(f"Loaded optimal hyperparameters from real baseline: {best_params}")
    
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
        
    for eps in epsilon_values:
        generator_type = f"tvae_dp_eps{eps}"
        print(f"\n--- Running Epsilon {eps} ---")
        
        all_metrics = []
        
        for seed in seeds:
            # Check if synthetic file exists first
            syn_csv_path = f"data/synthetic/{dataset_name}_tvae_dp_eps{eps}_seed{seed}.csv"
            if not os.path.exists(syn_csv_path):
                print(f"Synthetic file {syn_csv_path} not found. Skipping seed {seed} for eps {eps}.")
                continue
                
            print(f"Processing Seed {seed}...")
            
            # 1. Regenerate real test split and baseline scaling for this seed
            if dataset_name == "german_credit":
                preprocess_german_credit(seed=seed)
            else:
                preprocess_gmsc(seed=seed)
                
            X_test_real = pd.read_csv(f"data/processed/X_test_{dataset_suffix}.csv")
            y_test_real = pd.read_csv(f"data/processed/y_test_{dataset_suffix}.csv").values.ravel()
            
            # 2. Re-fit real preprocessing on raw train split of this seed
            fitted_transformers = fit_real_preprocessing(dataset_name, seed)
            
            # 3. Load synthetic training split
            df_synthetic = pd.read_csv(syn_csv_path)
            
            # 4. Preprocess synthetic features using fitted real transformers
            X_train_syn, y_train_syn = preprocess_synthetic_data(
                df_synthetic=df_synthetic,
                dataset_name=dataset_name,
                fitted=fitted_transformers,
                target_column=target_column
            )
            
            # 5. Train XGBoost classifier on preprocessed synthetic data
            _, metrics = train_synthetic_model(
                X_train_syn=X_train_syn,
                y_train_syn=y_train_syn,
                X_test_real=X_test_real,
                y_test_real=y_test_real,
                params=best_params,
                dataset_name=dataset_name,
                generator_type=generator_type,
                seed=seed
            )
            all_metrics.append(metrics)
            
        if not all_metrics:
            print(f"No successful runs for Epsilon {eps}. Skipping summary.")
            continue
            
        # Save aggregated utility metrics summary
        results_dir = f"results/{generator_type}/"
        os.makedirs(results_dir, exist_ok=True)
        summary_path = os.path.join(results_dir, f"{dataset_name}_{generator_type}_xgboost_summary.json")
        
        summary = {
            "dataset": dataset_name,
            "generator_type": generator_type,
            "best_parameters": best_params,
            "seeds": [run["seed"] for run in all_metrics],
            "runs": all_metrics
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Aggregated summary written to {summary_path}")

def main():
    print("==========================================================")
    print("Shap Synthetic Credit Risk: Starting DP-TVAE Downstream Training")
    print("==========================================================\n")
    
    # Load epsilon values from config
    dp_config = load_json("configs/experiments/tvae_dp_default.json")
    epsilon_values = dp_config["dp_parameters"]["epsilon_values"]
    
    # Run pipelines for German Credit
    run_dp_training_pipeline("german_credit", epsilon_values)
    
    # Run pipelines for GMSC
    run_dp_training_pipeline("gmsc", epsilon_values)
    
    print("\n==========================================================")
    print("DP Downstream Model Training Execution Finished Successfully")
    print("==========================================================")

if __name__ == "__main__":
    main()
