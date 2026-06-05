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

def run_synthetic_training_pipeline(dataset_name, generator_type):
    print("\n" + "="*70)
    print(f"Starting Downstream Training on {generator_type.upper()} Data for Dataset: {dataset_name}")
    print("="*70)
    
    seeds = load_json("configs/seeds/seeds.json")["seeds"]
    
    # Load optimal hyperparameters from real baseline training
    baseline_summary_path = f"results/baseline/{dataset_name}_real_xgboost_summary.json"
    if not os.path.exists(baseline_summary_path):
        raise FileNotFoundError(
            f"Baseline summary file {baseline_summary_path} not found. "
            "Please run Phase 2 baseline model training first."
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
        
    all_metrics = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        
        # 1. Regenerate real test split and baseline scaling for this seed
        # This writes X_test_{suffix}.csv and y_test_{suffix}.csv to disk for this seed.
        if dataset_name == "german_credit":
            preprocess_german_credit(seed=seed)
        else:
            preprocess_gmsc(seed=seed)
            
        X_test_real = pd.read_csv(f"data/processed/X_test_{dataset_suffix}.csv")
        y_test_real = pd.read_csv(f"data/processed/y_test_{dataset_suffix}.csv").values.ravel()
        
        # 2. Re-fit real preprocessing on raw train split of this seed
        fitted_transformers = fit_real_preprocessing(dataset_name, seed)
        
        # 3. Load synthetic training split
        syn_csv_path = f"data/synthetic/{dataset_name}_{generator_type}_seed{seed}.csv"
        if not os.path.exists(syn_csv_path):
            print(f"Synthetic file {syn_csv_path} not found. Skipping seed {seed}.")
            continue
        df_synthetic = pd.read_csv(syn_csv_path)
        
        # 4. Preprocess synthetic features using fitted real transformers
        X_train_syn, y_train_syn = preprocess_synthetic_data(
            df_synthetic=df_synthetic,
            dataset_name=dataset_name,
            fitted=fitted_transformers,
            target_column=target_column
        )
        
        # 5. Train XGBoost classifier on preprocessed synthetic data
        # Evaluates on real test split and saves predictions
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
        
    # Save aggregated utility metrics summary
    summary = {
        "dataset": dataset_name,
        "generator_type": generator_type,
        "best_parameters": best_params,
        "seeds": seeds,
        "runs": all_metrics
    }
    
    results_dir = f"results/{generator_type}/"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"{dataset_name}_{generator_type}_xgboost_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nTraining on {generator_type.upper()} synthetic data for {dataset_name} completed!")
    print(f"Summary written to {summary_path}")

def main():
    print("==========================================================")
    print("Shap Synthetic Credit Risk: Starting Synthetic Model Downstream Training")
    print("==========================================================\n")
    
    # Run pipelines for German Credit
    run_synthetic_training_pipeline("german_credit", "ctgan")
    run_synthetic_training_pipeline("german_credit", "tvae")
    
    # Run pipelines for GMSC
    run_synthetic_training_pipeline("gmsc", "ctgan")
    run_synthetic_training_pipeline("gmsc", "tvae")
    
    print("\n==========================================================")
    print("Downstream Model Training Execution Finished Successfully")
    print("==========================================================")

if __name__ == "__main__":
    main()
