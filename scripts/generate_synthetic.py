import os
import sys
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.synthetic.generation import get_sdv_metadata, train_generator, generate_synthetic_data

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_synthetic_pipeline(dataset_name):
    print("\n" + "="*60)
    print(f"Starting Synthetic Generation for Dataset: {dataset_name}")
    print("="*60)
    
    # Load seeds and configurations
    seeds = load_json("configs/seeds/seeds.json")["seeds"]
    
    # Load dataset-specific config to get raw file path and target column
    if dataset_name == "german_credit":
        ds_config = load_json("configs/datasets/german_credit.json")
        raw_path = "data/raw/german_credit.csv"
        target_column = ds_config["target_column"]
    elif dataset_name == "gmsc":
        ds_config = load_json("configs/datasets/gmsc.json")
        raw_path = "data/raw/gmsc_sampled_10k.csv"
        target_column = ds_config["target_column"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    if not os.path.exists(raw_path):
        print(f"Raw file {raw_path} not found. Skipping {dataset_name} synthetic generation.")
        return
        
    df = pd.read_csv(raw_path)
    
    # Drop index columns if present in GMSC to avoid modeling row indices
    if dataset_name == "gmsc" and "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    # Load default experiment params
    ctgan_params = load_json("configs/experiments/ctgan_default.json")["parameters"]
    tvae_params = load_json("configs/experiments/tvae_default.json")["parameters"]
    
    # Set up output directories
    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("models/ctgan", exist_ok=True)
    os.makedirs("models/tvae", exist_ok=True)
    
    for seed in seeds:
        print(f"\n--- Processing Seed {seed} for {dataset_name} ---")
        
        # 1. Reconstruct the exact raw training split for this seed
        train_df, _ = train_test_split(
            df,
            test_size=0.3,
            stratify=df[target_column],
            random_state=seed
        )
        print(f"Training split reconstructed. Shape: {train_df.shape}")
        
        # 2. Extract and validate metadata
        numerical_features = ds_config.get("numerical_features", [])
        categorical_features = ds_config.get("categorical_features", [])
        metadata = get_sdv_metadata(train_df, dataset_name, numerical_features, categorical_features, target_column)
        
        # 3. Train CTGAN and generate synthetic data
        start_time = time.time()
        ctgan_model = train_generator(
            train_df=train_df,
            metadata=metadata,
            generator_type="ctgan",
            config_params=ctgan_params,
            seed=seed
        )
        ctgan_fit_time = time.time() - start_time
        print(f"CTGAN fit completed in {ctgan_fit_time:.2f} seconds.")
        
        synthetic_ctgan = generate_synthetic_data(ctgan_model, len(train_df))
        ctgan_csv_path = f"data/synthetic/{dataset_name}_ctgan_seed{seed}.csv"
        synthetic_ctgan.to_csv(ctgan_csv_path, index=False)
        print(f"Saved CTGAN synthetic data to {ctgan_csv_path}")
        
        ctgan_model_path = f"models/ctgan/{dataset_name}_ctgan_seed{seed}_v1.pkl"
        ctgan_model.save(ctgan_model_path)
        print(f"Saved CTGAN synthesizer model to {ctgan_model_path}")
        
        # 4. Train TVAE and generate synthetic data
        start_time = time.time()
        tvae_model = train_generator(
            train_df=train_df,
            metadata=metadata,
            generator_type="tvae",
            config_params=tvae_params,
            seed=seed
        )
        tvae_fit_time = time.time() - start_time
        print(f"TVAE fit completed in {tvae_fit_time:.2f} seconds.")
        
        synthetic_tvae = generate_synthetic_data(tvae_model, len(train_df))
        tvae_csv_path = f"data/synthetic/{dataset_name}_tvae_seed{seed}.csv"
        synthetic_tvae.to_csv(tvae_csv_path, index=False)
        print(f"Saved TVAE synthetic data to {tvae_csv_path}")
        
        tvae_model_path = f"models/tvae/{dataset_name}_tvae_seed{seed}_v1.pkl"
        tvae_model.save(tvae_model_path)
        print(f"Saved TVAE synthesizer model to {tvae_model_path}")

def main():
    print("==========================================================")
    print("Shap Synthetic Credit Risk: Starting Synthetic Data Generation")
    print("==========================================================\n")
    
    overall_start = time.time()
    
    # Run German Credit generation
    run_synthetic_pipeline("german_credit")
    
    # Run GMSC generation
    run_synthetic_pipeline("gmsc")
    
    print("\n==========================================================")
    print(f"Synthetic Generation Finished. Total time: {time.time() - overall_start:.2f}s")
    print("==========================================================")

if __name__ == "__main__":
    main()
