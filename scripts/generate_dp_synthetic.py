import torch
import os
import sys
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.synthetic.generation import get_sdv_metadata, set_seed, generate_synthetic_data
from src.synthetic.dp_tvae import DPTVAESynthesizer

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_dp_synthetic_pipeline(dataset_name):
    print("\n" + "="*60)
    print(f"Starting DP Synthetic Generation for Dataset: {dataset_name}")
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
        print(f"Raw file {raw_path} not found. Skipping {dataset_name} DP generation.")
        return
        
    df = pd.read_csv(raw_path)
    
    # Drop index columns if present in GMSC to avoid modeling row indices
    if dataset_name == "gmsc" and "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    # Load default DP experiment params
    dp_config = load_json("configs/experiments/tvae_dp_default.json")
    params = dp_config["parameters"].copy()
    dp_params = dp_config["dp_parameters"]
    
    # Set up output directories
    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("models/tvae_dp", exist_ok=True)
    
    for seed in seeds:
        print(f"\n--- Processing Seed {seed} for {dataset_name} ---")
        
        # Reconstruct the exact raw training split for this seed
        train_df, _ = train_test_split(
            df,
            test_size=0.3,
            stratify=df[target_column],
            random_state=seed
        )
        
        # Extract and validate metadata
        numerical_features = ds_config.get("numerical_features", [])
        categorical_features = ds_config.get("categorical_features", [])
        metadata = get_sdv_metadata(train_df, dataset_name, numerical_features, categorical_features, target_column)
        
        # Clean/convert parameters
        tvae_keys = [
            'epochs', 'batch_size', 'compress_dims', 'decompress_dims', 
            'embedding_dim', 'l2scale', 'loss_factor', 
            'enforce_min_max_values', 'enforce_rounding', 'cuda'
        ]
        tvae_params = {k: v for k, v in params.items() if k in tvae_keys}
        if 'compress_dims' in tvae_params and isinstance(tvae_params['compress_dims'], list):
            tvae_params['compress_dims'] = tuple(tvae_params['compress_dims'])
        if 'decompress_dims' in tvae_params and isinstance(tvae_params['decompress_dims'], list):
            tvae_params['decompress_dims'] = tuple(tvae_params['decompress_dims'])
            
        for eps in dp_params["epsilon_values"]:
            csv_path = f"data/synthetic/{dataset_name}_tvae_dp_eps{eps}_seed{seed}.csv"
            model_path = f"models/tvae_dp/{dataset_name}_tvae_dp_eps{eps}_seed{seed}.pkl"
            
            if os.path.exists(csv_path):
                print(f"  -> Epsilon {eps}: Already exists. Skipping.")
                continue
                
            print(f"  -> Training DP-TVAE with epsilon={eps}, delta={dp_params['delta']}...")
            
            # Set seed for training reproducibility
            set_seed(seed)
            
            start_time = time.time()
            synthesizer = DPTVAESynthesizer(
                target_epsilon=eps,
                target_delta=dp_params["delta"],
                max_grad_norm=dp_params["max_grad_norm"],
                metadata=metadata,
                **tvae_params
            )
            
            # Fit model
            synthesizer.fit(train_df)
            fit_time = time.time() - start_time
            print(f"     Fit completed in {fit_time:.2f} seconds.")
            
            # Sample synthetic data
            synthetic_df = generate_synthetic_data(synthesizer, len(train_df))
            synthetic_df.to_csv(csv_path, index=False)
            print(f"     Saved synthetic data to {csv_path}")
            
            # Save synthesizer model (skipped to avoid cloudpickle MemoryError)
            # synthesizer.save(model_path)
            # print(f"     Saved DP-TVAE synthesizer model to {model_path}")
            pass

def main():
    print("==========================================================")
    print("Shap Synthetic Credit Risk: Starting DP-TVAE Data Generation")
    print("==========================================================\n")
    
    overall_start = time.time()
    
    # Run German Credit generation
    run_dp_synthetic_pipeline("german_credit")
    
    # Run GMSC generation
    run_dp_synthetic_pipeline("gmsc")
    
    print("\n==========================================================")
    print(f"DP Synthetic Generation Finished. Total time: {time.time() - overall_start:.2f}s")
    print("==========================================================")

if __name__ == "__main__":
    main()
