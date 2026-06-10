"""
generate_missing_gmsc.py
Generates synthetic data for specific missing GMSC seeds.
Runs each seed in an isolated subprocess to avoid memory fragmentation.
"""
import os
import sys
import json
import time
import subprocess

MISSING_SEEDS = [99, 314, 2718]
DATASET = "gmsc"

def generate_one_seed(seed):
    """Run synthetic generation for a single seed via a helper script."""
    script = f"""
import os
import sys
import gc
import json
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.synthetic.generation import get_sdv_metadata, train_generator, generate_synthetic_data

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

seed = {seed}
dataset_name = "{DATASET}"
raw_path = "data/raw/gmsc_sampled_10k.csv"

print(f"\\n=== Seed {{seed}} for {{dataset_name}} ===")
gc.collect()

ds_config = load_json("configs/datasets/gmsc.json")
target_column = ds_config["target_column"]

df = pd.read_csv(raw_path)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

train_df, _ = train_test_split(df, test_size=0.3, stratify=df[target_column], random_state=seed)
print(f"Training split: {{train_df.shape}}")

numerical_features = ds_config.get("numerical_features", [])
categorical_features = ds_config.get("categorical_features", [])
metadata = get_sdv_metadata(train_df, dataset_name, numerical_features, categorical_features, target_column)

ctgan_params = load_json("configs/experiments/ctgan_default.json")["parameters"]
tvae_params = load_json("configs/experiments/tvae_default.json")["parameters"]

os.makedirs("data/synthetic", exist_ok=True)
os.makedirs("models/ctgan", exist_ok=True)
os.makedirs("models/tvae", exist_ok=True)

# --- CTGAN ---
ctgan_path = f"data/synthetic/{{dataset_name}}_ctgan_seed{{seed}}.csv"
if not os.path.exists(ctgan_path):
    print("Training CTGAN...")
    ctgan_model = train_generator(train_df, metadata, "ctgan", ctgan_params, seed)
    synth_ctgan = generate_synthetic_data(ctgan_model, len(train_df))
    synth_ctgan.to_csv(ctgan_path, index=False)
    ctgan_model.save(f"models/ctgan/{{dataset_name}}_ctgan_seed{{seed}}_v1.pkl")
    print(f"Saved: {{ctgan_path}}")
    del ctgan_model, synth_ctgan
    gc.collect()
else:
    print(f"CTGAN already exists: {{ctgan_path}}")

# --- TVAE ---
tvae_path = f"data/synthetic/{{dataset_name}}_tvae_seed{{seed}}.csv"
if not os.path.exists(tvae_path):
    print("Training TVAE...")
    tvae_model = train_generator(train_df, metadata, "tvae", tvae_params, seed)
    synth_tvae = generate_synthetic_data(tvae_model, len(train_df))
    synth_tvae.to_csv(tvae_path, index=False)
    tvae_model.save(f"models/tvae/{{dataset_name}}_tvae_seed{{seed}}_v1.pkl")
    print(f"Saved: {{tvae_path}}")
    del tvae_model, synth_tvae
    gc.collect()
else:
    print(f"TVAE already exists: {{tvae_path}}")

print(f"Seed {{seed}} complete.")
"""
    # Write the inline script to a temp file
    tmp_path = f"scripts/_tmp_gen_seed_{seed}.py"
    with open(tmp_path, "w") as f:
        f.write(script)
    
    result = subprocess.run(
        [sys.executable, tmp_path],
        capture_output=False,
        text=True
    )
    
    # Clean up temp file
    try:
        os.remove(tmp_path)
    except:
        pass
    
    return result.returncode == 0

def main():
    print("=" * 60)
    print(f"Generating missing GMSC seeds: {MISSING_SEEDS}")
    print("=" * 60)
    
    overall_start = time.time()
    
    for seed in MISSING_SEEDS:
        ctgan_path = f"data/synthetic/{DATASET}_ctgan_seed{seed}.csv"
        tvae_path = f"data/synthetic/{DATASET}_tvae_seed{seed}.csv"
        
        if os.path.exists(ctgan_path) and os.path.exists(tvae_path):
            print(f"\nSeed {seed}: both files exist, skipping.")
            continue
        
        print(f"\nProcessing seed {seed} in isolated subprocess...")
        start = time.time()
        success = generate_one_seed(seed)
        elapsed = time.time() - start
        
        if success:
            print(f"Seed {seed} done in {elapsed:.1f}s [OK]")
        else:
            print(f"Seed {seed} FAILED after {elapsed:.1f}s [FAIL]")
    
    print(f"\nTotal time: {time.time() - overall_start:.1f}s")
    print("Done.")

if __name__ == "__main__":
    main()
