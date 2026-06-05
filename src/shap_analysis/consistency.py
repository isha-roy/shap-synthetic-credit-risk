import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr

def compute_model_shap(
    model, X_test, dataset_name, generator_type, seed, 
    results_dir="results/shap/", figures_dir="figures/shap/"
):
    """
    Computes test set SHAP attributions using TreeExplainer for a synthetic-trained model,
    saves the raw values and rankings, and generates Beeswarm and Bar plots.
    """
    print(f"Computing SHAP values for {dataset_name} ({generator_type.upper()}, Seed {seed})...")
    
    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle SHAP multi-class format discrepancies in shap>=0.45.0
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    elif len(shap_values.shape) == 3:
        # (n_samples, n_features, n_classes)
        shap_values_class1 = shap_values[:, :, 1]
    else:
        # Standard 2D array
        shap_values_class1 = shap_values
        
    # Create target directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Save raw SHAP values as numpy binary
    shap_val_path = os.path.join(results_dir, f"{dataset_name}_{generator_type}_xgboost_seed{seed}_v1_shap_values.npy")
    np.save(shap_val_path, shap_values_class1)
    
    # 2. Compute mean absolute SHAP attributions and save ranking
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    features = X_test.columns.tolist()
    
    ranking_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)
    
    ranking_path = os.path.join(results_dir, f"{dataset_name}_{generator_type}_xgboost_seed{seed}_v1_shap_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)
    
    # 3. Save feature rank positions
    rank_pos_df = ranking_df.copy()
    rank_pos_df["rank_position"] = np.arange(1, len(features) + 1)
    # Sort alphabetically by feature name for alignment
    rank_pos_df = rank_pos_df.sort_values(by="feature")
    
    rank_pos_path = os.path.join(results_dir, f"{dataset_name}_{generator_type}_xgboost_seed{seed}_v1_shap_rank_position.csv")
    rank_pos_df.to_csv(rank_pos_path, index=False)
    
    # 4. Generate and save SHAP Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_test, show=False)
    beeswarm_path = os.path.join(figures_dir, f"{dataset_name}_{generator_type}_xgboost_seed{seed}_v1_shap_beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches='tight')
    plt.close()
    
    # 5. Generate and save SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_test, plot_type="bar", show=False)
    bar_path = os.path.join(figures_dir, f"{dataset_name}_{generator_type}_xgboost_seed{seed}_v1_shap_bar.png")
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP calculations and plots completed for {dataset_name} ({generator_type.upper()}, Seed {seed})")
    return shap_values_class1

def evaluate_ranking_similarity(dataset_name, generator_type, seed, results_dir="results/shap/"):
    """
    Loads real baseline and synthetic rank position files, sorts them,
    and calculates Spearman rank correlation (rho, p-value) and top-K feature overlaps.
    """
    real_path = os.path.join(results_dir, f"{dataset_name}_real_xgboost_seed{seed}_v1_shap_rank_position.csv")
    syn_path = os.path.join(results_dir, f"{dataset_name}_{generator_type}_xgboost_seed{seed}_v1_shap_rank_position.csv")
    
    if not os.path.exists(real_path) or not os.path.exists(syn_path):
        raise FileNotFoundError(f"SHAP rank position files not found. Real: {real_path}, Syn: {syn_path}")
        
    df_real = pd.read_csv(real_path)
    df_syn = pd.read_csv(syn_path)
    
    # Sort alphabetically by feature name to ensure alignment
    df_real_sorted = df_real.sort_values(by="feature").reset_index(drop=True)
    df_syn_sorted = df_syn.sort_values(by="feature").reset_index(drop=True)
    
    assert (df_real_sorted["feature"] == df_syn_sorted["feature"]).all(), "Feature names mismatch between real and synthetic ranking files."
    
    # Calculate Spearman correlation coefficient and p-value on mean absolute SHAP values
    coef, p_val = spearmanr(df_real_sorted["mean_abs_shap"], df_syn_sorted["mean_abs_shap"])
    
    # Handle NaN values if they arise due to constant zero rankings
    if np.isnan(coef):
        coef = 0.0
        
    # Get top 5 features
    top5_real_list = df_real.sort_values(by="mean_abs_shap", ascending=False).head(5)["feature"].tolist()
    top5_syn_list = df_syn.sort_values(by="mean_abs_shap", ascending=False).head(5)["feature"].tolist()
    
    top5_real = set(top5_real_list)
    top5_syn = set(top5_syn_list)
    
    top5_overlap = len(top5_real.intersection(top5_syn))
    top5_jaccard = top5_overlap / len(top5_real.union(top5_syn)) if len(top5_real.union(top5_syn)) > 0 else 0.0
    
    # Get top 10 features
    top10_real_list = df_real.sort_values(by="mean_abs_shap", ascending=False).head(10)["feature"].tolist()
    top10_syn_list = df_syn.sort_values(by="mean_abs_shap", ascending=False).head(10)["feature"].tolist()
    
    top10_real = set(top10_real_list)
    top10_syn = set(top10_syn_list)
    
    top10_overlap = len(top10_real.intersection(top10_syn))
    top10_jaccard = top10_overlap / len(top10_real.union(top10_syn)) if len(top10_real.union(top10_syn)) > 0 else 0.0
    
    return {
        "dataset": dataset_name,
        "generator_type": generator_type,
        "seed": seed,
        "spearman_rho": float(coef),
        "spearman_pvalue": float(p_val),
        "top5_overlap_count": int(top5_overlap),
        "top5_jaccard": float(top5_jaccard),
        "top5_real_features": top5_real_list,
        "top5_syn_features": top5_syn_list,
        "top10_overlap_count": int(top10_overlap),
        "top10_jaccard": float(top10_jaccard),
        "top10_real_features": top10_real_list,
        "top10_syn_features": top10_syn_list
    }
