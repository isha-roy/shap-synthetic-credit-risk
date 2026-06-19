import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_utility_privacy_scatter():
    print("Generating utility-privacy scatter plots...")
    # Load inference risk
    inf_german = load_json("results/privacy/inference_risk_german.json")
    inf_gmsc = load_json("results/privacy/inference_risk_gmsc.json")
    
    # Load downstream utility runs
    real_g = load_json("results/baseline/german_credit_real_xgboost_summary.json")
    ctgan_g = load_json("results/ctgan/german_credit_ctgan_xgboost_summary.json")
    tvae_g = load_json("results/tvae/german_credit_tvae_xgboost_summary.json")
    
    real_c = load_json("results/baseline/gmsc_real_xgboost_summary.json")
    ctgan_c = load_json("results/ctgan/gmsc_ctgan_xgboost_summary.json")
    tvae_c = load_json("results/tvae/gmsc_tvae_xgboost_summary.json")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    # ------------------ Subplot (a): German Credit ------------------
    ax = axes[0]
    ctgan_auc = [r["roc_auc"] for r in ctgan_g["runs"]]
    tvae_auc = [r["roc_auc"] for r in tvae_g["runs"]]
    
    ctgan_risk = inf_german["ctgan"]["scores_per_seed"]
    tvae_risk = inf_german["tvae"]["scores_per_seed"]
    
    # Scatter points
    ax.scatter(ctgan_auc, ctgan_risk, color='crimson', marker='o', s=80, label='CTGAN', alpha=0.85, edgecolor='black')
    ax.scatter(tvae_auc, tvae_risk, color='royalblue', marker='s', s=80, label='TVAE', alpha=0.85, edgecolor='black')
    
    # Privacy threshold line
    thresh_g = inf_german["ctgan"]["threshold"]
    ax.axhline(y=thresh_g, color='red', linestyle='--', linewidth=2, label=f'Privacy Threshold ({thresh_g:.4f})')
    
    # Real baseline AUC line (mean +/- std)
    real_auc_vals = [r["roc_auc"] for r in real_g["runs"]]
    real_mean = np.mean(real_auc_vals)
    real_std = np.std(real_auc_vals)
    ax.axvline(x=real_mean, color='gray', linestyle='-.', linewidth=2, label=f'Real Baseline AUC ({real_mean:.4f})')
    ax.axvspan(real_mean - real_std, real_mean + real_std, color='gray', alpha=0.15)
    
    ax.set_title("(a) German Credit (XGBoost Utility vs. Privacy)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Downstream ROC-AUC", fontsize=11, fontweight='bold')
    ax.set_ylabel("Inference Risk Score", fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax.set_ylim(-0.02, 0.85)
    
    # ------------------ Subplot (b): GMSC ------------------
    ax = axes[1]
    ctgan_auc_c = [r["roc_auc"] for r in ctgan_c["runs"]]
    tvae_auc_c = [r["roc_auc"] for r in tvae_c["runs"]]
    
    ctgan_risk_c = inf_gmsc["ctgan"]["scores_per_seed"]
    tvae_risk_c = inf_gmsc["tvae"]["scores_per_seed"]
    
    # Scatter points
    ax.scatter(ctgan_auc_c, ctgan_risk_c, color='crimson', marker='o', s=80, label='CTGAN', alpha=0.85, edgecolor='black')
    ax.scatter(tvae_auc_c, tvae_risk_c, color='royalblue', marker='s', s=80, label='TVAE', alpha=0.85, edgecolor='black')
    
    # Privacy threshold line
    thresh_c = inf_gmsc["ctgan"]["threshold"]
    ax.axhline(y=thresh_c, color='red', linestyle='--', linewidth=2, label=f'Privacy Threshold ({thresh_c:.4f})')
    
    # Real baseline AUC line
    real_auc_vals_c = [r["roc_auc"] for r in real_c["runs"]]
    real_mean_c = np.mean(real_auc_vals_c)
    real_std_c = np.std(real_auc_vals_c)
    ax.axvline(x=real_mean_c, color='gray', linestyle='-.', linewidth=2, label=f'Real Baseline AUC ({real_mean_c:.4f})')
    ax.axvspan(real_mean_c - real_std_c, real_mean_c + real_std_c, color='gray', alpha=0.15)
    
    ax.set_title("(b) GMSC (XGBoost Utility vs. Privacy)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Downstream ROC-AUC", fontsize=11, fontweight='bold')
    ax.set_ylabel("Inference Risk Score", fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax.set_ylim(-0.02, 0.85)
    
    plt.suptitle("Utility-Privacy Tradeoff Scatter Plots Across 10 Seeds", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = "figures/paper/utility_privacy_scatter.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Utility-privacy scatter plot successfully saved to {output_path}")

def generate_shap_consistency_boxplots():
    print("Generating SHAP consistency boxplots...")
    # Load Spearman rho values across seeds
    g_shap = load_json("results/shap/german_credit_shap_consistency_summary.json")
    c_shap = load_json("results/shap/gmsc_shap_consistency_summary.json")
    
    g_ctgan_rho = [r["spearman_rho"] for r in g_shap["ctgan_runs"]]
    g_tvae_rho = [r["spearman_rho"] for r in g_shap["tvae_runs"]]
    
    c_ctgan_rho = [r["spearman_rho"] for r in c_shap["ctgan_runs"]]
    c_tvae_rho = [r["spearman_rho"] for r in c_shap["tvae_runs"]]
    
    # Format into DataFrame for plotting
    data = []
    for rho in g_ctgan_rho:
        data.append({"Dataset": "German Credit", "Generator": "CTGAN", "Spearman Rho": rho})
    for rho in g_tvae_rho:
        data.append({"Dataset": "German Credit", "Generator": "TVAE", "Spearman Rho": rho})
    for rho in c_ctgan_rho:
        data.append({"Dataset": "GMSC", "Generator": "CTGAN", "Spearman Rho": rho})
    for rho in c_tvae_rho:
        data.append({"Dataset": "GMSC", "Generator": "TVAE", "Spearman Rho": rho})
        
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    
    # Boxplot
    ax = sns.boxplot(
        x="Dataset", y="Spearman Rho", hue="Generator", data=df,
        palette={"CTGAN": "crimson", "TVAE": "royalblue"},
        width=0.5, fliersize=0, boxprops=dict(alpha=0.7, edgecolor='black'),
        medianprops=dict(color='black', linewidth=1.5)
    )
    
    # Overlaid stripplot for individual runs (seeds)
    sns.stripplot(
        x="Dataset", y="Spearman Rho", hue="Generator", data=df,
        dodge=True, marker='o', size=7, color='black', linewidth=1,
        edgecolor='gray', alpha=0.8, legend=False
    )
    
    # Target lines
    plt.axhline(y=0.7, color='green', linestyle=':', linewidth=1.5, label='High Consistency (>= 0.7)')
    plt.axhline(y=0.4, color='orange', linestyle=':', linewidth=1.5, label='Moderate Consistency (>= 0.4)')
    
    plt.title("SHAP Explanation Consistency (Spearman Rank Correlation rho) Across 10 Seeds", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Spearman Rank Correlation (Rho)", fontsize=11, fontweight='bold')
    plt.xlabel("Dataset", fontsize=11, fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    
    output_path = "figures/paper/shap_consistency_boxplots.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"SHAP consistency boxplot successfully saved to {output_path}")

def generate_mia_roc_curves():
    print("Generating Membership Inference Attack ROC curves...")
    # Load predictions/distances for German Credit
    from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc
    from src.models.synthetic import fit_real_preprocessing, preprocess_synthetic_data
    from scipy.spatial.distance import cdist
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    datasets = [
        ("german_credit", "german", "class", axes[0], "(a) German Credit MIA ROC curves"),
        ("gmsc", "gmsc", "SeriousDlqin2yrs", axes[1], "(b) GMSC MIA ROC curves")
    ]
    
    for ds_name, suffix, target_col, ax, title in datasets:
        if ds_name == "german_credit":
            preprocess_german_credit(seed=42)
        else:
            preprocess_gmsc(seed=42)
            
        X_train_real = pd.read_csv(f"data/processed/X_train_{suffix}.csv")
        X_test_real = pd.read_csv(f"data/processed/X_test_{suffix}.csv")
        
        generators = [
            ('CTGAN', f"data/synthetic/{ds_name}_ctgan_seed42.csv", 'crimson'),
            ('TVAE', f"data/synthetic/{ds_name}_tvae_seed42.csv", 'royalblue')
        ]
        
        for label, syn_path, color in generators:
            if os.path.exists(syn_path):
                df_syn = pd.read_csv(syn_path)
                fitted = fit_real_preprocessing(ds_name, seed=42)
                X_syn_proc, _ = preprocess_synthetic_data(df_syn, ds_name, fitted, target_col)
                
                dists_train = cdist(X_train_real, X_syn_proc, metric='euclidean')
                d_train = dists_train.min(axis=1)
                
                dists_test = cdist(X_test_real, X_syn_proc, metric='euclidean')
                d_test = dists_test.min(axis=1)
                
                scores = np.concatenate([-d_train, -d_test])
                labels = np.concatenate([np.ones(len(d_train)), np.zeros(len(d_test))])
                
                fpr, tpr, _ = roc_curve(labels, scores)
                auc = roc_auc_score(labels, scores)
                
                ax.plot(fpr, tpr, label=f"{label} (MIA AUC = {auc:.4f})", color=color, linewidth=2.5)
                
        ax.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=10)
        
    plt.suptitle("Distance-based Membership Inference Attack (MIA) ROC Curves (Seed 42)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = "figures/paper/mia_roc_curves.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"MIA ROC curves successfully saved to {output_path}")

def generate_roc_curves_combined():
    print("Generating combined 4-panel ROC curve figure...")
    from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc
    from sklearn.metrics import roc_curve, roc_auc_score
    import joblib
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.set_theme(style="whitegrid")
    
    # Seed 42 setups
    preprocess_german_credit(seed=42)
    preprocess_gmsc(seed=42)
    
    # ------------------ (a) German Credit Downstream XGBoost ROC ------------------
    ax = axes[0, 0]
    X_test_g = pd.read_csv("data/processed/X_test_german.csv")
    y_test_g = pd.read_csv("data/processed/y_test_german.csv").values.ravel()
    y_test_g = np.where(y_test_g == 'bad', 1, 0) if y_test_g.dtype == object else y_test_g.astype(int)
    
    g_xgb_models = [
        ('Real Baseline', "models/baseline/german_credit_real_xgboost_seed42_v1.joblib", 'darkgray', '--'),
        ('CTGAN Synthetic', "models/ctgan/german_credit_ctgan_xgboost_seed42_v1.joblib", 'crimson', '-'),
        ('TVAE Synthetic', "models/tvae/german_credit_tvae_xgboost_seed42_v1.joblib", 'royalblue', '-')
    ]
    for label, m_path, color, style in g_xgb_models:
        if os.path.exists(m_path):
            model = joblib.load(m_path)
            y_prob = model.predict_proba(X_test_g)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_g, y_prob)
            auc = roc_auc_score(y_test_g, y_prob)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})", color=color, linestyle=style, linewidth=2.5)
            
    ax.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5)
    ax.set_title("(a) German Credit: Downstream XGBoost Utility", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=9)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    
    # ------------------ (b) German Credit Downstream LightGBM ROC ------------------
    ax = axes[0, 1]
    g_lgbm_models = [
        ('Real Baseline', "models/baseline/german_credit_real_lgbm_seed42_v1.joblib", 'darkgray', '--'),
        ('CTGAN Synthetic', "models/ctgan/german_credit_ctgan_lgbm_seed42_v1.joblib", 'crimson', '-'),
        ('TVAE Synthetic', "models/tvae/german_credit_tvae_lgbm_seed42_v1.joblib", 'royalblue', '-')
    ]
    for label, m_path, color, style in g_lgbm_models:
        if os.path.exists(m_path):
            model = joblib.load(m_path)
            y_prob = model.predict_proba(X_test_g)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_g, y_prob)
            auc = roc_auc_score(y_test_g, y_prob)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})", color=color, linestyle=style, linewidth=2.5)
            
    ax.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5)
    ax.set_title("(b) German Credit: Downstream LightGBM Utility", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=9)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    
    # ------------------ (c) GMSC Downstream XGBoost ROC ------------------
    ax = axes[1, 0]
    X_test_c = pd.read_csv("data/processed/X_test_gmsc.csv")
    y_test_c = pd.read_csv("data/processed/y_test_gmsc.csv").values.ravel().astype(int)
    
    c_xgb_models = [
        ('Real Baseline', "models/baseline/gmsc_real_xgboost_seed42_v1.joblib", 'darkgray', '--'),
        ('CTGAN Synthetic', "models/ctgan/gmsc_ctgan_xgboost_seed42_v1.joblib", 'crimson', '-'),
        ('TVAE Synthetic', "models/tvae/gmsc_tvae_xgboost_seed42_v1.joblib", 'royalblue', '-')
    ]
    for label, m_path, color, style in c_xgb_models:
        if os.path.exists(m_path):
            model = joblib.load(m_path)
            y_prob = model.predict_proba(X_test_c)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_c, y_prob)
            auc = roc_auc_score(y_test_c, y_prob)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})", color=color, linestyle=style, linewidth=2.5)
            
    ax.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5)
    ax.set_title("(c) GMSC: Downstream XGBoost Utility", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=9)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    
    # ------------------ (d) GMSC Downstream LightGBM ROC ------------------
    ax = axes[1, 1]
    c_lgbm_models = [
        ('Real Baseline', "models/baseline/gmsc_real_lgbm_seed42_v1.joblib", 'darkgray', '--'),
        ('CTGAN Synthetic', "models/ctgan/gmsc_ctgan_lgbm_seed42_v1.joblib", 'crimson', '-'),
        ('TVAE Synthetic', "models/tvae/gmsc_tvae_lgbm_seed42_v1.joblib", 'royalblue', '-')
    ]
    for label, m_path, color, style in c_lgbm_models:
        if os.path.exists(m_path):
            model = joblib.load(m_path)
            y_prob = model.predict_proba(X_test_c)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_c, y_prob)
            auc = roc_auc_score(y_test_c, y_prob)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})", color=color, linestyle=style, linewidth=2.5)
            
    ax.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5)
    ax.set_title("(d) GMSC: Downstream LightGBM Utility", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=9)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.suptitle("Combined Downstream Model Utility ROC Curves (Seed 42)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = "figures/paper/roc_curves_combined.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Combined ROC curve plot successfully saved to {output_path}")

def generate_shap_beeswarm_combined():
    print("Generating combined 6-panel SHAP beeswarm figure...")
    # Open all individual beeswarm images
    gc_real = Image.open("figures/paper/german_credit_real_shap_beeswarm.png")
    gc_ctgan = Image.open("figures/paper/german_credit_ctgan_shap_beeswarm.png")
    gc_tvae = Image.open("figures/paper/german_credit_tvae_shap_beeswarm.png")
    
    gmsc_real = Image.open("figures/paper/gmsc_real_shap_beeswarm.png")
    gmsc_ctgan = Image.open("figures/paper/gmsc_ctgan_shap_beeswarm.png")
    gmsc_tvae = Image.open("figures/paper/gmsc_tvae_shap_beeswarm.png")
    
    # Ensure they have uniform sizing
    w1, h1 = gc_real.size
    
    # We will build a 2x3 grid canvas
    # Let's crop margins if necessary, but pasting them directly is cleanest
    canvas_w = w1 * 3
    canvas_h = h1 * 2
    
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    
    # Row 1: German Credit
    canvas.paste(gc_real.resize((w1, h1)), (0, 0))
    canvas.paste(gc_ctgan.resize((w1, h1)), (w1, 0))
    canvas.paste(gc_tvae.resize((w1, h1)), (w1 * 2, 0))
    
    # Row 2: GMSC
    canvas.paste(gmsc_real.resize((w1, h1)), (0, h1))
    canvas.paste(gmsc_ctgan.resize((w1, h1)), (w1, h1))
    canvas.paste(gmsc_tvae.resize((w1, h1)), (w1 * 2, h1))
    
    # Add panel text overlays
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    
    # Draw simple annotations
    # Panel coordinates
    labels = [
        ("(a) German Credit Real", 15, 15),
        ("(b) German Credit CTGAN", w1 + 15, 15),
        ("(c) German Credit TVAE", w1 * 2 + 15, 15),
        ("(d) GMSC Real", 15, h1 + 15),
        ("(e) GMSC CTGAN", w1 + 15, h1 + 15),
        ("(f) GMSC TVAE", w1 * 2 + 15, h1 + 15)
    ]
    
    for text, x, y in labels:
        draw.text((x, y), text, fill=(0, 0, 0))
        
    output_path = "figures/paper/shap_beeswarm_combined.png"
    canvas.save(output_path, dpi=(300, 300))
    print(f"Combined beeswarm plot successfully saved to {output_path}")

def copy_heatmaps():
    print("Copying and renaming SHAP heatmaps...")
    src_g = "figures/paper/german_credit_shap_heatmap.png"
    src_c = "figures/paper/gmsc_shap_heatmap.png"
    
    dst_g = "figures/paper/shap_heatmap_german.png"
    dst_c = "figures/paper/gmsc.png"
    
    shutil.copyfile(src_g, dst_g)
    shutil.copyfile(src_c, dst_c)
    print(f"Copied heatmap to {dst_g}")
    print(f"Copied heatmap to {dst_c}")

def main():
    os.makedirs("figures/paper", exist_ok=True)
    generate_roc_curves_combined()
    generate_shap_beeswarm_combined()
    generate_shap_consistency_boxplots()
    generate_mia_roc_curves()
    generate_utility_privacy_scatter()
    copy_heatmaps()
    print("\nAll missing combined visualizations successfully generated!")

if __name__ == "__main__":
    main()
