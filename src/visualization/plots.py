import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.spatial.distance import cdist

# Import preprocessing pipelines
from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc

def plot_utility_privacy_tradeoff(summary_stats, dataset_name, output_path):
    """
    Generates a scatter plot mapping Downstream ROC-AUC against Mean DCR with error bars.
    Draws the Real baseline as a horizontal line with shaded standard deviation.
    Saves the plot in both PNG and PDF formats.
    """
    stats = summary_stats['metrics_summary']
    
    real_auc_mean = stats['real_auc']['mean']
    real_auc_std = stats['real_auc']['std']
    
    ctgan_auc_mean = stats['ctgan_auc']['mean']
    ctgan_auc_std = stats['ctgan_auc']['std']
    ctgan_dcr_mean = stats['ctgan_dcr']['mean']
    ctgan_dcr_std = stats['ctgan_dcr']['std']
    
    tvae_auc_mean = stats['tvae_auc']['mean']
    tvae_auc_std = stats['tvae_auc']['std']
    tvae_dcr_mean = stats['tvae_dcr']['mean']
    tvae_dcr_std = stats['tvae_dcr']['std']
    
    plt.figure(figsize=(8, 6))
    
    # 1. Plot Real Baseline AUC horizontal line and standard deviation band
    plt.axhline(y=real_auc_mean, color='darkgray', linestyle='--', linewidth=2, label='Real Baseline')
    plt.axhspan(real_auc_mean - real_auc_std, real_auc_mean + real_auc_std, color='darkgray', alpha=0.15)
    
    # 2. Plot CTGAN and TVAE tradeoff points with error bars (crosshairs)
    plt.errorbar(
        x=ctgan_dcr_mean, y=ctgan_auc_mean, 
        xerr=ctgan_dcr_std, yerr=ctgan_auc_std, 
        fmt='o', color='crimson', markersize=10, elinewidth=2, capsize=6,
        label='CTGAN'
    )
    plt.errorbar(
        x=tvae_dcr_mean, y=tvae_auc_mean, 
        xerr=tvae_dcr_std, yerr=tvae_auc_std, 
        fmt='s', color='royalblue', markersize=10, elinewidth=2, capsize=6,
        label='TVAE'
    )
    
    plt.title(f"Utility-Privacy Tradeoff: Downstream AUC vs DCR ({dataset_name.replace('_', ' ').title()})", fontsize=12, pad=15)
    plt.ylabel("Downstream Model Utility (ROC-AUC on Real Test Set)", fontsize=10)
    plt.xlabel("Privacy Metric: Distance to Closest Record (DCR) - Higher is More Private", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10)
    
    # Save plot as PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    # Save plot as PDF
    if output_path.endswith('.png'):
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
        
    plt.close()
    print(f"Utility-Privacy tradeoff plot saved to {output_path} (and PDF)")

def plot_explainability_consistency(summary_stats, dataset_name, output_path):
    """
    Generates a bar plot comparing SHAP Spearman correlation (rho) for CTGAN vs TVAE.
    Adds benchmark lines for consistency thresholds.
    Saves the plot in both PNG and PDF formats.
    """
    stats = summary_stats['metrics_summary']
    
    ctgan_mean = stats['ctgan_shap']['mean']
    ctgan_std = stats['ctgan_shap']['std']
    
    tvae_mean = stats['tvae_shap']['mean']
    tvae_std = stats['tvae_shap']['std']
    
    plt.figure(figsize=(7, 6))
    
    generators = ['CTGAN', 'TVAE']
    means = [ctgan_mean, tvae_mean]
    stds = [ctgan_std, tvae_std]
    
    # Draw bars with standard deviation error bars
    bars = plt.bar(
        generators, means, yerr=stds, 
        color=['crimson', 'royalblue'], alpha=0.85, edgecolor='black', capsize=8, width=0.5
    )
    
    # Add consistency threshold lines
    plt.axhline(y=0.7, color='forestgreen', linestyle=':', alpha=0.8, linewidth=1.5, label='High Consistency (>= 0.7)')
    plt.axhline(y=0.4, color='orange', linestyle=':', alpha=0.8, linewidth=1.5, label='Moderate Consistency (>= 0.4)')
    
    # Attach labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, height + 0.02, 
            f"{height:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
        
    plt.ylim(0, 1.05)
    plt.title(f"SHAP Consistency: Spearman Correlation vs Real Baseline ({dataset_name.replace('_', ' ').title()})", fontsize=12, pad=15)
    plt.ylabel("Spearman Rank Correlation (Rho)", fontsize=10)
    plt.xlabel("Generative Model Type", fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=9)
    
    # Save plot as PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    # Save plot as PDF
    if output_path.endswith('.png'):
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
        
    plt.close()
    print(f"SHAP consistency comparison plot saved to {output_path} (and PDF)")

def plot_feature_rank_heatmap(dataset_name, output_path):
    """
    Loads rank position CSVs across all 5 seeds, calculates the mean rank position
    for each feature under Real, CTGAN, and TVAE, and plots a rank heatmap.
    Saves in both PNG and PDF formats.
    """
    seeds = [42, 123, 456, 789, 1337]
    results_dir = "results/shap/"
    
    feature_ranks = {}
    
    for seed in seeds:
        for gen in ['real', 'ctgan', 'tvae']:
            csv_path = os.path.join(results_dir, f"{dataset_name}_{gen}_xgboost_seed{seed}_v1_shap_rank_position.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    feat = row['feature']
                    rank = row['rank_position']
                    if feat not in feature_ranks:
                        feature_ranks[feat] = {'real': [], 'ctgan': [], 'tvae': []}
                    feature_ranks[feat][gen].append(rank)
                    
    mean_ranks = []
    for feat, ranks in feature_ranks.items():
        mean_ranks.append({
            'feature': feat,
            'Real Baseline': np.mean(ranks['real']) if ranks['real'] else np.nan,
            'CTGAN': np.mean(ranks['ctgan']) if ranks['ctgan'] else np.nan,
            'TVAE': np.mean(ranks['tvae']) if ranks['tvae'] else np.nan
        })
        
    df_ranks = pd.DataFrame(mean_ranks)
    df_ranks = df_ranks.sort_values(by='Real Baseline').reset_index(drop=True)
    
    features = df_ranks['feature'].tolist()
    rank_matrix = df_ranks[['Real Baseline', 'CTGAN', 'TVAE']].values
    
    fig_height = 10 if len(features) > 12 else 6
    plt.figure(figsize=(8, fig_height))
    im = plt.imshow(rank_matrix, cmap='YlGnBu_r', aspect='auto')
    
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            val = rank_matrix[i, j]
            plt.text(j, i, f"{val:.1f}", ha='center', va='center', 
                     color='black' if val > (len(features)/2) else 'white',
                     fontweight='bold')
                     
    plt.colorbar(im, label='Mean Rank Position (Lower is More Important)')
    plt.xticks(ticks=[0, 1, 2], labels=['Real Baseline', 'CTGAN', 'TVAE'], fontsize=10, fontweight='bold')
    plt.yticks(ticks=np.arange(len(features)), labels=features, fontsize=9)
    plt.title(f"SHAP Feature Rank Comparison Heatmap\n({dataset_name.replace('_', ' ').title()})", fontsize=12, pad=15)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    if output_path.endswith('.png'):
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
        
    plt.close()
    print(f"SHAP feature-rank heatmap saved to {output_path} (and PDF)")

def plot_model_roc_curves(dataset_name, output_path):
    """
    Evaluates Real Baseline, CTGAN, and TVAE XGBoost models for Seed 42
    on the Real Test set and plots their ROC curves.
    Saves in both PNG and PDF formats.
    """
    if dataset_name == "german_credit":
        preprocess_german_credit(seed=42)
        dataset_suffix = "german"
        target_col = "class"
    else:
        preprocess_gmsc(seed=42)
        dataset_suffix = "gmsc"
        target_col = "SeriousDlqin2yrs"
        
    X_test_real = pd.read_csv(f"data/processed/X_test_{dataset_suffix}.csv")
    y_test_real = pd.read_csv(f"data/processed/y_test_{dataset_suffix}.csv").values.ravel()
    
    if dataset_name == "german_credit" and y_test_real.dtype == object:
        y_test_real = np.where(y_test_real == 'bad', 1, 0)
    else:
        y_test_real = y_test_real.astype(int)
        
    plt.figure(figsize=(7, 6))
    
    models_info = [
        ('Real Baseline', f"models/baseline/{dataset_name}_real_xgboost_seed42_v1.joblib", 'darkgray', '--'),
        ('CTGAN Synthetic', f"models/ctgan/{dataset_name}_ctgan_xgboost_seed42_v1.joblib", 'crimson', '-'),
        ('TVAE Synthetic', f"models/tvae/{dataset_name}_tvae_xgboost_seed42_v1.joblib", 'royalblue', '-')
    ]
    
    for label, model_path, color, style in models_info:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_prob = model.predict_proba(X_test_real)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_real, y_prob)
            auc = roc_auc_score(y_test_real, y_prob)
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})", color=color, linestyle=style, linewidth=2)
        else:
            print(f"Model {model_path} not found. Skipping ROC curve plot.")
            
    plt.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (FPR)', fontsize=10)
    plt.ylabel('True Positive Rate (TPR)', fontsize=10)
    plt.title(f"XGBoost ROC Curves on Real Test Set (Seed 42)\n({dataset_name.replace('_', ' ').title()})", fontsize=12, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=9)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    if output_path.endswith('.png'):
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
        
    plt.close()
    print(f"Model ROC curves saved to {output_path} (and PDF)")

def plot_mia_roc_curves(dataset_name, output_path):
    """
    Computes closest distance of each train and test record to synthetic records,
    and plots Membership Inference Attack ROC Curves for CTGAN and TVAE (Seed 42).
    Saves in both PNG and PDF formats.
    """
    if dataset_name == "german_credit":
        preprocess_german_credit(seed=42)
        dataset_suffix = "german"
        target_col = "class"
    else:
        preprocess_gmsc(seed=42)
        dataset_suffix = "gmsc"
        target_col = "SeriousDlqin2yrs"
        
    X_train_real = pd.read_csv(f"data/processed/X_train_{dataset_suffix}.csv")
    X_test_real = pd.read_csv(f"data/processed/X_test_{dataset_suffix}.csv")
    
    plt.figure(figsize=(7, 6))
    
    mia_info = [
        ('CTGAN', f"data/synthetic/{dataset_name}_ctgan_seed42.csv", 'crimson'),
        ('TVAE', f"data/synthetic/{dataset_name}_tvae_seed42.csv", 'royalblue')
    ]
    
    for label, syn_path, color in mia_info:
        if os.path.exists(syn_path):
            df_syn = pd.read_csv(syn_path)
            
            from src.models.synthetic import fit_real_preprocessing, preprocess_synthetic_data
            fitted = fit_real_preprocessing(dataset_name, seed=42)
            X_syn_proc, _ = preprocess_synthetic_data(df_syn, dataset_name, fitted, target_col)
            
            dists_train = cdist(X_train_real, X_syn_proc, metric='euclidean')
            d_train = dists_train.min(axis=1)
            
            dists_test = cdist(X_test_real, X_syn_proc, metric='euclidean')
            d_test = dists_test.min(axis=1)
            
            scores = np.concatenate([-d_train, -d_test])
            labels = np.concatenate([np.ones(len(d_train)), np.zeros(len(d_test))])
            
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = roc_auc_score(labels, scores)
            
            plt.plot(fpr, tpr, label=f"{label} (MIA AUC = {auc:.4f})", color=color, linewidth=2)
        else:
            print(f"Synthetic file {syn_path} not found. Skipping MIA plot.")
            
    plt.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (FPR)', fontsize=10)
    plt.ylabel('True Positive Rate (TPR)', fontsize=10)
    plt.title(f"Distance-based Membership Inference Attack (Seed 42)\n({dataset_name.replace('_', ' ').title()})", fontsize=12, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=9)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    if output_path.endswith('.png'):
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
        
    plt.close()
    print(f"MIA ROC curves saved to {output_path} (and PDF)")

def copy_and_convert_shap_beeswarms():
    """
    Copies pre-generated Seed 42 beeswarm plots to the figures/paper/ folder,
    and converts/saves them as PDFs.
    """
    mapping = [
        ("german_credit", "real"), ("german_credit", "ctgan"), ("german_credit", "tvae"),
        ("gmsc", "real"), ("gmsc", "ctgan"), ("gmsc", "tvae")
    ]
    
    for dataset, gen in mapping:
        src_png = f"figures/shap/{dataset}_{gen}_xgboost_seed42_v1_shap_beeswarm.png"
        dst_png = f"figures/paper/{dataset}_{gen}_shap_beeswarm.png"
        dst_pdf = f"figures/paper/{dataset}_{gen}_shap_beeswarm.pdf"
        
        if os.path.exists(src_png):
            os.makedirs(os.path.dirname(dst_png), exist_ok=True)
            shutil.copyfile(src_png, dst_png)
            print(f"Copied SHAP beeswarm to {dst_png}")
            
            try:
                img = Image.open(src_png)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                    background.save(dst_pdf, "PDF", resolution=100.0)
                else:
                    img.convert('RGB').save(dst_pdf, "PDF", resolution=100.0)
                print(f"Converted SHAP beeswarm to {dst_pdf}")
            except Exception as e:
                print(f"Error converting {src_png} to PDF: {e}")
        else:
            print(f"Source beeswarm PNG {src_png} not found. Skipping.")

def generate_dataset_stats_table(output_dir):
    """
    Creates dataset_statistics_table.tex summarizing dataset stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats_latex = r"""\begin{table}[htbp]
\centering
\caption{Key Dataset Characteristics and Evaluation Settings}
\label{tab:dataset_stats}
\begin{tabular}{lccccc}
\hline
\textbf{Dataset} & \textbf{Total Records} & \textbf{Features} & \textbf{Numerical Features} & \textbf{Categorical Features} & \textbf{Positive Target Ratio} \\
\hline
German Credit & 1,000 & 20 & 7 & 13 & 30.0\% (Class: `bad') \\
GMSC (Sampled) & 10,000 & 10 & 10 & 0 & 6.69\% (Class: `1') \\
\hline
\end{tabular}
\end{table}
"""
    table_path = os.path.join(output_dir, "dataset_statistics_table.tex")
    with open(table_path, 'w') as f:
        f.write(stats_latex)
    print(f"Dataset stats table saved to {table_path}")

def generate_latex_tables(german_stats, gmsc_stats, output_dir):
    """
    Generates LaTeX-formatted strings for model utility and privacy comparisons.
    Saves tables as .tex files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    g_m = german_stats['metrics_summary']
    c_m = gmsc_stats['metrics_summary']
    
    # 1. LaTeX Table 1: Model Predictive Utility Comparison
    utility_latex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Downstream Classification Performance (XGBoost) comparing Real Baseline vs. Synthetic-Trained Models (Mean $\pm$ Std)}}
\label{{tab:utility_comparison}}
\begin{{tabular}}{{lccccc}}
\hline
\textbf{{Dataset}} & \textbf{{Model Type}} & \textbf{{ROC-AUC}} & \textbf{{95\% CI}} & \textbf{{F1-Score}} & \textbf{{95\% CI}} \\
\hline
German Credit & Real Baseline & {g_m['real_auc']['mean']:.4f} $\pm$ {g_m['real_auc']['std']:.4f} & [{g_m['real_auc']['ci_95_lower']:.4f}, {g_m['real_auc']['ci_95_upper']:.4f}] & {g_m['real_f1']['mean']:.4f} $\pm$ {g_m['real_f1']['std']:.4f} & [{g_m['real_f1']['ci_95_lower']:.4f}, {g_m['real_f1']['ci_95_upper']:.4f}] \\
German Credit & CTGAN Synthetic & {g_m['ctgan_auc']['mean']:.4f} $\pm$ {g_m['ctgan_auc']['std']:.4f} & [{g_m['ctgan_auc']['ci_95_lower']:.4f}, {g_m['ctgan_auc']['ci_95_upper']:.4f}] & {g_m['ctgan_f1']['mean']:.4f} $\pm$ {g_m['ctgan_f1']['std']:.4f} & [{g_m['ctgan_f1']['ci_95_lower']:.4f}, {g_m['ctgan_f1']['ci_95_upper']:.4f}] \\
German Credit & TVAE Synthetic & {g_m['tvae_auc']['mean']:.4f} $\pm$ {g_m['tvae_auc']['std']:.4f} & [{g_m['tvae_auc']['ci_95_lower']:.4f}, {g_m['tvae_auc']['ci_95_upper']:.4f}] & {g_m['tvae_f1']['mean']:.4f} $\pm$ {g_m['tvae_f1']['std']:.4f} & [{g_m['tvae_f1']['ci_95_lower']:.4f}, {g_m['tvae_f1']['ci_95_upper']:.4f}] \\
\hline
GMSC & Real Baseline & {c_m['real_auc']['mean']:.4f} $\pm$ {c_m['real_auc']['std']:.4f} & [{c_m['real_auc']['ci_95_lower']:.4f}, {c_m['real_auc']['ci_95_upper']:.4f}] & {c_m['real_f1']['mean']:.4f} $\pm$ {c_m['real_f1']['std']:.4f} & [{c_m['real_f1']['ci_95_lower']:.4f}, {c_m['real_f1']['ci_95_upper']:.4f}] \\
GMSC & CTGAN Synthetic & {c_m['ctgan_auc']['mean']:.4f} $\pm$ {c_m['ctgan_auc']['std']:.4f} & [{c_m['ctgan_auc']['ci_95_lower']:.4f}, {c_m['ctgan_auc']['ci_95_upper']:.4f}] & {c_m['ctgan_f1']['mean']:.4f} $\pm$ {c_m['ctgan_f1']['std']:.4f} & [{c_m['ctgan_f1']['ci_95_lower']:.4f}, {c_m['ctgan_f1']['ci_95_upper']:.4f}] \\
GMSC & TVAE Synthetic & {c_m['tvae_auc']['mean']:.4f} $\pm$ {c_m['tvae_auc']['std']:.4f} & [{c_m['tvae_auc']['ci_95_lower']:.4f}, {c_m['tvae_auc']['ci_95_upper']:.4f}] & {c_m['tvae_f1']['mean']:.4f} $\pm$ {c_m['tvae_f1']['std']:.4f} & [{c_m['tvae_f1']['ci_95_lower']:.4f}, {c_m['tvae_f1']['ci_95_upper']:.4f}] \\
\hline
\end{{tabular}}
\end{{table}}
"""

    utility_path = os.path.join(output_dir, "utility_comparison_table.tex")
    with open(utility_path, 'w') as f:
        f.write(utility_latex)
    print(f"LaTeX utility table saved to {utility_path}")
    
    # 2. LaTeX Table 2: Explainability Consistency and Privacy Scores Comparison
    privacy_latex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Explainability Fidelity (SHAP Spearman correlation $\rho$) and Tabular Privacy Metrics (DCR, NNDR, MIA AUC) (Mean $\pm$ Std)}}
\label{{tab:privacy_explainability_comparison}}
\begin{{tabular}}{{lccccc}}
\hline
\textbf{{Dataset}} & \textbf{{Generator}} & \textbf{{SHAP Spearman $\rho$}} & \textbf{{Mean DCR}} & \textbf{{Mean NNDR}} & \textbf{{MIA ROC-AUC}} \\
\hline
German Credit & CTGAN & {g_m['ctgan_shap']['mean']:.4f} $\pm$ {g_m['ctgan_shap']['std']:.4f} & {g_m['ctgan_dcr']['mean']:.4f} $\pm$ {g_m['ctgan_dcr']['std']:.4f} & {g_m['ctgan_nndr']['mean']:.4f} $\pm$ {g_m['ctgan_nndr']['std']:.4f} & {g_m['ctgan_mia']['mean']:.4f} $\pm$ {g_m['ctgan_mia']['std']:.4f} \\
German Credit & TVAE & {g_m['tvae_shap']['mean']:.4f} $\pm$ {g_m['tvae_shap']['std']:.4f} & {g_m['tvae_dcr']['mean']:.4f} $\pm$ {g_m['tvae_dcr']['std']:.4f} & {g_m['tvae_nndr']['mean']:.4f} $\pm$ {g_m['tvae_nndr']['std']:.4f} & {g_m['tvae_mia']['mean']:.4f} $\pm$ {g_m['tvae_mia']['std']:.4f} \\
\hline
GMSC & CTGAN & {c_m['ctgan_shap']['mean']:.4f} $\pm$ {c_m['ctgan_shap']['std']:.4f} & {c_m['ctgan_dcr']['mean']:.4f} $\pm$ {c_m['ctgan_dcr']['std']:.4f} & {c_m['ctgan_nndr']['mean']:.4f} $\pm$ {c_m['ctgan_nndr']['std']:.4f} & {c_m['ctgan_mia']['mean']:.4f} $\pm$ {c_m['ctgan_mia']['std']:.4f} \\
GMSC & TVAE & {c_m['tvae_shap']['mean']:.4f} $\pm$ {c_m['tvae_shap']['std']:.4f} & {c_m['tvae_dcr']['mean']:.4f} $\pm$ {c_m['tvae_dcr']['std']:.4f} & {c_m['tvae_nndr']['mean']:.4f} $\pm$ {c_m['tvae_nndr']['std']:.4f} & {c_m['tvae_mia']['mean']:.4f} $\pm$ {c_m['tvae_mia']['std']:.4f} \\
\hline
\end{{tabular}}
\end{{table}}
"""

    privacy_path = os.path.join(output_dir, "privacy_consistency_comparison_table.tex")
    with open(privacy_path, 'w') as f:
        f.write(privacy_latex)
    print(f"LaTeX privacy/consistency table saved to {privacy_path}")
