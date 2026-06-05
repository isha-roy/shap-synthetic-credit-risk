import os
import matplotlib.pyplot as plt
import numpy as np

def plot_utility_privacy_tradeoff(summary_stats, dataset_name, output_path):
    """
    Generates a scatter plot mapping Downstream ROC-AUC against Mean DCR with error bars.
    Draws the Real baseline as a horizontal line with shaded standard deviation.
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
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Utility-Privacy tradeoff plot saved to {output_path}")

def plot_explainability_consistency(summary_stats, dataset_name, output_path):
    """
    Generates a bar plot comparing SHAP Spearman correlation (rho) for CTGAN vs TVAE.
    Adds benchmark lines for consistency thresholds.
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
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"SHAP consistency comparison plot saved to {output_path}")

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
    g_tests = german_stats['paired_tests']
    c_tests = gmsc_stats['paired_tests']
    
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
