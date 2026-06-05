import os
import sys
import json

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.visualization.plots import plot_utility_privacy_tradeoff, plot_explainability_consistency, generate_latex_tables

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    print("======================================================================")
    print("Shap Synthetic Credit Risk: Generating Final Paper Visualizations")
    print("======================================================================\n")
    
    # Target directories
    figures_dir = "figures/paper/"
    summaries_dir = "results/summaries/"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    
    # 1. Load statistical validation results
    german_stats_path = "results/statistics/german_credit_statistical_validation.json"
    gmsc_stats_path = "results/statistics/gmsc_statistical_validation.json"
    
    if not os.path.exists(german_stats_path) or not os.path.exists(gmsc_stats_path):
        raise FileNotFoundError(
            "Statistical summaries not found. Run scripts/run_statistical_validation.py first."
        )
        
    german_stats = load_json(german_stats_path)
    gmsc_stats = load_json(gmsc_stats_path)
    
    # 2. Generate tradeoff scatter plots
    print("Generating Utility-Privacy tradeoff scatter plots...")
    plot_utility_privacy_tradeoff(
        summary_stats=german_stats, 
        dataset_name="german_credit", 
        output_path=os.path.join(figures_dir, "german_credit_utility_privacy_tradeoff.png")
    )
    plot_utility_privacy_tradeoff(
        summary_stats=gmsc_stats, 
        dataset_name="gmsc", 
        output_path=os.path.join(figures_dir, "gmsc_utility_privacy_tradeoff.png")
    )
    
    # 3. Generate SHAP consistency bar plots
    print("\nGenerating SHAP consistency comparison plots...")
    plot_explainability_consistency(
        summary_stats=german_stats, 
        dataset_name="german_credit", 
        output_path=os.path.join(figures_dir, "german_credit_shap_consistency.png")
    )
    plot_explainability_consistency(
        summary_stats=gmsc_stats, 
        dataset_name="gmsc", 
        output_path=os.path.join(figures_dir, "gmsc_shap_consistency.png")
    )
    
    # 4. Generate LaTeX tables
    print("\nGenerating LaTeX table markup...")
    generate_latex_tables(german_stats, gmsc_stats, summaries_dir)
    
    print("\n======================================================================")
    print("Final Paper Visualizations and LaTeX Tables Generated Successfully")
    print("======================================================================")

if __name__ == "__main__":
    main()
