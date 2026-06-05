import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV

def tune_xgboost(X_train, y_train, dataset_name):
    """
    Runs GridSearchCV with 5-fold CV to find the best XGBoost hyperparameters on training data.
    Uses scale_pos_weight to address class imbalance.
    """
    print(f"Tuning hyperparameters for {dataset_name} baseline...")
    # Calculate scale_pos_weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f} (neg={num_neg}, pos={num_pos})")
    
    # Define XGBoost base classifier
    clf = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False
    )
    
    # Load hyperparameter grid from configs or define here
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300]
    }
    
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found for {dataset_name}: {grid_search.best_params_}")
    print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_

def train_baseline_model(X_train, y_train, X_test, y_test, params, dataset_name, seed, model_dir="models/baseline/"):
    """
    Trains XGBoost with selected hyperparameters, evaluates on test set, and saves model.
    """
    print(f"Training baseline XGBoost for {dataset_name} (Seed {seed})...")
    # Calculate scale_pos_weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    
    # Create classifier
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=seed,
        use_label_encoder=False
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Evaluate performance
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "dataset": dataset_name,
        "seed": seed,
        "parameters": params,
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }
    
    # Save the model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{dataset_name}_real_xgboost_seed{seed}_v1.joblib")
    joblib.dump(model, model_path)
    print(f"Saved baseline model to {model_path}")
    print(f"Metrics: ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    return model, metrics

def compute_shap_reference(model, X_test, dataset_name, seed, results_dir="results/shap/", figures_dir="figures/shap/"):
    """
    Computes test set SHAP attributions using TreeExplainer, saves data and rankings, and generates summary plots.
    """
    print(f"Computing SHAP values for {dataset_name} (Seed {seed})...")
    # TreeExplainer fits tree models directly
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # In shap>=0.45.0, TreeExplainer on binary XGBoost might return a 2D array or list/3D array
    # If it is a list of arrays (one per class), we take class 1 (positive class) attributions
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    elif len(shap_values.shape) == 3:
        # If shape is (n_samples, n_features, n_classes), take index 1 for class 1
        shap_values_class1 = shap_values[:, :, 1]
    else:
        # Standard 2D array of shap values (usually default for binary XGBoost)
        shap_values_class1 = shap_values
        
    # Create target directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Save raw SHAP values as numpy binary
    shap_val_path = os.path.join(results_dir, f"{dataset_name}_real_xgboost_seed{seed}_v1_shap_values.npy")
    np.save(shap_val_path, shap_values_class1)
    
    # 2. Compute mean absolute SHAP attributions and save ranking
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    features = X_test.columns.tolist()
    
    ranking_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)
    
    ranking_path = os.path.join(results_dir, f"{dataset_name}_real_xgboost_seed{seed}_v1_shap_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)
    
    # 3. Save feature rank positions
    rank_pos_df = ranking_df.copy()
    rank_pos_df["rank_position"] = np.arange(1, len(features) + 1)
    # Sort alphabetically by feature name so we can easily compare feature rank positions
    rank_pos_df = rank_pos_df.sort_values(by="feature")
    
    rank_pos_path = os.path.join(results_dir, f"{dataset_name}_real_xgboost_seed{seed}_v1_shap_rank_position.csv")
    rank_pos_df.to_csv(rank_pos_path, index=False)
    
    # 4. Generate and save SHAP Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_test, show=False)
    beeswarm_path = os.path.join(figures_dir, f"{dataset_name}_real_xgboost_seed{seed}_v1_shap_beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches='tight')
    plt.close()
    
    # 5. Generate and save SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_test, plot_type="bar", show=False)
    bar_path = os.path.join(figures_dir, f"{dataset_name}_real_xgboost_seed{seed}_v1_shap_bar.png")
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP calculations and plots completed for {dataset_name} (Seed {seed})")
    return shap_values_class1
