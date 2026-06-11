import os
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def train_synthetic_model_lgbm(
    X_train_syn, y_train_syn, X_test_real, y_test_real, 
    params, dataset_name, generator_type, seed, 
    model_dir="models/", results_dir="results/"
):
    """
    Trains LightGBM classifier on preprocessed synthetic training split, evaluates
    on real test split, and saves prediction files and model artifacts.
    """
    print(f"Training LightGBM on {generator_type.upper()} synthetic data for {dataset_name} (Seed {seed})...")
    
    # Dynamic scale_pos_weight
    num_neg = (y_train_syn == 0).sum()
    num_pos = (y_train_syn == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    # Determine num_leaves based on max_depth to avoid warning/issues
    max_depth = params.get('max_depth', -1)
    num_leaves = 2 ** max_depth - 1 if max_depth > 0 else 31
    
    model = lgb.LGBMClassifier(
        **params,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        objective='binary',
        metric='auc',
        random_state=seed,
        verbose=-1
    )
    
    model.fit(X_train_syn, y_train_syn)
    
    # Evaluate on real test split
    y_pred = model.predict(X_test_real)
    y_pred_proba = model.predict_proba(X_test_real)[:, 1]
    
    roc_auc = roc_auc_score(y_test_real, y_pred_proba)
    f1 = f1_score(y_test_real, y_pred)
    precision = precision_score(y_test_real, y_pred, zero_division=0)
    recall = recall_score(y_test_real, y_pred, zero_division=0)
    cm = confusion_matrix(y_test_real, y_pred)
    
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
    os.makedirs(os.path.join(model_dir, generator_type), exist_ok=True)
    model_path = os.path.join(model_dir, generator_type, f"{dataset_name}_{generator_type}_lgbm_seed{seed}_v1.joblib")
    joblib.dump(model, model_path)
    print(f"Saved synthetic-trained model to {model_path}")
    
    # Save test set predictions (probabilities and labels)
    os.makedirs(os.path.join(results_dir, generator_type), exist_ok=True)
    preds_df = pd.DataFrame({
        "y_true": y_test_real.ravel(),
        "y_pred": y_pred,
        "y_prob": y_pred_proba
    })
    preds_path = os.path.join(results_dir, generator_type, f"{dataset_name}_{generator_type}_lgbm_seed{seed}_v1_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved real test predictions to {preds_path}")
    
    return model, metrics
