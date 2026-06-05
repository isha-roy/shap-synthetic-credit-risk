import os
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def download_german_credit(output_path="data/raw/german_credit.csv"):
    """Downloads German Credit (credit-g OpenML ID 31) and saves to output_path."""
    print("Downloading German Credit dataset from OpenML...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Fetch OpenML dataset
    dataset = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
    df = dataset.frame
    df.to_csv(output_path, index=False)
    print(f"German Credit dataset saved to {output_path}. Shape: {df.shape}")
    return df

def sample_gmsc(input_path="data/raw/cs-training.csv", output_path="data/raw/gmsc_sampled_10k.csv"):
    """Verifies cs-training.csv exists, takes a stratified 10k sample, and saves it."""
    print("Checking for raw GMSC dataset...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Raw GMSC file not found at '{input_path}'. "
            "Please download 'cs-training.csv' from Kaggle (https://www.kaggle.com/c/GiveMeSomeCredit) "
            "and place it in the 'data/raw/' directory."
        )
    df = pd.read_csv(input_path)
    # GMSC has an unnamed index column as the first column. Let's drop it.
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    target = 'SeriousDlqin2yrs'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in raw GMSC columns.")
    
    # Sample 10,000 rows maintaining target ratio
    sample_df, _ = train_test_split(
        df,
        train_size=10000,
        stratify=df[target],
        random_state=42
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df.to_csv(output_path, index=False)
    print(f"GMSC 10k stratified sample saved to {output_path}. Shape: {sample_df.shape}")
    return sample_df

def preprocess_german_credit(
    raw_path="data/raw/german_credit.csv",
    processed_dir="data/processed/",
    model_dir="models/baseline/"
):
    """Preprocesses German Credit dataset: imputation, encoding, scaling, stratified splitting."""
    print("Preprocessing German Credit dataset...")
    if not os.path.exists(raw_path):
        download_german_credit(raw_path)
        
    df = pd.read_csv(raw_path)
    
    target = 'class'
    X = df.drop(columns=[target])
    y = df[target]
    
    # 1. Stratified split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Identify numerical and categorical columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Imputation (fit ONLY on train)
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
    X_test_num = pd.DataFrame(num_imputer.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)
    
    if len(cat_cols) > 0:
        X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_cols]), columns=cat_cols, index=X_train.index)
        X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[cat_cols]), columns=cat_cols, index=X_test.index)
    else:
        X_train_cat = pd.DataFrame(index=X_train.index)
        X_test_cat = pd.DataFrame(index=X_test.index)
        
    # 3. Encoding categoricals for XGBoost
    encoders = {}
    X_train_cat_enc = X_train_cat.copy()
    X_test_cat_enc = X_test_cat.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        X_train_cat_enc[col] = le.fit_transform(X_train_cat[col])
        X_test_cat_enc[col] = le.transform(X_test_cat[col])
        encoders[col] = le
        
    # 4. Scaling numericals (scaler fits ONLY on training numericals)
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=num_cols, index=X_train.index)
    X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=num_cols, index=X_test.index)
    
    # Recombine features, maintaining column order
    X_train_proc = pd.concat([X_train_num_scaled, X_train_cat_enc], axis=1)[X.columns]
    X_test_proc = pd.concat([X_test_num_scaled, X_test_cat_enc], axis=1)[X.columns]
    
    # Map target: 'bad' (default/risk) to 1, 'good' to 0
    y_train_enc = y_train.map({'good': 0, 'bad': 1}).astype(int)
    y_test_enc = y_test.map({'good': 0, 'bad': 1}).astype(int)
    target_encoder = {'good': 0, 'bad': 1}
    
    # Save processed datasets
    os.makedirs(processed_dir, exist_ok=True)
    X_train_proc.to_csv(os.path.join(processed_dir, "X_train_german.csv"), index=False)
    y_train_enc.to_csv(os.path.join(processed_dir, "y_train_german.csv"), index=False)
    X_test_proc.to_csv(os.path.join(processed_dir, "X_test_german.csv"), index=False)
    y_test_enc.to_csv(os.path.join(processed_dir, "y_test_german.csv"), index=False)
    
    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(num_imputer, os.path.join(model_dir, "german_num_imputer.joblib"))
    joblib.dump(cat_imputer, os.path.join(model_dir, "german_cat_imputer.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "german_scaler.joblib"))
    joblib.dump(encoders, os.path.join(model_dir, "german_cat_encoders.joblib"))
    joblib.dump(target_encoder, os.path.join(model_dir, "german_target_encoder.joblib"))
    
    print("German Credit dataset preprocessing completed successfully!")

def preprocess_gmsc(
    raw_path="data/raw/gmsc_sampled_10k.csv",
    processed_dir="data/processed/",
    model_dir="models/baseline/"
):
    """Preprocesses GMSC dataset: numerical imputation, scaling, stratified splitting."""
    print("Preprocessing GMSC dataset...")
    if not os.path.exists(raw_path):
        sample_gmsc(output_path=raw_path)
        
    df = pd.read_csv(raw_path)
    
    target = 'SeriousDlqin2yrs'
    X = df.drop(columns=[target])
    y = df[target]
    
    # 1. Stratified split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # 2. Imputation (fit ONLY on train)
    num_cols = X_train.columns.tolist()
    num_imputer = SimpleImputer(strategy='median')
    X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
    X_test_num = pd.DataFrame(num_imputer.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)
    
    # 3. Scaling (scaler fits ONLY on training numericals)
    scaler = StandardScaler()
    X_train_proc = pd.DataFrame(scaler.fit_transform(X_train_num), columns=num_cols, index=X_train.index)
    X_test_proc = pd.DataFrame(scaler.transform(X_test_num), columns=num_cols, index=X_test.index)
    
    y_train_proc = y_train.rename(target)
    y_test_proc = y_test.rename(target)
    
    # Save processed datasets
    os.makedirs(processed_dir, exist_ok=True)
    X_train_proc.to_csv(os.path.join(processed_dir, "X_train_gmsc.csv"), index=False)
    y_train_proc.to_csv(os.path.join(processed_dir, "y_train_gmsc.csv"), index=False)
    X_test_proc.to_csv(os.path.join(processed_dir, "X_test_gmsc.csv"), index=False)
    y_test_proc.to_csv(os.path.join(processed_dir, "y_test_gmsc.csv"), index=False)
    
    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(num_imputer, os.path.join(model_dir, "gmsc_num_imputer.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "gmsc_scaler.joblib"))
    
    print("GMSC dataset preprocessing completed successfully!")
