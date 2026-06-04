# Dataset Acquisition and Reference Notes

This document provides details on where to acquire the datasets and how they should be structured.

## Dataset 1: German Credit Dataset

* **Source**: UCI Machine Learning Repository / OpenML (`credit-g`, Dataset ID: 31)
* **URL**: [OpenML credit-g](https://www.openml.org/search?type=data&status=active&id=31)
* **Format**: ARFF/CSV
* **Description**:
  * **Rows**: 1,000
  * **Features**: 20 (7 numerical, 13 categorical)
  * **Target**: `class` (binary: `good` / `bad`)
  * **Imbalance**: 700 `good` (70%) and 300 `bad` (30%)
* **Local Storage Path**: `data/raw/german_credit.csv`

## Dataset 2: Give Me Some Credit (GMSC)

* **Source**: Kaggle competition "Give Me Some Credit"
* **URL**: [Kaggle GMSC](https://www.kaggle.com/c/GiveMeSomeCredit)
* **Format**: CSV
* **Description (Original)**:
  * **Rows**: 150,000
  * **Features**: 10 (all numerical)
  * **Target**: `SeriousDlqin2yrs` (binary: `1` default, `0` non-default)
* **Sampling Protocol**:
  * Due to local CPU/RAM constraints (8 GB RAM), training CTGAN/TVAE on the full 150,000 rows is computationally prohibitive.
  * A stratified random sample of **10,000 rows** must be drawn based on the target column `SeriousDlqin2yrs`.
  * The sampling code must use a fixed random seed (42) to ensure the exact same evaluation subset is generated.
* **Local Storage Path**:
  * Original full raw: `data/raw/cs-training.csv` (from Kaggle)
  * Sampled raw: `data/raw/gmsc_sampled_10k.csv` (this file will serve as the starting point for GMSC modeling)
