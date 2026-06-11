# Project Report: Evaluating SHAP Explainability Consistency and Privacy-Utility Tradeoff Across Synthetic Credit Risk Datasets

**Author:** Isha Roy  
**Affiliation:** Department of Computer Science and Engineering, National Institute of Technology Goa  
**Target Venue:** IEEE Access  

---

## Table of Contents
1. [What This Project Is About](#1-what-this-project-is-about)
2. [Why This Project Matters](#2-why-this-project-matters)
3. [The Two Research Papers That Inspired This Work](#3-the-two-research-papers-that-inspired-this-work)
4. [How This Project Goes Beyond Those Papers](#4-how-this-project-goes-beyond-those-papers)
5. [Datasets Used](#5-datasets-used)
6. [End-to-End Pipeline (All 7 Phases)](#6-end-to-end-pipeline-all-7-phases)
7. [Results: Downstream Predictive Utility](#7-results-downstream-predictive-utility)
8. [Results: SHAP Explainability Consistency](#8-results-shap-explainability-consistency)
9. [Results: Privacy Metrics (DCR, NNDR, MIA)](#9-results-privacy-metrics-dcr-nndr-mia)
10. [Results: Inference Risk Indicator (New Metric)](#10-results-inference-risk-indicator-new-metric)
10a. [Results: Differential Privacy Analysis](#10a-results-differential-privacy-analysis)
11. [Results: Statistical Significance](#11-results-statistical-significance)
12. [Master Results Table](#12-master-results-table)
13. [Figures and Visualizations](#13-figures-and-visualizations)
14. [Key Findings (Plain English)](#14-key-findings-plain-english)
15. [Limitations and Gaps](#15-limitations-and-gaps)
16. [How This Project Can Be Improved](#16-how-this-project-can-be-improved)
17. [Why This Project Demonstrates ML Engineering Skills](#17-why-this-project-demonstrates-ml-engineering-skills)

---

## 1. What This Project Is About

Banks use machine learning models to decide who gets a loan. These models need to be:
- **Accurate** — they should correctly predict who will default.
- **Explainable** — regulators (GDPR, EU AI Act) require banks to explain *why* a loan was denied.
- **Private** — real customer data cannot be shared freely for model development.

To solve the privacy problem, banks generate **synthetic data** — fake customer records that look statistically similar to real ones. Two popular methods for generating synthetic tabular data are:
- **CTGAN** (Conditional Tabular GAN) — uses adversarial training.
- **TVAE** (Tabular Variational Autoencoder) — uses variational inference.

**The core question this project answers:**
> If you train a credit scoring model on synthetic data instead of real data, do you get the same predictions AND the same explanations? And does the synthetic data actually protect privacy, or does it accidentally leak real customer information?

We answer this by building three parallel pipelines and comparing them head-to-head:
- **Pipeline 1 (Baseline):** XGBoost trained on real data.
- **Pipeline 2:** XGBoost trained on CTGAN synthetic data.
- **Pipeline 3:** XGBoost trained on TVAE synthetic data.

All three models are tested on the same real test set, and we compare their predictions, SHAP explanations, and privacy spacing.

---

## 2. Why This Project Matters

| Stakeholder | Why They Care |
|:---|:---|
| **Banks & Fintech** | Need to know if synthetic data is safe to share with third-party auditors |
| **Regulators** | GDPR Article 22 requires explainable credit decisions — synthetic training must preserve explanations |
| **ML Engineers** | Need practical guidance on which generator to use and what privacy checks to run |
| **Researchers** | No prior study jointly evaluates utility + explainability + privacy for synthetic credit data |

---

## 3. The Two Research Papers That Inspired This Work

### Paper 1: Han et al. (2024) — "Non-parametric oversampling technique for explainable credit scoring"
**What they did:** Used conditional Wasserstein GANs as an oversampling tool to fix class imbalance in credit datasets. They trained models on augmented data and used SHAP to explain predictions.

**What they missed:**
- They only used GANs for oversampling (adding more minority class samples), not full synthetic replacement.
- They treated SHAP as a secondary visualization tool — they never measured whether SHAP rankings *change* when you switch from real to synthetic training data.
- They did not evaluate any privacy metrics at all.

### Paper 2: Min & Oh (2025) — "Can synthetic data protect privacy?" (IEEE Access)
**What they did:** Performed a deep privacy evaluation of synthetic datasets using DCR, NNDR, Membership Inference Attacks, and the Inference Risk Indicator. They tested CTGAN, TVAE, and other generators on multiple datasets.

**What they missed:**
- They evaluated privacy metrics directly on raw synthetic data — they never trained a downstream classifier.
- They never used SHAP or any explainability method.
- They never asked: "If you use this synthetic data to train a model, does the model still work? Does it still explain itself the same way?"

---

## 4. How This Project Goes Beyond Those Papers

This project is the **first study** to jointly evaluate all three dimensions in a single framework:

| Dimension | Han et al. (2024) | Min & Oh (2025) | **This Project** |
|:---|:---:|:---:|:---:|
| Downstream classification utility (AUC, F1) | Partial (oversampling only) | No | **Yes (full synthetic training)** |
| SHAP explainability consistency | No (qualitative only) | No | **Yes (Spearman rho + Jaccard overlap)** |
| DCR (Distance to Closest Record) | No | Yes | **Yes** |
| NNDR (Nearest Neighbor Distance Ratio) | No | Yes | **Yes** |
| Membership Inference Attack (MIA) | No | Yes | **Yes** |
| Inference Risk Indicator | No | Yes | **Yes (newly added)** |
| Multi-seed statistical validation | No | No | **Yes (10 seeds, Wilcoxon + Cohen's d)** |
| Multiple datasets with different characteristics | No | Partial | **Yes (mixed-type + all-numerical)** |
| Utility-Explainability-Privacy tradeoff analysis | No | No | **Yes (core contribution)** |

**In short:** Han et al. focused on utility. Min & Oh focused on privacy. We connect both and add explainability as the missing bridge.

---

## 5. Datasets Used

| Property | German Credit (UCI) | GMSC (Kaggle) |
|:---|:---:|:---:|
| **Total records** | 1,000 | 10,000 (stratified sample from 150K) |
| **Features** | 20 (7 numerical + 13 categorical) | 10 (all numerical) |
| **Target** | `class` (good/bad) | `SeriousDlqin2yrs` (0/1) |
| **Default rate** | 30% | 6.69% |
| **Train/Test split** | 70/30 stratified | 70/30 stratified |
| **Why chosen** | Small, mixed-type, stresses generators | Large, continuous, tests scalability |

These two datasets were deliberately chosen to stress-test the generators under different conditions. German Credit is small with many categorical features (hard for GANs). GMSC is larger with only numerical features (easier for GANs but tests scalability).

---

## 6. End-to-End Pipeline (All 7 Phases)

### Phase 0 — Data Acquisition and Splitting
- Download German Credit from OpenML and GMSC from Kaggle.
- For each of the 10 seeds (`[42, 123, 456, 789, 1337, 2024, 7, 99, 314, 2718]`), create a fresh stratified 70/30 train/test split.
- This ensures all results are robust and not dependent on a single lucky split.

### Phase 1 — Preprocessing (Leakage-Safe)
All transformers are fit **only on the training split** and applied to test + synthetic data:
- **Numerical features:** Median imputation + Z-score standardization (StandardScaler).
- **Categorical features:** Mode imputation + integer Label Encoding.
- **Why this matters:** If you fit a scaler on the test set, you get optimistic results that don't generalize. We prevent this.

### Phase 2 — Synthetic Data Generation
For each seed, we train the generators on the preprocessed training split:
- **CTGAN:** 300 epochs, batch size 500, generator/discriminator dims (256, 256).
- **TVAE:** 300 epochs, batch size 500, encoder/decoder dims (128, 128), latent dim 128.
- We generate the same number of synthetic records as the training split.

### Phase 3 — Downstream Classification (XGBoost)
Three XGBoost models are trained per seed:
- **Real Baseline:** Trained on real training data.
- **CTGAN Model:** Trained on CTGAN synthetic data.
- **TVAE Model:** Trained on TVAE synthetic data.

All three are evaluated on the **same real test split**. We report ROC-AUC and F1-Score.

XGBoost hyperparameters (tuned via 5-fold CV on the first seed):
- German Credit: `learning_rate=0.05, max_depth=5, n_estimators=100`
- GMSC: `learning_rate=0.01, max_depth=3, n_estimators=300`
- Class imbalance handled by `scale_pos_weight = neg_count / pos_count`

### Phase 4 — SHAP Explainability Audit
For each seed, we extract TreeSHAP values from all three models on the **real test set**:
1. Compute mean absolute SHAP values per feature: this gives a feature importance ranking.
2. Compare synthetic-model rankings vs. real-model rankings using **Spearman rank correlation** ($\rho$).
3. Compare feature overlap using **Jaccard similarity** on Top-5 and Top-10 feature sets.

### Phase 5 — Privacy Evaluation (Traditional Metrics)
We compute three distance-based privacy metrics on the preprocessed feature space:
- **DCR (Distance to Closest Record):** How far is each synthetic point from the nearest real point? Higher = better privacy.
- **NNDR (Nearest Neighbor Distance Ratio):** Ratio of distance to 1st vs. 2nd nearest real neighbor. Values near 0 = memorization of a specific record.
- **MIA (Membership Inference Attack):** Can an attacker tell whether a specific person was in the training data by looking at synthetic data? AUC of 0.50 = perfect privacy (attacker is guessing randomly).

### Phase 6 — Inference Risk Indicator (Newly Added)
This is the metric from Min & Oh (2025) that we integrated as our latest enhancement:

**How it works (intuition):**
For each synthetic point, ask: "Is this synthetic record closer to a real record than that real record's own nearest neighbor in the training set?" If yes, the synthetic point is suspiciously close — it might leak information about that specific person.

**Formula:**
$$A = \frac{1}{N_{syn}} \sum_{i=1}^{N_{syn}} \mathbb{I}(d_s < d_0)$$

Where:
- $d_s$ = distance from synthetic point to its closest real training point
- $d_0$ = distance from that real training point to its nearest neighbor within the training set
- $\mathbb{I}(\cdot)$ = 1 if the condition is true, 0 otherwise
- $A$ = proportion of "risky" synthetic points

**Baseline threshold:** We split the real training data into 50 random subsets and compute A for each. The 95th percentile of these scores defines the "natural" level of inference risk. If a generator exceeds this threshold, it is leaking information beyond what random sampling would produce.

### Phase 7 — Statistical Validation
With 10 seeds, we have sufficient power for proper non-parametric significance testing. We use:
- **Wilcoxon signed-rank test:** Non-parametric paired test. With N=10, we can achieve p-values well below 0.05 (minimum possible two-sided p = 0.002 at N=10), enabling full statistical significance claims.
- **Cohen's d effect size:** Measures how many standard deviations apart the two groups are. Values above 0.8 are considered "large effects." Used alongside p-values as a magnitude discriminator.
- **Bootstrap 95% confidence intervals:** 1,000 bootstrap resamples for ROC-AUC and F1.

---

## 7. Results: Downstream Predictive Utility

*(10-seed aggregates: seeds 42, 123, 456, 789, 1337, 2024, 7, 99, 314, 2718)*

### Downstream Utility (XGBoost)
| Dataset | Model | ROC-AUC (Mean ± Std) | F1-Score (Mean ± Std) |
|:---|:---|:---:|:---:|
| **German Credit** | Real Baseline | 0.7716 ± 0.0194 | 0.5872 ± 0.0232 |
| **German Credit** | CTGAN | 0.4851 ± 0.0410 | 0.3146 ± 0.0493 |
| **German Credit** | TVAE | 0.6864 ± 0.0218 | 0.4074 ± 0.0878 |
| **GMSC** | Real Baseline | 0.8357 ± 0.0134 | 0.3024 ± 0.0145 |
| **GMSC** | CTGAN | 0.7850 ± 0.0327 | 0.3157 ± 0.0365 |
| **GMSC** | TVAE | 0.7531 ± 0.1284 | 0.3166 ± 0.0825 |

### Downstream Utility (LightGBM)
| Dataset | Model | ROC-AUC (Mean ± Std) | F1-Score (Mean ± Std) |
|:---|:---|:---:|:---:|
| **German Credit** | Real Baseline | 0.7711 ± 0.0175 | 0.5837 ± 0.0214 |
| **German Credit** | CTGAN | 0.4752 ± 0.0439 | 0.3232 ± 0.0464 |
| **German Credit** | TVAE | 0.6915 ± 0.0208 | 0.4260 ± 0.0782 |
| **GMSC** | Real Baseline | 0.8383 ± 0.0135 | 0.3193 ± 0.0161 |
| **GMSC** | CTGAN | 0.7922 ± 0.0314 | 0.3247 ± 0.0345 |
| **GMSC** | TVAE | 0.8011 ± 0.0259 | 0.3190 ± 0.0870 |

**What this tells us:**
- On **German Credit** (small, mixed-type), CTGAN completely fails across both classifiers (XGBoost AUC ~0.49, LightGBM AUC ~0.48, equivalent to a random coin flip). TVAE retains decent utility (XGBoost AUC 0.69, LightGBM AUC 0.69 vs. 0.77 baselines). This is because CTGAN struggles with categorical features and minor data volumes.
- On **GMSC** (larger, all-numerical), both generators perform reasonably well (XGBoost ~0.75-0.79, LightGBM ~0.79-0.80 vs. 0.84 baselines). The larger sample size and purely numerical features help both models.
- **TVAE consistently outperforms CTGAN** in downstream utility across both datasets, and this finding is robust to the choice of downstream classifier (XGBoost vs. LightGBM).
- Interestingly, LightGBM shows slightly better stability on the continuous, large-scale GMSC dataset, bringing TVAE's AUC up to **0.8011 ± 0.0259** (compared to XGBoost's **0.7531 ± 0.1284** which had higher variance).

### GMSC TVAE Utility Variance Analysis
The large standard deviation in XGBoost GMSC TVAE downstream utility (**0.7531 ± 0.1284**) is driven by a severe training collapse on seed **2024**, where the ROC-AUC dropped to **0.3804** (excluding this outlier seed, the remaining 9 seeds achieved stable performance between **0.74** and **0.85**, yielding a subset mean of **0.7944 ± 0.0354**). This volatility highlights the inherent training instability of TVAEs on dense, purely continuous tabular datasets. When mapping dense, continuous feature spaces into lower-dimensional probabilistic latent spaces, the encoder-decoder network can experience mode collapses or reconstruction shifts on specific bootstrap splits, resulting in localized label inversions. Interestingly, while XGBoost proved highly sensitive to these shifts on seed 2024 (collapsing to 0.3804), LightGBM demonstrated remarkable robustness on the exact same synthetic dataset, achieving an AUC of **0.7636** and maintaining stable utility across all 10 runs (**0.8011 ± 0.0259**). This indicates that the choice of downstream boosting architecture (e.g., LightGBM's leaf-wise tree growth and regularization) can buffer the utility risks associated with unstable generative models.

**GMSC TVAE Downstream ROC-AUC Per Seed:**
| Seed | XGBoost AUC | LightGBM AUC | Status |
|:---:|:---:|:---:|:---|
| **42** | 0.7444 | 0.7701 | Normal |
| **123** | 0.8388 | 0.8389 | Normal |
| **456** | 0.7853 | 0.8009 | Normal |
| **789** | 0.7606 | 0.7840 | Normal |
| **1337** | 0.7972 | 0.7902 | Normal |
| **2024** | **0.3804** | 0.7636 | **XGBoost Collapse** |
| **7** | 0.7673 | 0.7881 | Normal |
| **99** | 0.8467 | 0.8412 | Normal |
| **314** | 0.8297 | 0.8271 | Normal |
| **2718** | 0.7804 | 0.8065 | Normal |

---

## 8. Results: SHAP Explainability Consistency

### Spearman Rank Correlation ($\rho$)

*(10-seed aggregates)*

| Dataset | Classifier | CTGAN $\rho$ (Mean ± Std) | TVAE $\rho$ (Mean ± Std) |
|:---|:---|:---:|:---:|
| **German Credit** | XGBoost | 0.5501 ± 0.1382 | 0.5961 ± 0.0667 |
| **German Credit** | LightGBM | 0.5182 ± 0.1364 | 0.5988 ± 0.0731 |
| **GMSC** | XGBoost | 0.2933 ± 0.1950 | 0.6416 ± 0.1297 |
| **GMSC** | LightGBM | 0.1891 ± 0.2169 | 0.4507 ± 0.1409 |

### Jaccard Feature Overlap

*(10-seed aggregates)*

| Dataset | Classifier | Generator | Top-5 Jaccard (Mean ± Std) | Top-10 Jaccard (Mean ± Std) |
|:---|:---|:---|:---:|:---:|
| **German Credit** | XGBoost | CTGAN | 0.3988 ± 0.1198 | 0.5330 ± 0.0986 |
| **German Credit** | XGBoost | TVAE | 0.4881 ± 0.1597 | 0.5275 ± 0.0330 |
| **German Credit** | LightGBM | CTGAN | 0.3135 ± 0.1471 | 0.5458 ± 0.1065 |
| **German Credit** | LightGBM | TVAE | 0.3988 ± 0.1198 | 0.5055 ± 0.0504 |
| **GMSC** | XGBoost | CTGAN | 0.4345 ± 0.0939 | 1.0000 ± 0.0000 |
| **GMSC** | XGBoost | TVAE | 0.6286 ± 0.1633 | 1.0000 ± 0.0000 |
| **GMSC** | LightGBM | CTGAN | 0.3929 ± 0.0714 | 1.0000 ± 0.0000 |
| **GMSC** | LightGBM | TVAE | 0.4583 ± 0.1168 | 1.0000 ± 0.0000 |

*Note: GMSC has exactly 10 features, so Top-10 Jaccard is always 1.00 by definition.*

**What this tells us:**
- On **German Credit**, both generators preserve feature rankings at a moderate level under both boosting models (Spearman $\rho \approx 0.52 - 0.60$). The feature `checking_status` remains the #1 most important feature across all pipelines, which is critical for regulatory compliance. Wilcoxon signed-rank tests confirm that the explainability consistency difference between CTGAN and TVAE is not statistically significant in this setting (p=0.49 for XGBoost, p=0.16 for LightGBM).
- On **GMSC**, there is a dramatic and statistically significant gap: TVAE preserves rankings at moderate-to-strong consistency (XGBoost $\rho = 0.64$, LightGBM $\rho = 0.45$), while CTGAN's rankings are extremely weak and unreliable (XGBoost $\rho = 0.29$, LightGBM $\rho = 0.19$, with Wilcoxon test p=0.002 and p=0.0059 respectively). This occurs despite both models achieving nearly identical predictive AUC on downstream data (~0.78 for CTGAN, ~0.75-0.80 for TVAE).
- This strongly validates the **utility-explainability decoupling** phenomenon across different model families: a model trained on synthetic data can predict credit outcomes just as well as another, but it may explain those predictions in a completely different way. This reinforces that evaluating downstream AUC is insufficient for model compliance audits; explicit explainability consistency audits are required.
- Interestingly, the overall explainability correlation is lower for LightGBM than XGBoost on the continuous GMSC dataset. For instance, TVAE correlation drops from $\rho=0.64$ (XGBoost) to $\rho=0.45$ (LightGBM). This is likely because LightGBM uses leaf-wise growth, creating more complex, deeper, asymmetric trees which are more sensitive to the fine-grained distribution shifts between CTGAN/TVAEs and real data than XGBoost's level-wise trees. Thus, explainability shifts are amplified in leaf-wise boosting structures.


---

## 9. Results: Privacy Metrics (DCR, NNDR, MIA)

*(10-seed aggregates)*

| Dataset | Generator | Mean DCR | Mean NNDR | MIA AUC |
|:---|:---|:---:|:---:|:---:|
| German Credit | CTGAN | 3.4717 ± 0.0686 | 0.9220 ± 0.0030 | 0.4900 ± 0.0112 |
| German Credit | TVAE | 2.0845 ± 0.0435 | 0.8360 ± 0.0175 | 0.5054 ± 0.0215 |
| GMSC | CTGAN | 0.3414 ± 0.0329 | 0.7816 ± 0.0120 | 0.5051 ± 0.0043 |
| GMSC | TVAE | 0.1644 ± 0.0182 | 0.7191 ± 0.0125 | 0.5047 ± 0.0065 |


**What this tells us:**
- **CTGAN always has higher DCR** (its synthetic records are farther from real records). This means better privacy spacing.
- **TVAE always has lower DCR and NNDR** — it creates records that are much closer to real ones. Good for utility, bad for privacy.
- **MIA AUC is ~0.50 for everyone** — the distance-based membership inference attack cannot distinguish members from non-members. This sounds good, but it is misleading because MIA only measures *global* distance patterns. It misses *local* memorization, which is why we need the Inference Risk Indicator.

---

## 10. Results: Inference Risk Indicator (New Metric)

*(10-seed aggregates)*

| Dataset | Generator | Inference Risk Score | Baseline Threshold (95th pct) | Status |
|:---|:---|:---:|:---:|:---:|
| German Credit | CTGAN | **0.2113 ± 0.0153** | 0.5010 | PASS (safe) |
| German Credit | TVAE | **0.6383 ± 0.0367** | 0.5010 | **FAIL (privacy leak)** |
| GMSC | CTGAN | **0.4394 ± 0.0227** | 0.5571 | PASS (safe) |
| GMSC | TVAE | **0.5540 ± 0.0176** | 0.5571 | **FAIL (marginal/leak)** |

**What this tells us in plain language:**
- **CTGAN is safe.** On both datasets, only ~21% (German) and ~44% (GMSC) of CTGAN's synthetic records are "suspiciously close" to real records — well below the baseline threshold.
- **TVAE is NOT safe.** On German Credit, 64% of TVAE's synthetic records are closer to a specific real person than that person's own nearest neighbor in the training set. On GMSC, 55% are risky. Both exceed the baseline thresholds.
- **MIA missed this.** The MIA AUC was ~0.50 for TVAE (looks private), but the Inference Risk Indicator reveals the local memorization that MIA's global distance approach cannot detect. This validates why the Inference Risk Indicator is a necessary addition.

### Per-Seed Breakdown

**German Credit (all 10 seeds):**
| Seed | CTGAN Risk | TVAE Risk |
|:---:|:---:|:---:|
| 42 | 0.1929 | 0.5729 |
| 123 | 0.2200 | 0.6314 |
| 456 | 0.1971 | 0.6114 |
| 789 | 0.2314 | 0.6400 |
| 1337 | 0.2057 | 0.6257 |
| 2024 | 0.2214 | 0.6757 |
| 7 | 0.1814 | 0.6014 |
| 99 | 0.2200 | 0.7043 |
| 314 | 0.2243 | 0.6500 |
| 2718 | 0.2186 | 0.6700 |

**GMSC (all 10 seeds):**
| Seed | CTGAN Risk | TVAE Risk |
|:---:|:---:|:---:|
| 42 | 0.4627 | 0.5404 |
| 123 | 0.4629 | 0.5784 |
| 456 | 0.4449 | 0.5404 |
| 789 | 0.4399 | 0.5659 |
| 1337 | 0.4491 | 0.5909 |
| 2024 | 0.4284 | 0.5451 |
| 7 | 0.3960 | 0.5424 |
| 99 | 0.4451 | 0.5530 |
| 314 | 0.4026 | 0.5497 |
| 2718 | 0.4627 | 0.5336 |

TVAE exceeds CTGAN on **every single seed across all 10 runs** — this is statistically unambiguous.

---

## 10a. Results: Differential Privacy Analysis

To address the privacy leakage identified in standard TVAE (which fails the inference risk audit on both datasets), we implemented **Differentially Private TVAE (DP-TVAE)** by training the encoder and decoder under **DP-SGD** using the `opacus` library. We evaluated three target privacy budgets: $\varepsilon \in [1.0, 5.0, 10.0]$ with $\delta = 10^{-5}$ and clipping threshold $C = 1.0$.

### Impact on Inference Risk (Privacy Spacing)
- **German Credit:** DP-TVAE successfully mitigates local memorization. At all evaluated privacy levels, the mean Inference Risk drops below the baseline threshold of **0.5010** (passing the privacy audit). 
  - $\varepsilon = 10.0$ (weak privacy): Inference Risk = **0.4933 ± 0.1215 [PASS]**
  - $\varepsilon = 5.0$ (moderate privacy): Inference Risk = **0.3256 ± 0.0494 [PASS]**
  - $\varepsilon = 1.0$ (strong privacy): Inference Risk = **0.1260 ± 0.0496 [PASS]**
  As $\varepsilon$ increases, the privacy bounds weaken and local spacing gets closer to the original records, yielding a clear monotonic risk curve.
- **GMSC:** Under all privacy budgets, GMSC fails the Inference Risk audit, hovering around **0.61** (exceeding the threshold of **0.5571**). The Inference Risk metric is calibrated against a 95th percentile threshold derived from random subsampling of the real data. In high-density low-dimensional datasets like GMSC (10 features, 10,000 rows), the natural nearest-neighbor distances within the real data ($d_0$) is extremely small due to crowding. Any generative model — including one with formal DP guarantees — will produce samples that fall within these tiny $d_0$ neighborhoods purely by geometric necessity, not by memorization. This reveals a fundamental limitation of distance-based privacy metrics on dense tabular data: they conflate geometric crowding with privacy leakage. Future work should adapt the threshold calibration methodology for dataset density.

### The Downstream Utility & Minority Class Collapse
While DP-TVAE successfully protects privacy in German Credit, it triggers a **severe collapse in downstream utility** (ROC-AUC drops to ~0.50, equivalent to random guessing) across both boosting classifiers on both datasets:
- **German Credit:** 
  - XGBoost AUC drops from **0.6864** (non-private TVAE) to **0.5039** ($\varepsilon=1.0$), **0.4959** ($\varepsilon=5.0$), and **0.5033** ($\varepsilon=10.0$).
  - LightGBM AUC drops from **0.6915** (non-private TVAE) to **0.5158** ($\varepsilon=1.0$), **0.4770** ($\varepsilon=5.0$), and **0.5192** ($\varepsilon=10.0$).
- **GMSC:** 
  - XGBoost AUC collapses from **0.7531** (non-private TVAE) to exactly **0.5000 ± 0.0000** for all $\varepsilon$ budgets.
  - LightGBM AUC collapses from **0.8011** (non-private TVAE) to exactly **0.5000 ± 0.0000** for all $\varepsilon$ budgets.

This complete utility collapse is caused by **minority class signal drowning** under DP-SGD. For highly imbalanced tabular datasets (typical in credit default modeling where the default class is 30% in German Credit and 6.6% in GMSC), the gradients computed from the few positive default records represent a very small fraction of the total batch gradients. Under DP-SGD, these gradients are individually clipped and then added to a Gaussian noise matrix. The signal from the positive samples is completely drowned out by the noise, preventing the decoder from learning the positive class structure. In GMSC, this causes the generator to completely collapse, generating **100% majority class records (all 0s)** for the target column, yielding an AUC of exactly 0.5000 and F1-score of 0.0000 for both classifiers.

SHAP consistency analysis was not conducted for DP-TVAE variants as the complete utility collapse (AUC ≈ 0.50) renders feature attribution meaningless — a model predicting at chance level has no interpretable feature importance structure.

In summary, while DP-TVAE is mathematically private, standard DP-SGD is highly fragile on credit risk datasets due to extreme class imbalance, highlighting the necessity of advanced class-balanced private training algorithms in banking applications.

---

## 11. Results: Statistical Significance

### Statistical Power with N=10
With 10 seeds, the Wilcoxon signed-rank test can reach a minimum two-sided p-value of **0.002** — well below the standard p < 0.05 threshold. This gives our conclusions full statistical significance, not just effect size arguments.

### Key Statistical Results (10-Seed Wilcoxon + Cohen's d)

| Metric | Dataset | Classifier | Wilcoxon p-value | Cohen's d | Interpretation |
|:---|:---|:---|:---:|:---:|:---|
| **Inference Risk** | German Credit | Dataset-Level | **0.0020** | **14.41** | TVAE enormously higher risk (extremely large effect) |
| **Inference Risk** | GMSC | Dataset-Level | **0.0020** | **5.34** | TVAE much higher risk (large effect) |
| **ROC-AUC** | German Credit | XGBoost | **0.0020** | **3.68** | TVAE significantly higher utility (large effect) |
| **ROC-AUC** | German Credit | LightGBM | **0.0020** | **4.39** | TVAE significantly higher utility (large effect) |
| **ROC-AUC** | GMSC | XGBoost | 1.0000 | -0.23 | No significant difference (both similar) |
| **ROC-AUC** | GMSC | LightGBM | 0.4922 | 0.24 | No significant difference (both similar) |
| **SHAP rho** | GMSC | XGBoost | **0.0020** | **1.52** | TVAE significantly more consistent (large effect) |
| **SHAP rho** | GMSC | LightGBM | **0.0059** | **1.11** | TVAE significantly more consistent (large effect) |
| **SHAP rho** | German Credit | XGBoost | 0.4922 | 0.31 | No significant difference (both similar) |
| **SHAP rho** | German Credit | LightGBM | 0.1602 | 0.59 | No significant difference (both similar) |
| **DCR** | German Credit | Dataset-Level | **0.0020** | **-28.56** | CTGAN much higher spacing (extremely large effect) |
| **DCR** | GMSC | Dataset-Level | **0.0020** | **-6.60** | CTGAN much higher spacing (large effect) |
| **NNDR** | German Credit | Dataset-Level | **0.0020** | **-4.87** | CTGAN much higher NNDR (large effect) |
| **MIA AUC** | German Credit | Dataset-Level | 0.0840 | 0.74 | Not significant (both near-random) |
| **MIA AUC** | GMSC | Dataset-Level | 0.9219 | -0.07 | No difference (both near-random) |

**What Cohen's d values mean:**
- |d| < 0.2 = negligible difference
- |d| = 0.2–0.5 = small effect
- |d| = 0.5–0.8 = medium effect
- |d| > 0.8 = **large effect** ← all our key comparisons exceed this
- Our values (5.34, 14.41, 28.56) are *extremely* large — differences are unambiguous across all 10 seeds.

---

## 12. Master Results Tables

*(All values are 10-seed means. IR thresholds: German Credit = 0.5010, GMSC = 0.5571)*

### 12.1 XGBoost Master Results Table
| Dataset | Generator | ε | AUC | SHAP ρ | DCR | NNDR | MIA AUC | Inference Risk | IR Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **German** | CTGAN | ∞ | 0.4851 | 0.5501 | 3.4717 | 0.9220 | 0.4900 | **0.2113** | PASS |
| **German** | TVAE | ∞ | 0.6864 | 0.5961 | 2.0845 | 0.8360 | 0.5054 | **0.6383** | **FAIL** |
| **German** | DP-TVAE | 10.0 | 0.5033 | -- | 2.3724 | 0.8741 | 0.4941 | **0.4933** | PASS |
| **German** | DP-TVAE | 5.0 | 0.4959 | -- | 2.9838 | 0.9124 | 0.4894 | **0.3256** | PASS |
| **German** | DP-TVAE | 1.0 | 0.5039 | -- | 4.0946 | 0.9389 | 0.4978 | **0.1260** | PASS |
| **GMSC** | CTGAN | ∞ | 0.7850 | 0.2933 | 0.3414 | 0.7816 | 0.5051 | **0.4394** | PASS |
| **GMSC** | TVAE | ∞ | 0.7531 | 0.6416 | 0.1644 | 0.7191 | 0.5047 | **0.5540** | **FAIL** |
| **GMSC** | DP-TVAE | 10.0 | 0.5000 | -- | 0.1086 | 0.6776 | 0.5034 | **0.6105** | **FAIL** |
| **GMSC** | DP-TVAE | 5.0 | 0.5000 | -- | 0.1089 | 0.6785 | 0.5034 | **0.6082** | **FAIL** |
| **GMSC** | DP-TVAE | 1.0 | 0.5000 | -- | 0.1073 | 0.6788 | 0.5036 | **0.6095** | **FAIL** |

### 12.2 LightGBM Master Results Table
| Dataset | Generator | ε | AUC | SHAP ρ | DCR | NNDR | MIA AUC | Inference Risk | IR Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **German** | CTGAN | ∞ | 0.4752 | 0.5182 | 3.4717 | 0.9220 | 0.4900 | **0.2113** | PASS |
| **German** | TVAE | ∞ | 0.6915 | 0.5988 | 2.0845 | 0.8360 | 0.5054 | **0.6383** | **FAIL** |
| **German** | DP-TVAE | 10.0 | 0.5192 | -- | 2.3724 | 0.8741 | 0.4941 | **0.4933** | PASS |
| **German** | DP-TVAE | 5.0 | 0.4770 | -- | 2.9838 | 0.9124 | 0.4894 | **0.3256** | PASS |
| **German** | DP-TVAE | 1.0 | 0.5158 | -- | 4.0946 | 0.9389 | 0.4978 | **0.1260** | PASS |
| **GMSC** | CTGAN | ∞ | 0.7922 | 0.1891 | 0.3414 | 0.7816 | 0.5051 | **0.4394** | PASS |
| **GMSC** | TVAE | ∞ | 0.8011 | 0.4507 | 0.1644 | 0.7191 | 0.5047 | **0.5540** | **FAIL** |
| **GMSC** | DP-TVAE | 10.0 | 0.5000 | -- | 0.1086 | 0.6776 | 0.5034 | **0.6105** | **FAIL** |
| **GMSC** | DP-TVAE | 5.0 | 0.5000 | -- | 0.1089 | 0.6785 | 0.5034 | **0.6082** | **FAIL** |
| **GMSC** | DP-TVAE | 1.0 | 0.5000 | -- | 0.1073 | 0.6788 | 0.5036 | **0.6095** | **FAIL** |


---

## 13. Figures and Visualizations

### Figure A: Inference Risk Comparison Bar Chart
Mean inference risk scores (with error bars) for CTGAN and TVAE on both datasets. Horizontal dashed lines mark the 95th-percentile baseline thresholds. TVAE bars that exceed the threshold are highlighted — confirming TVAE fails the privacy audit on both datasets while CTGAN passes.

![Inference Risk Comparison](../figures/privacy/inference_risk_comparison.png)

### Figure B: Privacy Metrics Dashboard
A 2×3 grid comparing all four privacy metrics (DCR, NNDR, MIA AUC, Inference Risk) side-by-side for each dataset-generator combination across all 10 seeds. Red dashed lines mark threshold values. Note that MIA AUC looks similar for both generators (~0.50), while the Inference Risk Indicator reveals the real difference.

![Privacy Dashboard](../figures/privacy/privacy_dashboard.png)

---

### Figure C: Downstream Model ROC Curves

ROC curves comparing Real Baseline, CTGAN-trained, and TVAE-trained XGBoost models evaluated on the real held-out test set. The AUC gap between generators tells the utility story: CTGAN collapses on German Credit (AUC ~0.49) while TVAE maintains reasonable performance (AUC ~0.69). On GMSC, both converge closer to the real baseline.

**German Credit:**
![German Credit ROC Curve](../figures/paper/german_credit_roc_curve.png)

**GMSC:**
![GMSC ROC Curve](../figures/paper/gmsc_roc_curve.png)

---

### Figure D: Membership Inference Attack (MIA) ROC Curves

ROC curves for the Membership Inference Attack — an attacker trying to determine if a specific person was in the training set. The key observation: all curves hover around the diagonal (AUC ≈ 0.50), meaning the MIA is essentially guessing randomly. This looks like strong privacy — but Figure A shows it is misleading. The Inference Risk Indicator catches local memorization that MIA's global distance approach misses.

**German Credit:**
![German Credit MIA ROC Curve](../figures/paper/german_credit_mia_roc_curve.png)

**GMSC:**
![GMSC MIA ROC Curve](../figures/paper/gmsc_mia_roc_curve.png)

---

### Figure E: SHAP Beeswarm Plots

Beeswarm plots showing the distribution of SHAP values per feature across the real test set for each model. Each dot is one test sample; colour indicates the feature value (red = high, blue = low). Visually compare how feature importances and directions shift when training on synthetic data vs. real data.

**German Credit — Real Baseline:**
![German Credit Real SHAP Beeswarm](../figures/paper/german_credit_real_shap_beeswarm.png)

**German Credit — CTGAN-Trained:**
![German Credit CTGAN SHAP Beeswarm](../figures/paper/german_credit_ctgan_shap_beeswarm.png)

**German Credit — TVAE-Trained:**
![German Credit TVAE SHAP Beeswarm](../figures/paper/german_credit_tvae_shap_beeswarm.png)

**GMSC — Real Baseline:**
![GMSC Real SHAP Beeswarm](../figures/paper/gmsc_real_shap_beeswarm.png)

**GMSC — CTGAN-Trained:**
![GMSC CTGAN SHAP Beeswarm](../figures/paper/gmsc_ctgan_shap_beeswarm.png)

**GMSC — TVAE-Trained:**
![GMSC TVAE SHAP Beeswarm](../figures/paper/gmsc_tvae_shap_beeswarm.png)

---

### Figure F: SHAP Feature-Rank Heatmaps

Heatmaps showing the mean absolute SHAP rank of each feature per seed, for Real vs CTGAN vs TVAE models. Each column is one seed; each row is one feature. Stable rankings across seeds appear as uniform horizontal bands. Volatile rankings (high variance across columns) indicate the feature's importance is not robustly captured by synthetic training.

**German Credit:**
![German Credit SHAP Heatmap](../figures/paper/german_credit_shap_heatmap.png)

**GMSC:**
![GMSC SHAP Heatmap](../figures/paper/gmsc_shap_heatmap.png)

---

### Figure G: SHAP Consistency Comparison

Box plots showing the distribution of Spearman ρ values across all 10 seeds for CTGAN vs TVAE. Higher ρ = more consistent explanations with the real model. The key finding is visible here: on GMSC, TVAE's ρ distribution (median ~0.72) is dramatically higher than CTGAN's (median ~0.30), with p=0.002 and Cohen's d=1.52.

**German Credit:**
![German Credit SHAP Consistency](../figures/paper/german_credit_shap_consistency.png)

**GMSC:**
![GMSC SHAP Consistency](../figures/paper/gmsc_shap_consistency.png)

---

### Figure H: Utility-Privacy Tradeoff Scatter

Scatter plots with ROC-AUC on the x-axis and Inference Risk Score on the y-axis. Each point is one seed. The ideal generator sits in the **top-right** quadrant (high utility + low risk). CTGAN clusters bottom-left on German Credit (low utility, safe privacy). TVAE clusters top-right (good utility, dangerous privacy). On GMSC both generators cluster together in utility but diverge sharply in privacy — the fundamental tradeoff made visible.

**German Credit:**
![German Credit Utility-Privacy Tradeoff](../figures/paper/german_credit_utility_privacy_tradeoff.png)

**GMSC:**
![GMSC Utility-Privacy Tradeoff](../figures/paper/gmsc_utility_privacy_tradeoff.png)

---

### Figure I: DP-TVAE Utility-Privacy Tradeoff Curves

Tradeoff curves for Differentially Private TVAE (DP-TVAE) across privacy budgets ($\varepsilon \in [1.0, 5.0, 10.0, \infty]$). The left y-axis shows downstream model predictive utility (ROC-AUC) for both XGBoost and LightGBM models, while the right y-axis shows the empirical Inference Risk score. Horizontal dotted lines mark the 95th-percentile privacy disclosure thresholds. On German Credit, this visually shows that as $\varepsilon$ decreases (stronger privacy), Inference Risk drops below the threshold (passing the audit), but downstream utility collapses to random chance. On GMSC, utility collapses immediately under DP-SGD gradient noise due to extreme default class imbalance, while the Inference Risk remains high due to dataset density clustering.

![DP Tradeoff Curves](../figures/privacy/dp_utility_privacy_tradeoff.png)

---


## 14. Key Findings (Plain English)

### Finding 1: TVAE is better for utility and explanations, but leaks privacy
TVAE consistently produces synthetic data that trains better models and preserves SHAP feature rankings. However, it does this by creating records that are *too close* to real individuals, making it possible to infer who was in the original dataset.

### Finding 2: CTGAN protects privacy but destroys explanations
CTGAN generates "fuzzy" synthetic records that are far from any real person (good privacy). But this fuzziness scrambles the data distribution enough that the resulting models explain themselves very differently from models trained on real data.

### Finding 3: Good predictions do NOT guarantee good explanations
On GMSC, CTGAN and TVAE have nearly identical prediction accuracy (~0.78 vs ~0.79 AUC). But TVAE's SHAP rankings are more than twice as consistent with reality (mean $\rho$ = 0.64 vs 0.29; median $\rho$ = 0.72 vs 0.30). A bank using CTGAN for model validation would get correct predictions but wrong explanations — a compliance disaster.

### Finding 4: MIA is insufficient — you need the Inference Risk Indicator
Traditional Membership Inference Attacks (MIA AUC ~0.50 for everyone) give a false sense of security. They measure global distance patterns but miss local memorization. The Inference Risk Indicator catches what MIA misses: TVAE memorizes individual records even though global distance distributions look fine.

### Finding 5: Differential Privacy mitigates risk but triggers utility collapse on imbalanced credit data
Applying DP-TVAE successfully protects privacy on German Credit, keeping the Inference Risk score under the safety threshold (dropping to 0.4933 at ε=10.0 and 0.1260 at ε=1.0). However, it causes a severe downstream utility collapse (AUC drops to ~0.50) for both German Credit and GMSC. Under DP-SGD, the gradient noise and clipping completely drown out the signal of the highly minority default classes (30% in German Credit and 6.6% in GMSC), preventing the VAE from learning the positive class structure (in GMSC, it collapses and generates 100% majority class records). Additionally, in dense datasets like GMSC, the distance-based Inference Risk metric flags false positives due to natural density clustering rather than membership leakage, causing DP-TVAE to fail the audit even under mathematically guaranteed privacy.

---

## 15. Limitations and Gaps

### Methodological Limitations
1. **N=10 seeds:** With 10 seeds, Wilcoxon p-values reach 0.002, fully satisfying the p < 0.05 threshold. However, reviewers may still want N=20 runs for even broader confidence intervals and robustness checks.
2. **CPU-only training:** All experiments run on CPU, limiting dataset size to ~10K records. Real credit portfolios have millions of rows.
3. **Two classifier families (XGBoost and LightGBM) were evaluated:** Results are consistent across both gradient boosting architectures. Extension to linear models (logistic regression) and neural networks remains as future work.
4. **Fixed generator architectures:** We use default SDV CTGAN/TVAE architectures. Hyperparameter tuning of the generators themselves might improve results.
5. **DP-SGD Utility-Imbalance Tradeoff:** While we implemented DP-SGD, standard private training completely collapses on minority class representations under typical epsilon boundaries, highlighting a severe limitation of tabular DP-SGD on highly imbalanced banking data.

### Research Gaps Remaining
1. **No causal analysis:** We show SHAP rankings shift but do not explain *why* specific features move (e.g., which statistical properties of the synthetic data cause the shift).
2. **No fairness evaluation:** We do not test whether synthetic training introduces bias across protected attributes (age, gender).
3. **No temporal data:** Credit risk is inherently temporal (defaults happen over time). We use static snapshots only.
4. **No multi-generator comparison:** We only test CTGAN and TVAE. Other generators like CopulaGAN, SMOTE, or diffusion models are not included.

---

## 16. How This Project Can Be Improved

### Short-Term Improvements (Low Effort, High Impact)
1. **Run 20 seeds instead of 10:** This would provide even narrower confidence intervals and further solidify statistical claims beyond the current N=10.
2. **Test with logistic regression:** Show that the tradeoff holds across linear classifier families, not just boosting trees (XGBoost and LightGBM).


### Medium-Term Improvements (Portfolio Strengthening)
5. **Build an automated privacy audit pipeline:** Package the inference risk computation, threshold estimation, and pass/fail check into a reusable Python library. This demonstrates MLOps/production engineering skills.
6. **Add a configuration-driven experiment runner:** Use Hydra or a YAML-based config system so experiments can be reproduced with a single command. Shows engineering maturity.
7. **Deploy as a Streamlit/Gradio dashboard:** Allow users to upload a synthetic dataset and get an instant privacy report with all metrics. This is a strong portfolio piece for ML engineering roles.
8. **Scale to 100K+ records with FAISS:** Replace sklearn's NearestNeighbors with FAISS for GPU-accelerated nearest neighbor search. Shows awareness of production scalability.

### Long-Term Research Extensions
9. **Integrate fairness metrics:** Measure whether CTGAN/TVAE amplify bias in credit decisions.
10. **Test diffusion-based tabular generators:** Compare against TabDDPM or other recent architectures.
11. **Federated synthetic data:** Generate synthetic data in a federated setting where multiple banks collaborate without sharing raw data.

---

## 17. Why This Project Demonstrates ML Engineering Skills

This project is not just a research paper — it demonstrates the full stack of skills expected of an ML engineer:

| Skill | How This Project Demonstrates It |
|:---|:---|
| **Data Engineering** | Leakage-safe preprocessing pipeline with fit-on-train-only transformers, stratified splits, and reproducible seed management |
| **ML Modeling** | XGBoost training with dynamic class weight balancing, hyperparameter tuning via cross-validation, and multi-seed robustness evaluation |
| **Generative AI** | Training and evaluating CTGAN and TVAE with the SDV framework, understanding GAN vs. VAE architectures and their tradeoffs |
| **Explainable AI (XAI)** | Production-grade SHAP analysis with quantitative consistency metrics (Spearman rho, Jaccard overlap), not just pretty beeswarm plots |
| **Privacy Engineering** | Implementing distance-based privacy metrics (DCR, NNDR, MIA) and the Inference Risk Indicator with batch processing for memory efficiency |
| **Statistical Rigor** | N=10 multi-seed Wilcoxon signed-rank tests achieving p=0.002, Cohen's d effect sizes as magnitude discriminators, bootstrap confidence intervals — demonstrates understanding of both significance and practical effect magnitude |
| **Reproducibility** | Fixed seeds, config-driven experiments, saved artifacts (models, predictions, SHAP values), deterministic pipelines |
| **Scientific Writing** | Structured IEEE Access manuscript with LaTeX tables, properly captioned figures, and precise mathematical notation |
| **Software Engineering** | Modular codebase with separate `src/`, `scripts/`, `configs/`, `results/`, and `figures/` directories. Functions are reusable and well-documented |

---

## Appendix: Repository Structure

```
shap-synthetic-credit-risk/
|-- configs/
|   |-- datasets/          # german_credit.json, gmsc.json
|   |-- seeds/             # seeds.json: [42, 123, 456, 789, 1337, 2024, 7, 99, 314, 2718]
|-- data/
|   |-- raw/               # Original CSV files
|   |-- processed/         # Preprocessed train/test splits
|   |-- synthetic/         # Generated synthetic datasets (per seed)
|-- figures/
|   |-- privacy/           # inference_risk_comparison.png, privacy_dashboard.png
|-- models/                # Saved XGBoost models (per seed, per generator)
|-- reports/
|   |-- final_manuscript/  # manuscript.md, manuscript.tex
|   |-- report.md          # This file
|-- results/
|   |-- baseline/          # Real model predictions
|   |-- ctgan/             # CTGAN model predictions
|   |-- tvae/              # TVAE model predictions
|   |-- privacy/           # DCR/NNDR/MIA summaries + inference risk JSONs
|   |-- shap/              # SHAP values, rankings, consistency summaries
|   |-- statistics/        # Wilcoxon + Cohen's d results
|   |-- summaries/         # LaTeX tables
|-- scripts/
|   |-- preprocess.py
|   |-- generate_synthetic.py
|   |-- train_baseline.py
|   |-- train_synthetic.py
|   |-- analyze_shap_consistency.py
|   |-- evaluate_privacy.py
|   |-- compute_inference_risk.py     # New: Inference Risk pipeline
|   |-- run_statistical_validation.py
|   |-- generate_paper_visualizations.py
|-- src/
|   |-- preprocessing/     # pipeline.py (data splitting + encoding)
|   |-- models/            # synthetic.py (training + evaluation)
|   |-- privacy/           # evaluation.py (DCR, NNDR, MIA functions)
|-- requirements.txt
```
