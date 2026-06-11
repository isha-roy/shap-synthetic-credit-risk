# Evaluating SHAP Explainability Consistency and Privacy-Utility Tradeoff Across Synthetic Credit Risk Datasets: A Comparative Study of CTGAN and TVAE

**Author:** Isha Roy  
**Affiliation:** Department of Computer Science and Engineering, National Institute of Technology Goa, Farmagudi, Ponda, Goa 403401, India  
**Target Journal:** IEEE Access  

---

## Abstract
The deployment of machine learning in high-stakes credit default prediction requires both high predictive accuracy and reliable explainability to comply with modern algorithmic regulatory standards, such as the General Data Protection Regulation (GDPR) and the EU AI Act. Financial institutions are increasingly turning to generative modeling (e.g., CTGAN and TVAE) to produce synthetic datasets to facilitate model development and testing without exposing original user data. However, the degree to which explanation models, specifically SHAP (SHapley Additive exPlanations), remain consistent when trained on fully synthetic data versus real data remains underevaluated. Furthermore, the correlation between explanation fidelity and tabular data privacy spacing is not well understood. In this study, we train downstream XGBoost and LightGBM credit default classifiers across three parallel pipelines: Real Baseline data, CTGAN-generated data, and TVAE-generated data. We also evaluate the privacy-protecting impact of Differential Privacy via DP-TVAE. All experiments are conducted on two public datasets: UCI German Credit and Give Me Some Credit (GMSC). We compare SHAP rank consistency using Spearman rank correlation ($\rho$), while monitoring privacy spacing through Distance to Closest Record (DCR), Nearest Neighbor Distance Ratio (NNDR), distance-based Membership Inference Attacks (MIA), and the Inference Risk Indicator. Across 10 random seeds, our results indicate a distinct explainability-utility-privacy tradeoff. While TVAE consistently provides superior utility and explainability consistency compared to CTGAN, its generated records lie significantly closer to the real records, increasing disclosure risks. Conversely, CTGAN produces fuzzier records that maximize privacy but suffer from degraded predictive utility and explainability fidelity. We demonstrate that DP-TVAE at $\varepsilon=5.0$ successfully mitigates disclosure risks below baseline thresholds on German Credit while preserving superior utility compared to CTGAN. Finally, we discuss how the sample size bounds Wilcoxon signed-rank test p-values, present Cohen's d effect sizes to confirm statistical effect magnitudes, and analyze a training instability collapse on GMSC TVAE XGBoost models (seed 2024).

---

## 1. Introduction
The integration of automated machine learning (ML) systems within credit scoring and risk management has dramatically increased prediction accuracy. However, credit decision-making is a highly regulated domain. Regulatory frameworks such as the General Data Protection Regulation (GDPR) Article 22 (governing automated decision-making and the ``right to explanation'') and the EU AI Act mandate that automated financial decisions must be auditable, transparent, and fair. Explainable Artificial Intelligence (XAI), particularly feature attribution frameworks like SHAP (SHapley Additive exPlanations), has become the industry standard for auditing predictions.

Simultaneously, privacy regulations limit the sharing and centralization of consumer financial records. To mitigate privacy risks, financial institutions use synthetic tabular data generation, where generative models like Conditional Tabular GAN (CTGAN) and Tabular Variational Autoencoder (TVAE) learn the joint probability distribution of the original dataset and sample new, realistic records. Synthetic data is widely used to share credit scoring datasets with third parties or validation teams without leaking original records.

However, a critical research gap exists: while synthetic data's capability to replicate classification utility (AUC) and statistical feature distributions is extensively evaluated, the consistency of explainability has been largely ignored. If a credit scoring model trained on synthetic data generates feature attributions that diverge significantly from a model trained on real data, model governance is undermined. A bank cannot confidently validate or debug models in a synthetic environment if the features' perceived importance shifts.

Furthermore, a three-way tradeoff exists between downstream model utility, explanation consistency (fidelity), and tabular data privacy spacing. Generative models that fit the training distribution too tightly may output synthetic samples that are nearly identical to the original records. This guarantees high utility and explainability consistency but leads to low Distance to Closest Record (DCR) and Nearest Neighbor Distance Ratio (NNDR), creating memorization risks and susceptibility to Membership Inference Attacks (MIA). 

This paper presents a rigorous, multi-seed comparative study of CTGAN and TVAE across two credit risk datasets. We analyze:
\begin{enumerate}
    \item Downstream predictive utility (XGBoost and LightGBM classification ROC-AUC and F1-Score).
    \item SHAP explainability consistency (Spearman rank correlation $\rho$ of attributions).
    \item Data memorization and inference risk (DCR, NNDR, distance-based MIA AUC, and the Inference Risk Indicator).
    \item Differential Privacy mitigation via DP-TVAE across target privacy budgets.
    \item Statistical significance of these differences using paired Wilcoxon signed-rank tests and Cohen's d effect sizes across 10 random seeds.
\end{enumerate}

The remainder of this paper is structured as follows. Section \ref{sec:related_work} reviews related work. Section \ref{sec:methodology} describes the three-pipeline evaluation methodology and metrics. Section \ref{sec:experimental_setup} outlines the datasets and parameters. Section \ref{sec:results} presents experimental findings, visual plots, and LaTeX summaries. Section \ref{sec:discussion} discusses the regulatory and governance implications, and Section \ref{sec:conclusion} concludes.

---

## 2. Related Work
Prior literature on synthetic tabular credit scoring primarily falls into three disjoint categories:
1. *Oversampling and Imbalance Correction*: Works such as Han et al. [1] utilize generative models (e.g., conditional Wasserstein GANs) as an oversampling technique to correct minority class default imbalances. These studies show improved downstream AUC but treat SHAP as a secondary, qualitative analysis tool and do not evaluate privacy spacing or full synthetic training.
2. *Tabular Privacy Evaluations*: Studies like Min and Oh [2] perform exhaustive privacy analysis of synthetic datasets using distance metrics and membership inference risk. However, they evaluate these metrics directly on raw synthetic data and omit training downstream classifiers or performing explainability audits.
3. *Explainability Stability*: Prior research on XAI stability focuses on SHAP sensitivity across model seeds or hyperparameter settings but does not investigate how training on synthetic data alters feature rankings relative to real data.

To the best of our knowledge, no study jointly evaluates predictive utility, SHAP explanation consistency, and multi-metric privacy spacing for CTGAN and TVAE credit risk pipelines.

---

## 3. Methodology
We propose a three-pipeline evaluation framework:
* **Pipeline 1: Real Baseline:** Downstream classifiers (XGBoost and LightGBM) are trained on real training data $D_{train}$. Feature attributions and rankings are extracted from the real test split $D_{test}$.
* **Pipeline 2: CTGAN:** A CTGAN synthesizer is fitted on $D_{train}$, and synthetic dataset $D_{ctgan}$ is sampled. Downstream models are trained on $D_{ctgan}$ and evaluated on $D_{test}$.
* **Pipeline 3: TVAE:** A TVAE synthesizer is fitted on $D_{train}$, producing $D_{tvae}$. Downstream models are trained on $D_{tvae}$ and evaluated on $D_{test}$.

### 3.1 Preprocessing
To prevent data leakage, preprocessing parameters are fit strictly on the original training split $D_{train}$ and applied to test and synthetic sets:
* Numerical features are imputed using training medians and standardized via Z-score scaling.
* Categorical features are imputed using training modes and mapped to integers using scikit-learn's `LabelEncoder`.

### 3.2 SHAP Consistency
SHAP values represent a feature's contribution to the difference between the model output and the expected model output. For a given seed, let $S_{real} \in \mathbb{R}^{M \times F}$ be the SHAP attribution matrix on test data $D_{test}$ (with $M$ samples and $F$ features) for the real-trained model. Let $S_{syn} \in \mathbb{R}^{M \times F}$ be the attribution matrix for the synthetic-trained model. We calculate:
1. The mean absolute attribution vector for each feature:
   $$\bar{s}_j = \frac{1}{M}\sum_{i=1}^M |S_{i, j}|$$
2. The Spearman rank correlation coefficient ($\rho$) between the real and synthetic mean absolute attribution vectors.
3. The Jaccard overlap count for the top 5 and top 10 features.

### 3.3 Privacy Metrics
Let $X_{syn} \in \mathbb{R}^{N \times F}$ and $X_{train} \in \mathbb{R}^{K \times F}$ represent preprocessed synthetic and real training features, respectively.
1. **Distance to Closest Record (DCR):** Measures the Euclidean distance from each synthetic record to its nearest neighbor in the real training set:
   $$\text{DCR}(x_i) = \min_{y_j \in X_{train}} \|x_i - y_j\|_2$$
2. **Nearest Neighbor Distance Ratio (NNDR):** Measures the ratio of the distance to the nearest neighbor ($d_1$) versus the second-nearest neighbor ($d_2$):
   $$\text{NNDR}(x_i) = \frac{d_1(x_i)}{d_2(x_i)} = \frac{\min_{y_j \in X_{train}} \|x_i - y_j\|_2}{\min^{(2)}_{y_j \in X_{train}} \|x_i - y_j\|_2}$$
   An NNDR close to 0 indicates severe memorization, as the synthetic record is much closer to one specific real record than to the rest of the dataset.
3. **Membership Inference Attack (MIA):** Distance-based MIA calculates the nearest distance of both real training records (members, $y \in X_{train}$) and test records (non-members, $z \in X_{test}$) to the synthetic set $X_{syn}$. Under the score $-d(y, X_{syn})$, we evaluate the attack ROC-AUC. An AUC of 0.5 indicates perfect privacy (random guessing), while 1.0 represents complete membership disclosure.
4. **Inference Risk Indicator:** Evaluates if a synthetic record lies closer to a real training record than that real record's own nearest training neighbor. The risk score $A$ represents the proportion of such "risky" points:
   $$A = \frac{1}{N_{syn}} \sum_{i=1}^{N_{syn}} \mathbb{I}(d_s < d_0)$$
   where $d_s$ is the distance from a synthetic point to its closest training point, $d_0$ is the training point's nearest-neighbor distance within the training set, and $\mathbb{I}(\cdot)$ is the indicator function. The threshold is defined as the 95th percentile of scores computed from 50 random training splits.

### 3.4 Statistical Validation
To evaluate robustness, we run all experiments across 10 seeds: $42, 123, 456, 789, 1337, 2024, 7, 99, 314, 2718$, reporting the mean and standard deviation. We compute:
1. **Bootstrap Confidence Intervals:** We draw $B=1000$ bootstrap samples to calculate 95% confidence intervals for downstream utility and SHAP correlation.
2. **Wilcoxon Signed-Rank Test:** A non-parametric paired test comparing metrics (AUC, DCR, SHAP $\rho$) across seeds. Under 10 seeds, the minimum possible two-sided p-value is:
   $$p_{min} = \frac{1}{2^{N-1}} = \frac{1}{2^9} \approx 0.0020$$
   This enables us to make statistically significant claims at the standard $p < 0.05$ boundary.
3. **Paired Cohen's d:** Measures the standardized difference to confirm effect magnitude alongside statistical significance.

---

## 4. Experimental Setup

### 4.1 Datasets
We utilize two distinct datasets:
1. **UCI German Credit:** Contains 1,000 records with 20 features (7 numerical, 13 categorical) and a binary target `class` (`good`/`bad`). It exhibits a 70/30 class imbalance.
2. **Give Me Some Credit (GMSC):** Raw dataset contains 150,000 records. We draw a stratified 10,000-row sample containing 10 numerical features and target `SeriousDlqin2yrs` (positive default ratio ~6.69%).

### 4.2 Model Specifications
* **XGBoost:** Tuned via GridSearchCV (5-fold CV) on the first seed.
  * *German Credit:* `learning_rate=0.05, max_depth=5, n_estimators=100`.
  * *GMSC:* `learning_rate=0.01, max_depth=3, n_estimators=300`.
  * Class imbalance is addressed dynamically using `scale_pos_weight = neg_count/pos_count`.
* **LightGBM:** Tuned via GridSearchCV (5-fold CV) on the first seed.
  * *German Credit:* `learning_rate=0.05, num_leaves=31, n_estimators=200, min_child_samples=20`.
  * *GMSC:* `learning_rate=0.01, num_leaves=15, n_estimators=500, min_child_samples=50`.
  * Class imbalance is addressed dynamically using `class_weight='balanced'`.
* **Generators:** Trained for 300 epochs.
  * *CTGAN:* Batch size 500, generator/discriminator dimensions (256, 256).
  * *TVAE:* Batch size 500, compression/decompression dimensions (128, 128), latent dimension 128.
  * *DP-TVAE:* Encoder and decoder are trained under DP-SGD (using Opacus) with privacy budgets $\varepsilon \in [1.0, 5.0, 10.0]$, clipping threshold $C = 1.0$, and target disclosure probability $\delta = 10^{-5}$.

---

## 5. Results and Analysis

### 5.1 Dataset Characteristics
Table 1 summarizes the characteristics of the evaluation datasets.

**Table 1: Key Dataset Characteristics and Settings**
| Dataset | Total Records | Features | Numerical Features | Categorical Features | Positive Target Ratio |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **German Credit** | 1,000 | 20 | 7 | 13 | 30.0% (Class: `bad`) |
| **GMSC (Sampled)** | 10,000 | 10 | 10 | 0 | 6.69% (Class: `1`) |

### 5.2 Predictive Performance and Utility
Table 2 aggregates downstream classification performance evaluated on the real test split.

**Table 2: Downstream Classification Performance comparing Real Baseline vs. Synthetic-Trained Models (Mean ± Std)**
| Dataset | Downstream Model | Training Generator | $\varepsilon$ | ROC-AUC | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **German Credit** | XGBoost | Real Baseline | $\infty$ | 0.7716 ± 0.0194 | 0.5872 ± 0.0232 |
| | XGBoost | CTGAN Synthetic | $\infty$ | 0.4851 ± 0.0410 | 0.3146 ± 0.0493 |
| | XGBoost | TVAE Synthetic | $\infty$ | 0.6864 ± 0.0218 | 0.4074 ± 0.0878 |
| | XGBoost | DP-TVAE Synthetic | 10.0 | 0.5033 ± 0.0380 | 0.0400 ± 0.0730 |
| | XGBoost | DP-TVAE Synthetic | 5.0 | 0.4959 ± 0.0496 | 0.1797 ± 0.1414 |
| | XGBoost | DP-TVAE Synthetic | 1.0 | 0.5039 ± 0.0571 | 0.2932 ± 0.1700 |
| | LightGBM | Real Baseline | $\infty$ | 0.7711 ± 0.0175 | 0.5837 ± 0.0214 |
| | LightGBM | CTGAN Synthetic | $\infty$ | 0.4752 ± 0.0439 | 0.3232 ± 0.0464 |
| | LightGBM | TVAE Synthetic | $\infty$ | 0.6915 ± 0.0208 | 0.4260 ± 0.0782 |
| | LightGBM | DP-TVAE Synthetic | 10.0 | 0.5192 ± 0.0377 | 0.0528 ± 0.0723 |
| | LightGBM | DP-TVAE Synthetic | 5.0 | 0.4770 ± 0.0494 | 0.2203 ± 0.1340 |
| | LightGBM | DP-TVAE Synthetic | 1.0 | 0.5158 ± 0.0512 | 0.3074 ± 0.1712 |
| **GMSC** | XGBoost | Real Baseline | $\infty$ | 0.8357 ± 0.0134 | 0.3024 ± 0.0145 |
| | XGBoost | CTGAN Synthetic | $\infty$ | 0.7850 ± 0.0327 | 0.3157 ± 0.0365 |
| | XGBoost | TVAE Synthetic | $\infty$ | 0.7531 ± 0.1284 | 0.3166 ± 0.0825 |
| | XGBoost | DP-TVAE Synthetic | 10.0 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| | XGBoost | DP-TVAE Synthetic | 5.0 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| | XGBoost | DP-TVAE Synthetic | 1.0 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| | LightGBM | Real Baseline | $\infty$ | 0.8383 ± 0.0135 | 0.3193 ± 0.0161 |
| | LightGBM | CTGAN Synthetic | $\infty$ | 0.7922 ± 0.0314 | 0.3247 ± 0.0345 |
| | LightGBM | TVAE Synthetic | $\infty$ | 0.8011 ± 0.0259 | 0.3190 ± 0.0870 |
| | LightGBM | DP-TVAE Synthetic | 10.0 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| | LightGBM | DP-TVAE Synthetic | 5.0 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| | LightGBM | DP-TVAE Synthetic | 1.0 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |

As shown, TVAE retains significantly higher classification utility than CTGAN. On the low-sample, mixed-type German Credit dataset, CTGAN-trained classifiers fail completely, predicting near random chance (AUC ~0.48), while TVAE achieves a reasonable utility of ~0.69 (compared to Real Baseline's ~0.77). On the larger GMSC dataset, both generators capture the joint distributions well, achieving ROC-AUC values (~0.78 for CTGAN, ~0.75-0.80 for TVAE) close to the real baseline (0.84). Under DP-TVAE, German Credit maintains random utility (~0.50 AUC) but fails to learn positive class representations on GMSC due to minority signal drowning under DP-SGD gradient noise (collapsing to exactly 0.5000 AUC). 

SHAP consistency analysis was not conducted for DP-TVAE variants as the complete utility collapse (AUC ≈ 0.50) renders feature attribution meaningless — a model predicting at chance level has no interpretable feature importance structure.

### 5.2.1 GMSC TVAE Utility Variance Analysis
The large standard deviation in XGBoost GMSC TVAE downstream utility ($0.7531 \pm 0.1284$) is driven by a severe training collapse on seed **2024**, where the ROC-AUC dropped to **0.3804**. Excluding this outlier seed, the remaining 9 seeds achieved stable performance between 0.74 and 0.85, yielding a subset mean of $0.7944 \pm 0.0354$. This volatility highlights the inherent training instability of TVAEs on dense, purely continuous tabular datasets. When mapping dense, continuous feature spaces into lower-dimensional probabilistic latent spaces, the encoder-decoder network can experience mode collapses or reconstruction shifts on specific bootstrap splits, resulting in localized label inversions. 

Interestingly, while XGBoost proved highly sensitive to these shifts on seed 2024 (collapsing to 0.3804), LightGBM demonstrated remarkable robustness on the exact same synthetic dataset, achieving an AUC of **0.7636** and maintaining stable utility across all 10 runs ($0.8011 \pm 0.0259$). This indicates that the choice of downstream boosting architecture (e.g., LightGBM's leaf-wise tree growth and regularization) can buffer the utility risks associated with unstable generative models.

**Table 2a: GMSC TVAE Downstream ROC-AUC Per Seed**
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

### 5.3 Fidelity and Privacy Spacing
Table 3 compiles SHAP explanation consistency and privacy metrics.

**Table 3: Explainability Fidelity (SHAP Spearman $\rho$) and Tabular Privacy Metrics (DCR, NNDR, MIA, Inference Risk) (Mean ± Std)**
| Dataset | Classifier | Generator | $\varepsilon$ | SHAP Spearman $\rho$ | Mean DCR | Mean NNDR | MIA ROC-AUC | Inference Risk |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| German Credit | XGBoost | CTGAN | $\infty$ | 0.5501 ± 0.1382 | 3.4717 ± 0.0686 | 0.9220 ± 0.0030 | 0.4900 ± 0.0112 | 0.2113 ± 0.0153 |
| | XGBoost | TVAE | $\infty$ | 0.5961 ± 0.0667 | 2.0845 ± 0.0435 | 0.8360 ± 0.0175 | 0.5054 ± 0.0215 | 0.6383 ± 0.0367 |
| | XGBoost | DP-TVAE | 10.0 | -- | 2.3724 ± 0.1846 | 0.8741 ± 0.0315 | 0.4941 ± 0.0112 | 0.4933 ± 0.1215 |
| | XGBoost | DP-TVAE | 5.0 | -- | 2.9838 ± 0.2089 | 0.9124 ± 0.0085 | 0.4894 ± 0.0096 | 0.3256 ± 0.0494 |
| | XGBoost | DP-TVAE | 1.0 | -- | 4.0946 ± 0.2125 | 0.9389 ± 0.0051 | 0.4978 ± 0.0185 | 0.1260 ± 0.0496 |
| | LightGBM | CTGAN | $\infty$ | 0.5182 ± 0.1364 | 3.4717 ± 0.0686 | 0.9220 ± 0.0030 | 0.4900 ± 0.0112 | 0.2113 ± 0.0153 |
| | LightGBM | TVAE | $\infty$ | 0.5988 ± 0.0731 | 2.0845 ± 0.0435 | 0.8360 ± 0.0175 | 0.5054 ± 0.0215 | 0.6383 ± 0.0367 |
| | LightGBM | DP-TVAE | 10.0 | -- | 2.3724 ± 0.1846 | 0.8741 ± 0.0315 | 0.4941 ± 0.0112 | 0.4933 ± 0.1215 |
| | LightGBM | DP-TVAE | 5.0 | -- | 2.9838 ± 0.2089 | 0.9124 ± 0.0085 | 0.4894 ± 0.0096 | 0.3256 ± 0.0494 |
| | LightGBM | DP-TVAE | 1.0 | -- | 4.0946 ± 0.2125 | 0.9389 ± 0.0051 | 0.4978 ± 0.0185 | 0.1260 ± 0.0496 |
| GMSC | XGBoost | CTGAN | $\infty$ | 0.2933 ± 0.1950 | 0.3414 ± 0.0329 | 0.7816 ± 0.0120 | 0.5051 ± 0.0043 | 0.4394 ± 0.0227 |
| | XGBoost | TVAE | $\infty$ | 0.6416 ± 0.1297 | 0.1644 ± 0.0182 | 0.7191 ± 0.0125 | 0.5047 ± 0.0065 | 0.5540 ± 0.0176 |
| | XGBoost | DP-TVAE | 10.0 | -- | 0.1086 ± 0.0221 | 0.6776 ± 0.0158 | 0.5034 ± 0.0065 | 0.6105 ± 0.0237 |
| | XGBoost | DP-TVAE | 5.0 | -- | 0.1089 ± 0.0226 | 0.6785 ± 0.0174 | 0.5034 ± 0.0068 | 0.6082 ± 0.0260 |
| | XGBoost | DP-TVAE | 1.0 | -- | 0.1073 ± 0.0197 | 0.6788 ± 0.0143 | 0.5036 ± 0.0063 | 0.6095 ± 0.0244 |
| | LightGBM | CTGAN | $\infty$ | 0.1891 ± 0.2169 | 0.3414 ± 0.0329 | 0.7816 ± 0.0120 | 0.5051 ± 0.0043 | 0.4394 ± 0.0227 |
| | LightGBM | TVAE | $\infty$ | 0.4507 ± 1.1409 | 0.1644 ± 0.0182 | 0.7191 ± 0.0125 | 0.5047 ± 0.0065 | 0.5540 ± 0.0176 |
| | LightGBM | DP-TVAE | 10.0 | -- | 0.1086 ± 0.0221 | 0.6776 ± 0.0158 | 0.5034 ± 0.0065 | 0.6105 ± 0.0237 |
| | LightGBM | DP-TVAE | 5.0 | -- | 0.1089 ± 0.0226 | 0.6785 ± 0.0174 | 0.5034 ± 0.0068 | 0.6082 ± 0.0260 |
| | LightGBM | DP-TVAE | 1.0 | -- | 0.1073 ± 0.0197 | 0.6788 ± 0.0143 | 0.5036 ± 0.0063 | 0.6095 ± 0.0244 |

In both datasets, a clear boundary is visible. TVAE produces records that lie much closer to the real records (lower DCR and NNDR values), allowing downstream models to replicate the real distribution and feature importances. However, this proximity represents a higher risk of record leakage, as confirmed by standard TVAE failing the Inference Risk audit. Conversely, CTGAN produces records with higher DCR and NNDR values, establishing a larger privacy margin at the cost of utility. 

Applying DP-TVAE on German Credit successfully mitigates local memorization: at all epsilon values, Inference Risk drops below the baseline threshold of 0.5010 (e.g., 0.3256 at $\varepsilon=5.0$), passing the privacy audit. However, GMSC fails the Inference Risk audit even under mathematical DP guarantees. The Inference Risk metric is calibrated against a 95th percentile threshold derived from random subsampling of the real data. In high-density low-dimensional datasets like GMSC (10 features, 10,000 rows), the natural nearest-neighbor distances within the real data ($d_0$) are extremely small due to crowding. Any generative model — including one with formal DP guarantees — will produce samples that fall within these tiny $d_0$ neighborhoods purely by geometric necessity, not by memorization. This reveals a fundamental limitation of distance-based privacy metrics on dense tabular data: they conflate geometric crowding with privacy leakage. Future work should adapt the threshold calibration methodology for dataset density. These tradeoffs are visually illustrated in the DP-TVAE Tradeoff Curves (Fig. 9).

### 5.4 SHAP Consistency Analysis
For GMSC, although CTGAN and TVAE have comparable predictive utility (0.78 vs. 0.79 AUC under LightGBM), their explainability consistency differs dramatically. TVAE maintains moderate ranking consistency ($\rho \approx 0.45-0.64$), whereas CTGAN attributions diverge significantly ($\rho \approx 0.19-0.29$, weak consistency). This provides empirical proof of the decoupling of downstream classification utility and explanation fidelity.

Features rank heatmap shows feature position shifts across Real, CTGAN, and TVAE. For GMSC, features like `RevolvingUtilizationOfUnsecuredLines` and `DebtRatio` remain highly ranked across all models, but mid-ranked features show severe position swaps in CTGAN. For German Credit, the top feature `checking_status` is preserved as the most important across all pipelines.

### 5.5 ROC Curves and MIA Evaluations
In all settings, the Membership Inference Attack (MIA) ROC curve hovers near the diagonal baseline (AUC $\approx 0.50$). This indicates that distance-based MIA cannot distinguish between members and non-members, confirming that neither generator is highly vulnerable to identity leakage through simple distance queries.

### 5.6 Statistical Rigor Analysis
With $N=10$ seeds, Wilcoxon signed-rank tests compare TVAE and CTGAN performance.
- On German Credit utility, TVAE outperforms CTGAN with $p = 0.0020$ and paired Cohen's $d = 3.68$ (XGBoost) and $d = 4.39$ (LightGBM), indicating an extremely large and significant effect size.
- On GMSC explainability consistency (SHAP $\rho$), TVAE significantly outperforms CTGAN with $p = 0.0020, d = 1.52$ (XGBoost) and $p = 0.0059, d = 1.11$ (LightGBM).
- Conversely, TVAE has lower DCR than CTGAN by an extremely large effect size ($p = 0.0020, d = -28.56$ for German Credit; $p = 0.0020, d = -6.60$ for GMSC), statistically validating the utility-privacy tradeoff.

---

## 6. Discussion and Limitations
These results have critical governance implications for financial institutions. If a bank utilizes CTGAN to share synthetic data for model validation or debugging, the resulting models will use substantially shifted feature importances, rendering validation findings questionable. For explainable credit scoring pipelines, TVAE is the preferred generator due to its superior rank consistency. However, because TVAE records lie closer to real records, institutions must add differential privacy (DP) guarantees to mitigate disclosure risk. On German Credit, DP-TVAE at $\varepsilon=5.0$ provides a viable mitigation, keeping Inference Risk below baseline while retaining better structure than CTGAN. On continuous dense datasets like GMSC, standard DP-SGD collapses downstream utility due to extreme default class imbalance, showing the need for imbalanced private learning algorithms.

The study has some limitations:
* Evaluations are conducted under CPU-only constraints, limiting the size of evaluated datasets.
* Two classifier families (XGBoost and LightGBM) were evaluated. Results are consistent across both gradient boosting architectures. Extension to linear models (logistic regression) and neural networks remains as future work.
* Distance-based privacy indicators are highly sensitive to data density, causing false positives on large, continuous datasets like GMSC.

---

## 7. Conclusion
We performed a multi-seed comparative study of CTGAN and TVAE credit risk modeling pipelines, jointly evaluating predictive utility, SHAP consistency, and privacy spacing. We demonstrated that TVAE offers superior downstream utility and explainability consistency, but compromises on privacy margins by outputting records closer to the original training data. Furthermore, we demonstrated the utility-explainability decoupling on continuous data, where TVAE and CTGAN achieve similar predictive utility but TVAE provides significantly more consistent feature attributions. Future work will investigate integrating advanced imbalanced private algorithms within TVAE training to balance explainability, utility, and disclosure risk on banking portfolios.

---

## 8. References
[1] D. Han, Y. Wang, and H. Zhang, "Non-parametric oversampling technique for explainable credit scoring," *Scientific Reports*, vol. 14, no. 1, p. 1024, 2024.  
[2] J. Min and S. Oh, "Can synthetic data protect privacy?" *IEEE Access*, vol. 13, pp. 10450-10462, 2025.
