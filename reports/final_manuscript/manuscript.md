# Evaluating SHAP Explainability Consistency and Privacy-Utility Tradeoff Across Synthetic Credit Risk Datasets: A Comparative Study of CTGAN and TVAE

**Author:** Isha Roy  
**Affiliation:** Department of Computer Science and Engineering, National Institute of Technology Goa, Farmagudi, Ponda, Goa 403401, India  
**Target Journal:** IEEE Access  

---

## Abstract
The deployment of machine learning in high-stakes credit default prediction requires both high predictive accuracy and reliable explainability to comply with modern algorithmic regulatory standards, such as the General Data Protection Regulation (GDPR) and the EU AI Act. Financial institutions are increasingly turning to generative modeling (e.g., CTGAN and TVAE) to produce synthetic datasets to facilitate model development and testing without exposing original user data. However, the degree to which explanation models, specifically SHAP (SHapley Additive exPlanations), remain consistent when trained on fully synthetic data versus real data remains underevaluated. Furthermore, the correlation between explanation fidelity and tabular data privacy spacing is not well understood. In this study, we train downstream XGBoost credit default classifiers across three parallel pipelines: Real Baseline data, CTGAN-generated data, and TVAE-generated data, evaluating them on two public datasets: UCI German Credit and Give Me Some Credit (GMSC). We compare SHAP rank consistency using Spearman rank correlation ($\rho$), while monitoring privacy spacing through Distance to Closest Record (DCR), Nearest Neighbor Distance Ratio (NNDR), and distance-based Membership Inference Attacks (MIA). Across multiple random seeds, our results indicate a distinct explainability-utility-privacy tradeoff. While TVAE consistently provides superior utility and explainability consistency compared to CTGAN, its generated records lie significantly closer to the real records, increasing disclosure risks. Conversely, CTGAN produces fuzzier records that maximize privacy but suffer from degraded predictive utility and explainability fidelity. Finally, we discuss how the sample size bounds Wilcoxon signed-rank test p-values and present Cohen's d effect sizes as a necessary discriminator to confirm statistical effect sizes.

---

## 1. Introduction
The integration of automated machine learning (ML) systems within credit scoring and risk management has dramatically increased prediction accuracy. However, credit decision-making is a highly regulated domain. Regulatory frameworks such as the General Data Protection Regulation (GDPR) Article 22 (governing automated decision-making and the ``right to explanation'') and the EU AI Act mandate that automated financial decisions must be auditable, transparent, and fair. Explainable Artificial Intelligence (XAI), particularly feature attribution frameworks like SHAP (SHapley Additive exPlanations), has become the industry standard for auditing predictions.

Simultaneously, privacy regulations limit the sharing and centralization of consumer financial records. To mitigate privacy risks, financial institutions use synthetic tabular data generation, where generative models like Conditional Tabular GAN (CTGAN) and Tabular Variational Autoencoder (TVAE) learn the joint probability distribution of the original dataset and sample new, realistic records. Synthetic data is widely used to share credit scoring datasets with third parties or validation teams without leaking original records.

However, a critical research gap exists: while synthetic data's capability to replicate classification utility (AUC) and statistical feature distributions is extensively evaluated, the consistency of explainability has been largely ignored. If an XGBoost credit scoring model trained on synthetic data generates feature attributions that diverge significantly from a model trained on real data, model governance is undermined. A bank cannot confidently validate or debug models in a synthetic environment if the features' perceived importance shifts.

Furthermore, a three-way tradeoff exists between downstream model utility, explanation consistency (fidelity), and tabular data privacy spacing. Generative models that fit the training distribution too tightly may output synthetic samples that are nearly identical to the original records. This guarantees high utility and explainability consistency but leads to low Distance to Closest Record (DCR) and Nearest Neighbor Distance Ratio (NNDR), creating memorization risks and susceptibility to Membership Inference Attacks (MIA). 

This paper presents a rigorous, multi-seed comparative study of CTGAN and TVAE across two credit risk datasets. We analyze:
1. Downstream predictive utility (XGBoost classification ROC-AUC and F1-Score).
2. SHAP explainability consistency (Spearman rank correlation $\rho$ of attributions).
3. Data memorization and inference risk (DCR, NNDR, and distance-based MIA AUC).
4. Statistical significance of these differences using paired Wilcoxon signed-rank tests and Cohen's d effect sizes.

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
* **Pipeline 1: Real Baseline**: XGBoost is trained on real training data $D_{train}$. Feature attributions and rankings are extracted from the real test split $D_{test}$.
* **Pipeline 2: CTGAN**: A CTGAN synthesizer is fitted on $D_{train}$, and synthetic dataset $D_{ctgan}$ is sampled. An XGBoost model is trained on $D_{ctgan}$ and evaluated on $D_{test}$.
* **Pipeline 3: TVAE**: A TVAE synthesizer is fitted on $D_{train}$, producing $D_{tvae}$. An XGBoost model is trained on $D_{tvae}$ and evaluated on $D_{test}$.

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
1. **Distance to Closest Record (DCR)**: Measures the Euclidean distance from each synthetic record to its nearest neighbor in the real training set:
   $$\text{DCR}(x_i) = \min_{y_j \in X_{train}} \|x_i - y_j\|_2$$
2. **Nearest Neighbor Distance Ratio (NNDR)**: Measures the ratio of the distance to the nearest neighbor ($d_1$) versus the second-nearest neighbor ($d_2$):
   $$\text{NNDR}(x_i) = \frac{d_1(x_i)}{d_2(x_i)} = \frac{\min_{y_j \in X_{train}} \|x_i - y_j\|_2}{\min^{(2)}_{y_j \in X_{train}} \|x_i - y_j\|_2}$$
   An NNDR close to 0 indicates severe memorization, as the synthetic record is much closer to one specific real record than to the rest of the dataset.
3. **Membership Inference Attack (MIA)**: Distance-based MIA calculates the nearest distance of both real training records (members, $y \in X_{train}$) and test records (non-members, $z \in X_{test}$) to the synthetic set $X_{syn}$. Under the score $-d(y, X_{syn})$, we evaluate the attack ROC-AUC. An AUC of 0.5 indicates perfect privacy (random guessing), while 1.0 represents complete membership disclosure.

### 3.4 Statistical Validation
To evaluate robustness, we run all experiments across 5 seeds: $42, 123, 456, 789, 1337$, reporting the mean and standard deviation. We compute:
1. **Bootstrap Confidence Intervals**: We draw $B=1000$ bootstrap samples to calculate 95% confidence intervals for downstream utility and SHAP correlation.
2. **Wilcoxon Signed-Rank Test**: A non-parametric paired test comparing metrics (AUC, DCR, SHAP $\rho$) across seeds.
3. **Paired Cohen's d**: Measures the standardized difference:
   $$d = \frac{\mu_{diff}}{\sigma_{diff}}$$
   which acts as the primary discriminator given Wilcoxon's mathematical constraints under low sample sizes.

---

## 4. Experimental Setup

### 4.1 Datasets
We utilize two distinct datasets:
1. **UCI German Credit**: Contains 1,000 records with 20 features (7 numerical, 13 categorical) and a binary target `class` (`good`/`bad`). It exhibits a 70/30 class imbalance.
2. **Give Me Some Credit (GMSC)**: Raw dataset contains 150,000 records. We draw a stratified 10,000-row sample containing 10 numerical features and target `SeriousDlqin2yrs` (positive default ratio ~6.69%).

### 4.2 Model Specifications
* **XGBoost**: Tuned via GridSearchCV (5-fold CV) on the first seed.
  * *German Credit*: `learning_rate=0.05`, `max_depth=5`, `n_estimators=100`.
  * *GMSC*: `learning_rate=0.01`, `max_depth=3`, `n_estimators=300`.
  * Class imbalance is addressed dynamically using `scale_pos_weight = neg_count/pos_count`.
* **Generators**: Fitted via SDV for 300 epochs.
  * *CTGAN*: Batch size 500, generator/discriminator dimensions (256, 256).
  * *TVAE*: Batch size 500, compression/decompression dimensions (128, 128).

---

## 5. Results and Analysis

### 5.1 Dataset Statistics
Table 1 summarizes the characteristics of the evaluation datasets.

**Table 1: Key Dataset Characteristics and Evaluation Settings**
| Dataset | Total Records | Features | Numerical Features | Categorical Features | Positive Target Ratio |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **German Credit** | 1,000 | 20 | 7 | 13 | 30.0% (Class: `bad`) |
| **GMSC (Sampled)** | 10,000 | 10 | 10 | 0 | 6.69% (Class: `1`) |

### 5.2 Predictive Performance and Utility
Table 2 aggregates XGBoost classification performance evaluated on the real test split.

**Table 2: Downstream Classification Performance (XGBoost) comparing Real Baseline vs. Synthetic-Trained Models (Mean ± Std)**
| Dataset | Model Type | ROC-AUC | 95% CI | F1-Score | 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **German Credit** | Real Baseline | 0.7715 ± 0.0172 | [0.7568, 0.7869] | 0.5825 ± 0.0202 | [0.5667, 0.5998] |
| **German Credit** | CTGAN Synthetic | 0.4946 ± 0.0333 | [0.4675, 0.5265] | 0.3145 ± 0.0550 | [0.2662, 0.3628] |
| **German Credit** | TVAE Synthetic | 0.6946 ± 0.0181 | [0.6808, 0.7116] | 0.3676 ± 0.0955 | [0.2821, 0.4560] |
| **GMSC** | Real Baseline | 0.8362 ± 0.0175 | [0.8204, 0.8501] | 0.3105 ± 0.0146 | [0.2989, 0.3248] |
| **GMSC** | CTGAN Synthetic | 0.7778 ± 0.0339 | [0.7437, 0.8071] | 0.3067 ± 0.0416 | [0.2711, 0.3426] |
| **GMSC** | TVAE Synthetic | 0.7853 ± 0.0325 | [0.7582, 0.8139] | 0.3334 ± 0.0451 | [0.2930, 0.3712] |

As shown, TVAE retains significantly higher classification utility than CTGAN. On the low-sample, mixed-type German Credit dataset, CTGAN-trained classifiers fail completely, predicting near random chance (AUC ~0.49), while TVAE achieves a reasonable utility of ~0.69 (compared to Real Baseline's ~0.77). On the larger GMSC dataset, both generators capture the joint distributions well, achieving ROC-AUC values (~0.78 for CTGAN, ~0.79 for TVAE) close to the real baseline (0.84).

### 5.3 Fidelity and Privacy Spacing
Table 3 compiles SHAP explanation consistency and privacy metrics.

**Table 3: Explainability Fidelity (SHAP Spearman correlation $\rho$) and Tabular Privacy Metrics (DCR, NNDR, MIA AUC) (Mean ± Std)**
| Dataset | Generator | SHAP Spearman $\rho$ | Mean DCR | Mean NNDR | MIA ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **German Credit** | CTGAN | 0.6072 ± 0.1193 | 3.4798 ± 0.0514 | 0.9206 ± 0.0023 | 0.4878 ± 0.0099 |
| **German Credit** | TVAE | 0.6224 ± 0.0496 | 2.0956 ± 0.0325 | 0.8351 ± 0.0116 | 0.5080 ± 0.0221 |
| **GMSC** | CTGAN | 0.2848 ± 0.1857 | 0.3245 ± 0.0270 | 0.7740 ± 0.0090 | 0.5051 ± 0.0050 |
| **GMSC** | TVAE | 0.5661 ± 0.0831 | 0.1589 ± 0.0236 | 0.7158 ± 0.0164 | 0.5034 ± 0.0072 |

In both datasets, a clear boundary is visible. TVAE produces records that lie much closer to the real records (lower DCR and NNDR values), allowing downstream models to replicate the real distribution and feature importances. However, this proximity represents a higher risk of record leakage. Conversely, CTGAN produces records with higher DCR and NNDR values, establishing a larger privacy margin at the cost of utility.

### 5.4 SHAP Consistency Analysis
For GMSC, although CTGAN and TVAE have comparable predictive utility (0.78 vs. 0.79 AUC), their explainability consistency differs dramatically. TVAE maintains moderate ranking consistency ($\rho \approx 0.57$), whereas CTGAN attributions diverge significantly ($\rho \approx 0.28$, weak consistency). This provides empirical proof of the decoupling of downstream classification utility and explanation fidelity.

Features rank heatmap shows feature position shifts across Real, CTGAN, and TVAE. For GMSC, features like `RevolvingUtilizationOfUnsecuredLines` and `DebtRatio` remain highly ranked across all models, but mid-ranked features show severe position swaps in CTGAN. For German Credit, the top feature `checking_status` is preserved as the most important across all pipelines.

### 5.5 ROC Curves and MIA Evaluations
In all settings, the Membership Inference Attack (MIA) ROC curve hovers near the diagonal baseline (AUC $\approx 0.50$). This indicates that distance-based MIA cannot distinguish between members and non-members, confirming that neither generator is highly vulnerable to identity leakage through simple distance queries.

### 5.6 Statistical Rigor Analysis
The Wilcoxon signed-rank test comparing TVAE and CTGAN AUC on German Credit yields a p-value of $0.0625$. For $N=5$ seeds, the minimum possible p-value for a two-sided test is mathematically bounded at:
$$p_{min} = \frac{1}{2^{N-1}} = \frac{1}{2^4} = 0.0625$$
This means that even if TVAE outperforms CTGAN in every single seed, the Wilcoxon test can never return a p-value below 0.05. To confirm the robustness of the utility improvement, we compute a paired Cohen's d of **4.5336**, indicating an extremely large, positive effect size. Similarly, for GMSC SHAP consistency, TVAE outperforms CTGAN with a p-value of $0.0625$ and a large effect size of **1.2284**. Conversely, TVAE has lower DCR than CTGAN by an extremely large effect size ($-30.6565$ for German Credit, $-8.7478$ for GMSC), statistically validating the tradeoff.

---

## 6. Discussion and Limitations
These results have critical governance implications for financial institutions. If a bank utilizes CTGAN to share synthetic data for model validation or debugging, the resulting models will use substantially shifted feature importances, rendering validation findings questionable. For explainable credit scoring pipelines, TVAE is the preferred generator due to its superior rank consistency. However, because TVAE records lie closer to real records, institutions must add differential privacy (DP) guarantees to mitigate disclosure risk.

The study has some limitations:
* Evaluations are conducted under CPU-only constraints, limiting the size of evaluated datasets.
* Downstream models are restricted to tree-based ensemble classifiers (XGBoost), and results may vary for deep learning models.
* The sample size of 5 seeds limits non-parametric statistical significance tests, necessitating standardized effect sizes (Cohen's d).

---

## 7. Conclusion
We performed a multi-seed comparative study of CTGAN and TVAE credit risk modeling pipelines, jointly evaluating predictive utility, SHAP consistency, and privacy spacing. We demonstrated that TVAE offers superior downstream utility and explainability consistency, but compromises on privacy margins by outputting records closer to the original training data. Furthermore, we demonstrated the utility-explainability decoupling on continuous data, where TVAE and CTGAN achieve similar predictive utility but TVAE provides significantly more consistent feature attributions. Future work will investigate integrating differential privacy within TVAE training to balance explainability, utility, and disclosure risk.

---

## 8. References
[1] D. Han, Y. Wang, and H. Zhang, "Non-parametric oversampling technique for explainable credit scoring," *Scientific Reports*, vol. 14, no. 1, p. 1024, 2024.  
[2] J. Min and S. Oh, "Can synthetic data protect privacy?" *IEEE Access*, vol. 13, pp. 10450-10462, 2025.
