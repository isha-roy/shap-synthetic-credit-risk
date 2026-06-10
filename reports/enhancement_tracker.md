# ðŸš€ Project Enhancement Tracker

> **Status:** In Progress  
> **Last Updated:** 2026-06-10  
> **Goal:** Strengthen the SHAP Synthetic Credit Risk project for ML Engineering portfolio and IEEE Access submission.

---

## âœ… Decisions Made (Don't Revisit)

- [x] **10 seeds is sufficient** â€” Wilcoxon p=0.002 clears p<0.05. No need for 20 seeds.
- [x] **Use LightGBM, not Logistic Regression** â€” TreeSHAP-compatible, industry-standard in credit scoring, more impressive to interviewers.

---

## Phase 1: Visualization Enhancement
> âš¡ Zero compute cost â€” all files already exist in `figures/paper/`. Just wire them into the report.

### Step 1.1 â€” Update `reports/report.md` Section 13

- [x] Add **Figure C: Downstream Model ROC Curves**
  - File: `../figures/paper/german_credit_roc_curve.png`
  - File: `../figures/paper/gmsc_roc_curve.png`
  - Caption: What to look for (AUC gap between real vs synthetic)

- [x] Add **Figure D: MIA Attack ROC Curves**
  - File: `../figures/paper/german_credit_mia_roc_curve.png`
  - File: `../figures/paper/gmsc_mia_roc_curve.png`
  - Caption: Why MIA AUC â‰ˆ 0.50 (attacker is guessing randomly)

- [x] Add **Figure E: SHAP Beeswarm Plots**
  - File: `../figures/paper/german_credit_real_shap_beeswarm.png`
  - File: `../figures/paper/german_credit_ctgan_shap_beeswarm.png`
  - File: `../figures/paper/german_credit_tvae_shap_beeswarm.png`
  - File: `../figures/paper/gmsc_real_shap_beeswarm.png`
  - File: `../figures/paper/gmsc_ctgan_shap_beeswarm.png`
  - File: `../figures/paper/gmsc_tvae_shap_beeswarm.png`
  - Caption: Feature importance shifts between real and synthetic training

- [x] Add **Figure F: SHAP Feature-Rank Heatmaps**
  - File: `../figures/paper/german_credit_shap_heatmap.png`
  - File: `../figures/paper/gmsc_shap_heatmap.png`
  - Caption: Feature rank heatmaps across seeds

- [x] Add **Figure G: SHAP Consistency Comparison**
  - File: `../figures/paper/german_credit_shap_consistency.png`
  - File: `../figures/paper/gmsc_shap_consistency.png`
  - Caption: Spearman ρ distribution across seeds

- [x] Add **Figure H: Utility-Privacy Tradeoff Scatter**
  - File: `../figures/paper/german_credit_utility_privacy_tradeoff.png`
  - File: `../figures/paper/gmsc_utility_privacy_tradeoff.png`
  - Caption: The 3-way tradeoff visualized in 2D

### Step 1.2 — Update `docs/index.html`

- [x] Add figure gallery section with all new figures
- [x] Verify all image paths render correctly in browser

### Step 1.3 — Verify Figure Quality

- [x] Open each PNG — check axis labels are readable
- [x] Check seed labels and color consistency across figures

### Phase 1 Report Consistency Check

- [x] Run: `Select-String -Path "reports\report.md" -Pattern "Figure [CDEFGH]"` — verify all figures referenced
- [x] Push changes to GitHub so GitHub Pages updates




---

## Phase 2: Differential Privacy (DP-TVAE)
> ðŸ• Compute-heavy. Start before bed and let it run overnight.

### Step 2.1 â€” Setup

- [ ] Install opacus: `pip install opacus`
- [ ] Verify installation: `python -c "import opacus; print(opacus.__version__)"`
- [ ] Create `configs/experiments/tvae_dp_default.json`
  ```json
  {
    "parameters": { "epochs": 300, "batch_size": 500, "compress_dims": [128,128],
                    "decompress_dims": [128,128], "embedding_dim": 128,
                    "enforce_min_max_values": true, "enforce_rounding": true, "cuda": false },
    "dp_parameters": { "epsilon_values": [1.0, 5.0, 10.0], "delta": 1e-5, "max_grad_norm": 1.0 }
  }
  ```

### Step 2.2 â€” Generate DP Synthetic Data

- [ ] Create `scripts/generate_dp_synthetic.py`
  - Wrap TVAE PyTorch optimizer with `opacus.PrivacyEngine`
  - Train 3 variants: Îµ=1 (strong), Îµ=5 (moderate), Îµ=10 (weak)
  - Output: `data/synthetic/{dataset}_tvae_dp_eps{e}_seed{seed}.csv`
  - 10 seeds Ã— 2 datasets Ã— 3 epsilon = 60 new files
- [ ] Run: `python scripts/generate_dp_synthetic.py`
- [ ] Verify output files exist in `data/synthetic/`

### Step 2.3 â€” Train Downstream Models on DP Synthetic Data

- [ ] Create `scripts/train_dp_synthetic.py`
  - Mirror of `train_synthetic.py` for DP-TVAE datasets
  - Output to `results/tvae_dp/`
- [ ] Run: `python scripts/train_dp_synthetic.py`
- [ ] Verify JSON summaries written to `results/tvae_dp/`

### Step 2.4 â€” Evaluate Privacy on DP Synthetic Data

- [ ] Create `scripts/evaluate_dp_privacy.py`
  - Reuse DCR, NNDR, MIA, Inference Risk logic
  - Compare: CTGAN vs TVAE vs DP-TVAE(Îµ=1,5,10)
- [ ] Run: `python scripts/evaluate_dp_privacy.py`
- [ ] Note inference risk score for each epsilon â€” find the crossover point

### Step 2.5 â€” Update Report After Phase 2

- [ ] Add new **Section 10a: Differential Privacy Results** to `reports/report.md`
  - Table: CTGAN / TVAE / DP-TVAE(Îµ=10,5,1) Ã— AUC / IR Score / IR Status
- [ ] Add DP-TVAE rows to **Section 12: Master Results Table**
- [ ] Update **Finding 5** â€” change "unavoidable without DP" â†’ "breakable at Îµ=5"
- [ ] Remove "No DP implementation" from **Section 15 Limitations**
- [ ] Add "Multi-generator privacy mitigation" row to **Section 4 comparison table**
- [ ] Re-run visualizations: `python scripts/generate_paper_visualizations.py`
- [ ] Update `docs/index.html` with DP results
- [ ] Run consistency check: `Select-String -Path "reports\report.md" -Pattern "without DP"`

---

## Phase 3: Multi-Classifier Validation (LightGBM)

### Step 3.1 â€” Setup

- [ ] Install LightGBM: `pip install lightgbm`
- [ ] Verify: `python -c "import lightgbm; print(lightgbm.__version__)"`
- [ ] Create `configs/models/lightgbm_german_credit.json`
- [ ] Create `configs/models/lightgbm_gmsc.json`

### Step 3.2 â€” Baseline Training (LightGBM on Real Data)

- [ ] Create `scripts/train_baseline_lgbm.py`
  - Use `lightgbm.LGBMClassifier`
  - 5-fold CV hyperparameter tuning on seed 42
  - All 10 seeds, append logic (same pattern as `train_baseline.py`)
  - Output: `results/baseline/german_credit_real_lgbm_summary.json`
  - Output: `results/baseline/gmsc_real_lgbm_summary.json`
- [ ] Run: `python scripts/train_baseline_lgbm.py`
- [ ] Verify both JSON files written

### Step 3.3 â€” Synthetic Training (LightGBM on CTGAN + TVAE)

- [ ] Create `scripts/train_synthetic_lgbm.py`
  - Mirror of `train_synthetic.py` using LightGBM
  - Output to `results/ctgan/` and `results/tvae/` (with `_lgbm_` in filename)
- [ ] Run: `python scripts/train_synthetic_lgbm.py`
- [ ] Verify 4 JSON summaries: german/gmsc Ã— ctgan/tvae

### Step 3.4 â€” SHAP Consistency Analysis (LightGBM)

- [ ] Create `scripts/analyze_shap_consistency_lgbm.py`
  - Use `shap.TreeExplainer(lgbm_model)` â€” same API as XGBoost
  - Compute Spearman Ï and top-5/top-10 overlap
  - Output: `results/shap/german_credit_lgbm_shap_consistency_summary.json`
  - Output: `results/shap/gmsc_lgbm_shap_consistency_summary.json`
- [ ] Run: `python scripts/analyze_shap_consistency_lgbm.py`
- [ ] **Key question to answer:** Does TVAE Ï >> CTGAN Ï gap on GMSC persist under LightGBM?

### Step 3.5 â€” Statistical Validation (LightGBM)

- [ ] Update `scripts/run_statistical_validation.py` to include LightGBM results
  - OR create `scripts/run_statistical_validation_lgbm.py`
- [ ] Run statistical validation
- [ ] Note p-values and Cohen's d for LightGBM comparisons

### Step 3.6 â€” Update Report After Phase 3

- [ ] Add LightGBM rows to **Section 7: Utility** table
- [ ] Add LightGBM rows to **Section 8: SHAP Consistency** table
- [ ] Add LightGBM rows to **Section 11: Statistical Significance** table
- [ ] Add LightGBM rows to **Section 12: Master Results Table**
- [ ] Update **Finding 3** â€” add "holds under LightGBM too (classifier-agnostic)"
- [ ] Add "Multi-classifier validation" row to **Section 4 comparison table**
- [ ] Update `docs/index.html`
- [ ] Run consistency check: `Select-String -Path "reports\report.md" -Pattern "only.*XGBoost|single classifier"`

---

## Phase 4: Final Report Polish

### Consistency Checks (Run After All Phases)

- [ ] No stale 5-seed references:
  ```powershell
  Select-String -Path "reports\report.md" -Pattern "(N=5|0\.0625|5 seeds)"
  ```
- [ ] No old Ï values:
  ```powershell
  Select-String -Path "reports\report.md" -Pattern "(0\.57 vs 0\.28|rho.*0\.60|0\.6072|0\.6224)"
  ```
- [ ] No "logistic regression" in improvements:
  ```powershell
  Select-String -Path "reports\report.md" -Pattern "logistic regression"
  ```
- [ ] No "without DP" framing after Phase 2:
  ```powershell
  Select-String -Path "reports\report.md" -Pattern "without DP|no differential"
  ```

### Final Steps

- [ ] Update Section 16 (Improvements): mark Phase 1â€“3 items as Done, add new next steps
- [ ] Update Section 17 (ML Engineering Skills): add DP and multi-classifier as demonstrated skills
- [ ] Proofread all new sections once
- [ ] Update `docs/index.html` with final state
- [ ] Git commit with message: `feat: add DP-TVAE, LightGBM, full visualization suite`
- [ ] Push to GitHub â€” verify GitHub Pages renders correctly

---

## ðŸ“Š Progress Summary

| Phase | Task | Status |
|---|---|---|
| 1 | Add ROC curves to report | ✅ Done |
| 1 | Add SHAP beeswarm to report | ✅ Done |
| 1 | Add SHAP rank heatmaps to report | ✅ Done |
| 1 | Add SHAP consistency plots to report | ✅ Done |
| 1 | Add utility-privacy tradeoff plots to report | ✅ Done |
| 1 | Update docs/index.html | ✅ Done |

| 2 | Install opacus + config | â¬œ Not started |
| 2 | generate_dp_synthetic.py | â¬œ Not started |
| 2 | train_dp_synthetic.py | â¬œ Not started |
| 2 | evaluate_dp_privacy.py | â¬œ Not started |
| 2 | Report update (DP section) | â¬œ Not started |
| 3 | Install LightGBM + configs | â¬œ Not started |
| 3 | train_baseline_lgbm.py | â¬œ Not started |
| 3 | train_synthetic_lgbm.py | â¬œ Not started |
| 3 | analyze_shap_consistency_lgbm.py | â¬œ Not started |
| 3 | Report update (LightGBM section) | â¬œ Not started |
| 4 | Final consistency checks | â¬œ Not started |
| 4 | Git commit + push | â¬œ Not started |


