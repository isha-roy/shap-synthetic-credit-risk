# CPU-Only Reproducibility Guidelines

This document details guidelines for running and reproducing the experimental results on CPU-constrained architectures, specifically targeting:
* **Operating System**: Windows 11
* **Python Version**: 3.11.9
* **Hardware Specs**: Intel i5-1335U CPU, 8 GB RAM, SSD, no GPU.

## Key Risks & Resource Limitations
With only 8 GB of RAM and no GPU acceleration, running deep learning models like CTGAN and TVAE can easily lead to memory starvation, kernel panics, or extremely slow runtimes if not properly managed.

## CPU Mitigation Strategies

1. **Modest Batch Sizes**
   * Keep generator batch sizes bounded (e.g., `batch_size=500` is recommended). Avoid very large batch sizes that require high peak memory.
   
2. **Epoch Management**
   * Default CTGAN/TVAE training is capped at 300 epochs. Monitor runtime for the first seed. If a single seed takes more than 1–2 hours on the German Credit dataset, downscale the epochs to a smaller, stable value (e.g., 150) for testing before launching full runs.

3. **Sequential Execution**
   * Never run multiple experiments in parallel (e.g., do not run CTGAN and TVAE simultaneously, or run multiple notebooks at once). Complete one seed and one model at a time.
   
4. **Caching Intermediates**
   * Save and load intermediate preprocessed data (`data/processed/`) to disk instead of keeping them in memory.
   * Serialize fitted scalers and imputers to `models/` to avoid re-fitting.

5. **SHAP Subsampling**
   * If computing SHAP tree values on the full dataset causes memory issues or hangs the kernel, compute SHAP on a controlled, stratified evaluation subset (e.g., a representative 1,000-row sample of the test set).

6. **GMSC Sampling**
   * The Kaggle Give Me Some Credit (GMSC) dataset is 150,000 rows. **Always work on the 10,000-row stratified subset** to stay within the 8 GB RAM limit during CTGAN/TVAE training.
