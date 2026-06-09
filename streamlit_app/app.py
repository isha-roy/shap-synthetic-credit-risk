"""
SHAP Synthetic Credit Risk — Interactive Research Dashboard
============================================================
A multi-page Streamlit app for exploring research results interactively.
Reads from pre-computed JSON/CSV result files — no recomputation needed.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

SEEDS = [42, 123, 456, 789, 1337]
DATASETS = ["german_credit", "gmsc"]
GENERATORS = ["ctgan", "tvae"]
DATASET_LABELS = {"german_credit": "German Credit (UCI)", "gmsc": "GMSC (Kaggle)"}

# Plotly color palette
COLORS = {
    "real": "#3b82f6",
    "ctgan": "#f472b6",
    "tvae": "#a78bfa",
    "pass": "#34d399",
    "fail": "#fb7185",
    "threshold": "#fbbf24",
    "bg": "#0a0e1a",
    "card": "#111827",
    "text": "#f1f5f9",
    "muted": "#64748b",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=13),
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
)


# ============================================================
# DATA LOADING HELPERS
# ============================================================
@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_stats(dataset):
    return load_json(RESULTS_DIR / "statistics" / f"{dataset}_statistical_validation.json")


@st.cache_data
def load_privacy(dataset):
    return load_json(RESULTS_DIR / "privacy" / f"{dataset}_privacy_summary.json")


@st.cache_data
def load_inference_risk(dataset):
    return load_json(RESULTS_DIR / "privacy" / f"inference_risk_{dataset.replace('german_credit','german')}.json")


@st.cache_data
def load_shap_consistency(dataset):
    return load_json(RESULTS_DIR / "shap" / f"{dataset}_shap_consistency_summary.json")


def get_metric_values(stats, key):
    """Extract mean, std, ci from stats dict."""
    m = stats["metrics_summary"][key]
    return m["mean"], m["std"], m.get("ci_95_lower"), m.get("ci_95_upper"), m.get("raw_values", [])


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SHAP Synthetic Credit Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(10, 14, 26, 0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Headers */
    h1, h2, h3 {
        color: #f1f5f9 !important;
    }

    /* Badge styling */
    .badge-pass {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-fail {
        background: rgba(251, 113, 133, 0.15);
        color: #fb7185;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📊 Research Dashboard")
    st.markdown("**SHAP × Synthetic Credit Risk**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📈 Predictive Utility", "🔍 SHAP Explainability",
         "🔐 Privacy Metrics", "📊 Statistical Tests"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("##### Quick Links")
    st.markdown("**Author:** Isha Roy")
    st.markdown("**Affiliation:** NIT Goa")
    st.markdown("**Seeds:** 42, 123, 456, 789, 1337")

    st.markdown("---")
    st.caption("Built with Streamlit • Data from pre-computed results")


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.markdown("# 🎓 SHAP Synthetic Credit Risk — Research Overview")
    st.markdown("*Evaluating SHAP Explainability Consistency and Privacy-Utility Tradeoff Across Synthetic Credit Risk Datasets*")

    st.markdown("---")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pipelines", "3", help="Real, CTGAN, TVAE")
    col2.metric("Datasets", "2", help="German Credit + GMSC")
    col3.metric("Seeds", "5", help="42, 123, 456, 789, 1337")
    col4.metric("Privacy Metrics", "4", help="DCR, NNDR, MIA, Inference Risk")

    st.markdown("---")

    # Core question
    st.markdown("### 🎯 Core Research Question")
    st.info(
        "If you train a credit scoring model on **synthetic data** instead of real data, "
        "do you get the **same predictions** AND the **same explanations**? And does the "
        "synthetic data actually **protect privacy**, or does it accidentally leak real customer information?"
    )

    # Three pipelines
    st.markdown("### 🔄 Three Parallel Pipelines")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔵 Pipeline 1")
        st.markdown("XGBoost trained on **Real Data** (Baseline)")
    with c2:
        st.markdown("#### 🟣 Pipeline 2")
        st.markdown("XGBoost trained on **CTGAN** synthetic data")
    with c3:
        st.markdown("#### 🟡 Pipeline 3")
        st.markdown("XGBoost trained on **TVAE** synthetic data")

    st.markdown("---")

    # Master results
    st.markdown("### 📋 Master Results Table")
    master_data = {
        "Dataset": ["German Credit", "German Credit", "GMSC", "GMSC"],
        "Generator": ["CTGAN", "TVAE", "CTGAN", "TVAE"],
        "AUC": [0.4946, 0.6946, 0.7778, 0.7853],
        "SHAP ρ": [0.6072, 0.6224, 0.2848, 0.5661],
        "DCR": [3.4798, 2.0956, 0.3245, 0.1589],
        "NNDR": [0.9206, 0.8351, 0.7740, 0.7158],
        "MIA AUC": [0.4878, 0.5080, 0.5051, 0.5034],
        "Inference Risk": [0.2094, 0.6163, 0.4519, 0.5632],
        "IR Status": ["✅ PASS", "❌ FAIL", "✅ PASS", "❌ FAIL"],
    }
    st.dataframe(pd.DataFrame(master_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Key findings
    st.markdown("### 💡 Key Findings")
    findings = [
        ("🟣 TVAE", "Better utility & explanations, but **fails privacy** (Inference Risk exceeds threshold on both datasets)"),
        ("🔵 CTGAN", "Protects privacy, but **destroys explanations** (SHAP ρ = 0.28 on GMSC)"),
        ("⚠️ Decoupling", "Good predictions ≠ good explanations (GMSC: similar AUC, 2x SHAP gap)"),
        ("🔐 MIA Insufficient", "MIA AUC ≈ 0.50 for all — misses local memorization that Inference Risk catches"),
        ("📊 Tradeoff", "Utility-Explainability-Privacy tradeoff is **fundamental** without differential privacy"),
    ]
    for emoji_title, desc in findings:
        st.markdown(f"**{emoji_title}:** {desc}")


# ============================================================
# PAGE 2: PREDICTIVE UTILITY
# ============================================================
elif page == "📈 Predictive Utility":
    st.markdown("# 📈 Downstream Predictive Utility")
    st.markdown("ROC-AUC and F1-Score comparison across all pipelines and datasets.")

    dataset = st.selectbox("Select Dataset", DATASETS, format_func=lambda x: DATASET_LABELS[x])
    stats = load_stats(dataset)

    st.markdown("---")

    # Metric cards
    col1, col2, col3 = st.columns(3)

    real_auc, real_auc_std, *_ = get_metric_values(stats, "real_auc")
    ctgan_auc, ctgan_auc_std, *_ = get_metric_values(stats, "ctgan_auc")
    tvae_auc, tvae_auc_std, *_ = get_metric_values(stats, "tvae_auc")

    col1.metric("Real Baseline AUC", f"{real_auc:.4f}", help=f"Std: ±{real_auc_std:.4f}")
    col2.metric("CTGAN AUC", f"{ctgan_auc:.4f}", delta=f"{ctgan_auc - real_auc:+.4f} vs Real")
    col3.metric("TVAE AUC", f"{tvae_auc:.4f}", delta=f"{tvae_auc - real_auc:+.4f} vs Real")

    st.markdown("---")

    # AUC comparison bar chart
    st.markdown("### ROC-AUC Comparison")
    real_vals = stats["metrics_summary"]["real_auc"]["raw_values"]
    ctgan_vals = stats["metrics_summary"]["ctgan_auc"]["raw_values"]
    tvae_vals = stats["metrics_summary"]["tvae_auc"]["raw_values"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Real", x=[f"Seed {s}" for s in SEEDS], y=real_vals,
        marker_color=COLORS["real"], opacity=0.9
    ))
    fig.add_trace(go.Bar(
        name="CTGAN", x=[f"Seed {s}" for s in SEEDS], y=ctgan_vals,
        marker_color=COLORS["ctgan"], opacity=0.9
    ))
    fig.add_trace(go.Bar(
        name="TVAE", x=[f"Seed {s}" for s in SEEDS], y=tvae_vals,
        marker_color=COLORS["tvae"], opacity=0.9
    ))
    fig.update_layout(
        barmode="group", title=f"ROC-AUC per Seed — {DATASET_LABELS[dataset]}",
        yaxis_title="ROC-AUC", xaxis_title="Seed",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    # F1 comparison
    st.markdown("### F1-Score Comparison")
    real_f1 = stats["metrics_summary"]["real_f1"]["raw_values"]
    ctgan_f1 = stats["metrics_summary"]["ctgan_f1"]["raw_values"]
    tvae_f1 = stats["metrics_summary"]["tvae_f1"]["raw_values"]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Real", x=[f"Seed {s}" for s in SEEDS], y=real_f1, marker_color=COLORS["real"], opacity=0.9))
    fig2.add_trace(go.Bar(name="CTGAN", x=[f"Seed {s}" for s in SEEDS], y=ctgan_f1, marker_color=COLORS["ctgan"], opacity=0.9))
    fig2.add_trace(go.Bar(name="TVAE", x=[f"Seed {s}" for s in SEEDS], y=tvae_f1, marker_color=COLORS["tvae"], opacity=0.9))
    fig2.update_layout(
        barmode="group", title=f"F1-Score per Seed — {DATASET_LABELS[dataset]}",
        yaxis_title="F1-Score", xaxis_title="Seed",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ROC curve image
    st.markdown("### ROC Curves (Seed 42)")
    roc_path = FIGURES_DIR / "paper" / f"{dataset}_roc_curve.png"
    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)

    # Summary table
    st.markdown("### Detailed Results")
    detail_data = []
    for model_key, label in [("real", "Real Baseline"), ("ctgan", "CTGAN"), ("tvae", "TVAE")]:
        auc_m, auc_s, auc_lo, auc_hi, _ = get_metric_values(stats, f"{model_key}_auc")
        f1_m, f1_s, f1_lo, f1_hi, _ = get_metric_values(stats, f"{model_key}_f1")
        detail_data.append({
            "Model": label,
            "AUC (Mean)": f"{auc_m:.4f}",
            "AUC (Std)": f"±{auc_s:.4f}",
            "AUC 95% CI": f"[{auc_lo:.4f}, {auc_hi:.4f}]" if auc_lo else "—",
            "F1 (Mean)": f"{f1_m:.4f}",
            "F1 (Std)": f"±{f1_s:.4f}",
            "F1 95% CI": f"[{f1_lo:.4f}, {f1_hi:.4f}]" if f1_lo else "—",
        })
    st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 3: SHAP EXPLAINABILITY
# ============================================================
elif page == "🔍 SHAP Explainability":
    st.markdown("# 🔍 SHAP Explainability Consistency")
    st.markdown("How well do synthetic-trained models preserve the same feature importance explanations as the real-trained model?")

    dataset = st.selectbox("Select Dataset", DATASETS, format_func=lambda x: DATASET_LABELS[x])
    stats = load_stats(dataset)
    shap_data = load_shap_consistency(dataset)

    st.markdown("---")

    # Metric cards
    col1, col2 = st.columns(2)
    ctgan_shap_m, ctgan_shap_s, *_ = get_metric_values(stats, "ctgan_shap")
    tvae_shap_m, tvae_shap_s, *_ = get_metric_values(stats, "tvae_shap")
    col1.metric("CTGAN SHAP ρ", f"{ctgan_shap_m:.4f}", help=f"Std: ±{ctgan_shap_s:.4f}")
    col2.metric("TVAE SHAP ρ", f"{tvae_shap_m:.4f}", help=f"Std: ±{tvae_shap_s:.4f}")

    st.markdown("---")

    # Spearman rho per seed
    st.markdown("### Spearman ρ per Seed")
    ctgan_rhos = [r["spearman_rho"] for r in shap_data["ctgan_runs"]]
    tvae_rhos = [r["spearman_rho"] for r in shap_data["tvae_runs"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"Seed {s}" for s in SEEDS], y=ctgan_rhos, mode="lines+markers",
        name="CTGAN", line=dict(color=COLORS["ctgan"], width=2),
        marker=dict(size=10)
    ))
    fig.add_trace(go.Scatter(
        x=[f"Seed {s}" for s in SEEDS], y=tvae_rhos, mode="lines+markers",
        name="TVAE", line=dict(color=COLORS["tvae"], width=2),
        marker=dict(size=10)
    ))
    fig.update_layout(
        title=f"SHAP Spearman ρ per Seed — {DATASET_LABELS[dataset]}",
        yaxis_title="Spearman ρ", xaxis_title="Seed",
        yaxis_range=[0, 1],
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    # Jaccard overlap
    st.markdown("### Jaccard Feature Overlap")
    jaccard_data = []
    for gen, runs in [("CTGAN", shap_data["ctgan_runs"]), ("TVAE", shap_data["tvae_runs"])]:
        for run in runs:
            jaccard_data.append({
                "Generator": gen,
                "Seed": run["seed"],
                "Top-5 Jaccard": f"{run['top5_jaccard']:.3f}",
                "Top-10 Jaccard": f"{run['top10_jaccard']:.3f}",
                "Spearman ρ": f"{run['spearman_rho']:.4f}",
            })
    st.dataframe(pd.DataFrame(jaccard_data), use_container_width=True, hide_index=True)

    # SHAP consistency figure
    st.markdown("### SHAP Feature Importance Comparison")
    consistency_path = FIGURES_DIR / "paper" / f"{dataset}_shap_consistency.png"
    if consistency_path.exists():
        st.image(str(consistency_path), use_container_width=True)

    # Beeswarm viewer
    st.markdown("### 🐝 SHAP Beeswarm Viewer")
    st.markdown("Compare SHAP beeswarm plots across models and seeds.")

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        beeswarm_model = st.selectbox("Model", ["real", "ctgan", "tvae"], format_func=str.upper)
    with bcol2:
        beeswarm_seed = st.selectbox("Seed", SEEDS)

    beeswarm_path = FIGURES_DIR / "shap" / f"{dataset}_{beeswarm_model}_xgboost_seed{beeswarm_seed}_v1_shap_beeswarm.png"
    if beeswarm_path.exists():
        st.image(str(beeswarm_path), use_container_width=True,
                 caption=f"{DATASET_LABELS[dataset]} — {beeswarm_model.upper()} — Seed {beeswarm_seed}")
    else:
        st.warning(f"Beeswarm plot not found: {beeswarm_path.name}")

    # Heatmap
    st.markdown("### SHAP Rank Heatmap")
    heatmap_path = FIGURES_DIR / "paper" / f"{dataset}_shap_heatmap.png"
    if heatmap_path.exists():
        st.image(str(heatmap_path), use_container_width=True)


# ============================================================
# PAGE 4: PRIVACY METRICS
# ============================================================
elif page == "🔐 Privacy Metrics":
    st.markdown("# 🔐 Privacy Metrics")
    st.markdown("DCR, NNDR, MIA, and Inference Risk evaluation for synthetic data privacy.")

    dataset = st.selectbox("Select Dataset", DATASETS, format_func=lambda x: DATASET_LABELS[x])
    privacy = load_privacy(dataset)
    ir_key = dataset.replace("german_credit", "german")
    ir_data = load_inference_risk(ir_key if ir_key != dataset else dataset)

    st.markdown("---")

    # Privacy metric cards
    st.markdown("### Distance-Based Metrics (Mean Across 5 Seeds)")

    col1, col2, col3 = st.columns(3)
    ctgan_dcr = [r["dcr_mean"] for r in privacy["ctgan_runs"]]
    tvae_dcr = [r["dcr_mean"] for r in privacy["tvae_runs"]]

    col1.metric("CTGAN DCR", f"{sum(ctgan_dcr)/len(ctgan_dcr):.4f}", help="Higher = better privacy")
    col2.metric("TVAE DCR", f"{sum(tvae_dcr)/len(tvae_dcr):.4f}", help="Higher = better privacy")
    col3.metric("DCR Gap", f"{sum(ctgan_dcr)/len(ctgan_dcr) - sum(tvae_dcr)/len(tvae_dcr):.4f}",
                help="CTGAN - TVAE difference")

    st.markdown("---")

    # DCR per seed
    st.markdown("### DCR per Seed")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="CTGAN", x=[f"Seed {s}" for s in SEEDS], y=ctgan_dcr,
        marker_color=COLORS["ctgan"], opacity=0.9
    ))
    fig.add_trace(go.Bar(
        name="TVAE", x=[f"Seed {s}" for s in SEEDS], y=tvae_dcr,
        marker_color=COLORS["tvae"], opacity=0.9
    ))
    fig.update_layout(
        barmode="group", title=f"Mean DCR per Seed — {DATASET_LABELS[dataset]}",
        yaxis_title="DCR (higher = more private)", **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    # NNDR per seed
    st.markdown("### NNDR per Seed")
    ctgan_nndr = [r["nndr_mean"] for r in privacy["ctgan_runs"]]
    tvae_nndr = [r["nndr_mean"] for r in privacy["tvae_runs"]]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="CTGAN", x=[f"Seed {s}" for s in SEEDS], y=ctgan_nndr, marker_color=COLORS["ctgan"], opacity=0.9))
    fig2.add_trace(go.Bar(name="TVAE", x=[f"Seed {s}" for s in SEEDS], y=tvae_nndr, marker_color=COLORS["tvae"], opacity=0.9))
    fig2.update_layout(barmode="group", title=f"Mean NNDR per Seed — {DATASET_LABELS[dataset]}", yaxis_title="NNDR", **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    # MIA
    st.markdown("### MIA AUC per Seed")
    ctgan_mia = [r["mia_auc"] for r in privacy["ctgan_runs"]]
    tvae_mia = [r["mia_auc"] for r in privacy["tvae_runs"]]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="CTGAN", x=[f"Seed {s}" for s in SEEDS], y=ctgan_mia, marker_color=COLORS["ctgan"], opacity=0.9))
    fig3.add_trace(go.Bar(name="TVAE", x=[f"Seed {s}" for s in SEEDS], y=tvae_mia, marker_color=COLORS["tvae"], opacity=0.9))
    fig3.add_hline(y=0.5, line_dash="dash", line_color=COLORS["threshold"], annotation_text="Random Baseline (0.5)")
    fig3.update_layout(barmode="group", title=f"MIA AUC per Seed — {DATASET_LABELS[dataset]}", yaxis_title="MIA AUC", **PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Inference Risk
    st.markdown("### 🚨 Inference Risk Indicator")
    st.markdown("Is each synthetic record closer to a real person than that person's own nearest neighbor?")

    ir_ctgan = ir_data["ctgan"]
    ir_tvae = ir_data["tvae"]

    col1, col2 = st.columns(2)
    col1.metric(
        "CTGAN Inference Risk",
        f"{ir_ctgan['mean']:.4f}",
        delta="✅ PASS" if not ir_ctgan["exceeds_threshold"] else "❌ FAIL",
        delta_color="normal" if not ir_ctgan["exceeds_threshold"] else "inverse"
    )
    col2.metric(
        "TVAE Inference Risk",
        f"{ir_tvae['mean']:.4f}",
        delta="✅ PASS" if not ir_tvae["exceeds_threshold"] else "❌ FAIL",
        delta_color="normal" if not ir_tvae["exceeds_threshold"] else "inverse"
    )

    # Inference risk bar chart with threshold
    threshold = ir_ctgan["threshold"]  # same for both
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        name="CTGAN", x=[f"Seed {s}" for s in SEEDS], y=ir_ctgan["scores_per_seed"],
        marker_color=COLORS["ctgan"], opacity=0.9
    ))
    fig4.add_trace(go.Bar(
        name="TVAE", x=[f"Seed {s}" for s in SEEDS], y=ir_tvae["scores_per_seed"],
        marker_color=COLORS["tvae"], opacity=0.9
    ))
    fig4.add_hline(y=threshold, line_dash="dash", line_color=COLORS["threshold"],
                   annotation_text=f"Threshold ({threshold:.4f})")
    fig4.update_layout(
        barmode="group",
        title=f"Inference Risk per Seed — {DATASET_LABELS[dataset]}",
        yaxis_title="Inference Risk Score",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Privacy boxplots figure
    st.markdown("### Privacy Boxplots")
    boxplot_path = FIGURES_DIR / "privacy" / f"{dataset}_privacy_boxplots.png"
    if boxplot_path.exists():
        st.image(str(boxplot_path), use_container_width=True)

    # Privacy dashboard
    st.markdown("### Full Privacy Dashboard")
    dashboard_path = FIGURES_DIR / "privacy" / "privacy_dashboard.png"
    if dashboard_path.exists():
        st.image(str(dashboard_path), use_container_width=True)

    # Full detail table
    st.markdown("### Detailed Privacy Results (Per-Seed)")
    detail_rows = []
    for gen, runs in [("CTGAN", privacy["ctgan_runs"]), ("TVAE", privacy["tvae_runs"])]:
        for run in runs:
            detail_rows.append({
                "Generator": gen,
                "Seed": run["seed"],
                "DCR Mean": f"{run['dcr_mean']:.4f}",
                "DCR 5th pct": f"{run['dcr_5th_percentile']:.4f}",
                "NNDR Mean": f"{run['nndr_mean']:.4f}",
                "MIA AUC": f"{run['mia_auc']:.4f}",
            })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 5: STATISTICAL TESTS
# ============================================================
elif page == "📊 Statistical Tests":
    st.markdown("# 📊 Statistical Significance Tests")
    st.markdown("Wilcoxon signed-rank tests and Cohen's d effect sizes across all metrics.")

    dataset = st.selectbox("Select Dataset", DATASETS, format_func=lambda x: DATASET_LABELS[x])
    stats = load_stats(dataset)

    st.markdown("---")

    # Explanation
    st.warning(
        "**The N=5 Problem:** With only 5 seeds, the Wilcoxon signed-rank test has a hard floor: "
        "p_min = 1/2⁴ = **0.0625**. Even when one method wins all 5 seeds, p < 0.05 is impossible. "
        "**Cohen's d effect size** is our primary discriminator."
    )

    st.markdown("### Effect Size Guide")
    guide_cols = st.columns(4)
    guide_cols[0].markdown("**|d| < 0.2**: Negligible")
    guide_cols[1].markdown("**|d| = 0.2–0.5**: Small")
    guide_cols[2].markdown("**|d| = 0.5–0.8**: Medium")
    guide_cols[3].markdown("**|d| > 0.8**: Large ✦")

    st.markdown("---")

    # All paired tests
    st.markdown("### Paired Statistical Tests (TVAE vs CTGAN)")
    paired_tests = stats["paired_tests"]

    test_rows = []
    for key, test in paired_tests.items():
        d = test["cohens_d"]
        if abs(d) > 0.8:
            effect = "Large ✦"
        elif abs(d) > 0.5:
            effect = "Medium"
        elif abs(d) > 0.2:
            effect = "Small"
        else:
            effect = "Negligible"

        test_rows.append({
            "Comparison": test["comparison"],
            "Wilcoxon Stat": f"{test['wilcoxon_stat']:.1f}",
            "p-value": f"{test['p_value']:.4f}",
            "Cohen's d": f"{d:.2f}",
            "Effect Size": effect,
        })

    st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)

    # Cohen's d visualization
    st.markdown("### Cohen's d Effect Sizes (Visual)")
    comparisons = [t["comparison"] for t in paired_tests.values()]
    cohens_ds = [t["cohens_d"] for t in paired_tests.values()]

    fig = go.Figure()
    colors = [COLORS["pass"] if d > 0 else COLORS["fail"] for d in cohens_ds]
    fig.add_trace(go.Bar(
        x=cohens_ds, y=comparisons, orientation="h",
        marker_color=colors, opacity=0.9,
        text=[f"{d:.2f}" for d in cohens_ds],
        textposition="auto"
    ))
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.add_vline(x=0.8, line_dash="dash", line_color=COLORS["threshold"], annotation_text="Large effect (0.8)")
    fig.add_vline(x=-0.8, line_dash="dash", line_color=COLORS["threshold"], annotation_text="Large effect (-0.8)")
    fig.update_layout(
        title=f"Cohen's d Effect Sizes — {DATASET_LABELS[dataset]}",
        xaxis_title="Cohen's d (positive = TVAE higher)",
        yaxis_title="",
        height=400,
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw values comparison
    st.markdown("---")
    st.markdown("### Raw Metric Values per Seed")

    metric_key = st.selectbox(
        "Select Metric",
        ["auc", "f1", "shap", "dcr", "nndr", "mia"],
        format_func=lambda x: {
            "auc": "ROC-AUC", "f1": "F1-Score", "shap": "SHAP ρ",
            "dcr": "DCR", "nndr": "NNDR", "mia": "MIA AUC"
        }[x]
    )

    ctgan_key = f"ctgan_{metric_key}"
    tvae_key = f"tvae_{metric_key}"

    if ctgan_key in stats["metrics_summary"] and tvae_key in stats["metrics_summary"]:
        ctgan_vals = stats["metrics_summary"][ctgan_key]["raw_values"]
        tvae_vals = stats["metrics_summary"][tvae_key]["raw_values"]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[f"Seed {s}" for s in SEEDS], y=ctgan_vals, mode="lines+markers",
            name="CTGAN", line=dict(color=COLORS["ctgan"], width=3), marker=dict(size=10)
        ))
        fig2.add_trace(go.Scatter(
            x=[f"Seed {s}" for s in SEEDS], y=tvae_vals, mode="lines+markers",
            name="TVAE", line=dict(color=COLORS["tvae"], width=3), marker=dict(size=10)
        ))
        fig2.update_layout(
            title=f"Per-Seed Comparison — {metric_key.upper()}",
            yaxis_title=metric_key.upper(),
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig2, use_container_width=True)
