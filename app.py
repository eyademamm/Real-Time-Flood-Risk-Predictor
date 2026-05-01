"""
app.py
------
Streamlit dashboard for the Flood Early Warning & Response System.

Pages:
  1. Dashboard    — live-style alert overview and key metrics
  2. Data Explorer — EDA: distributions, correlations, class balance
  3. Model Report  — confusion matrix, ROC curve, feature importance, comparison
  4. Predictor     — interactive sliders → real-time flood probability

Run:
    streamlit run app.py
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from data_loader import download_dataset, preprocess, FEATURE_COLS, BINARY_TARGET

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f0f4f8;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
  }
  .alert-normal   { background:#e1f5ee; border-left:4px solid #1D9E75; padding:12px; border-radius:6px; color:black; }
  .alert-watch    { background:#faeeda; border-left:4px solid #EF9F27; padding:12px; border-radius:6px; color:black; }
  .alert-warning  { background:#fef3cd; border-left:4px solid #BA7517; padding:12px; border-radius:6px; color:black; }
  .alert-emergency{ background:#fcebeb; border-left:4px solid #E24B4A; padding:12px; border-radius:6px; color:black; }
</style>
""", unsafe_allow_html=True)


# ── Cached data & model loading ────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = download_dataset()
    return df

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model_path  = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    fcols_path  = "models/feature_cols.pkl"
    if not all(os.path.exists(p) for p in [model_path, scaler_path, fcols_path]):
        return None, None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    fcols  = joblib.load(fcols_path)
    return model, scaler, fcols

def load_metrics():
    path = "reports/metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3140/3140445.png", width=60)
    st.title("Flood EWS")
    st.caption("Early Warning & Response System")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠  Dashboard", "📊  Data Explorer", "🤖  Model Report", "🔮  Flood Predictor"],
    )
    st.divider()
    st.caption("Smart Cities Course Project\nPowered by scikit-learn & XGBoost")

    if not os.path.exists("models/best_model.pkl"):
        st.warning("Model not trained yet.")
        if st.button("Train Model Now"):
            with st.spinner("Training… (this may take a few minutes)"):
                from train import train_and_evaluate
                train_and_evaluate()
                st.success("Training complete!")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.title("🌊 Flood Early Warning & Response System")
    st.caption("Smart City Monitoring — Delta Basin Network")
    st.divider()

    # ── Alert level selector (simulation) ─────────────────────────────────────
    st.subheader("Current Alert Status")
    alert_col, info_col = st.columns([2, 3])

    with alert_col:
        level = st.selectbox(
            "Simulate alert level",
            ["NORMAL", "WATCH", "WARNING", "EMERGENCY"],
            index=1,
        )

    ALERT_CONFIG = {
        "NORMAL":    ("🟢", "alert-normal",    "No flood risk detected. All sensors within safe limits."),
        "WATCH":     ("🟡", "alert-watch",     "Rising water levels observed. Monitoring closely."),
        "WARNING":   ("🟠", "alert-warning",   "Flood conditions developing. Agencies notified."),
        "EMERGENCY": ("🔴", "alert-emergency", "Active flooding in progress. Evacuations underway."),
    }
    icon, css_class, msg = ALERT_CONFIG[level]
    with info_col:
        st.markdown(
            f'<div class="{css_class}"><strong>{icon} {level}</strong> — {msg}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Key metrics ────────────────────────────────────────────────────────────
    SCENARIO_METRICS = {
        "NORMAL":    dict(wl=1.2, rain=8,  flow=180,  soil=34, stations_ok=5, risk_pct=12),
        "WATCH":     dict(wl=2.4, rain=22, flow=390,  soil=58, stations_ok=4, risk_pct=45),
        "WARNING":   dict(wl=3.6, rain=38, flow=650,  soil=78, stations_ok=2, risk_pct=72),
        "EMERGENCY": dict(wl=5.1, rain=55, flow=1100, soil=95, stations_ok=1, risk_pct=94),
    }
    m = SCENARIO_METRICS[level]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Water Level",    f"{m['wl']} m",     delta="+0.3m/hr" if level != "NORMAL" else "Stable")
    c2.metric("Rainfall Rate",  f"{m['rain']} mm/hr")
    c3.metric("River Flow",     f"{m['flow']} m³/s")
    c4.metric("Soil Saturation",f"{m['soil']}%",    delta="Rising" if level != "NORMAL" else None)
    c5.metric("Stations Online",f"{m['stations_ok']}/5")
    c6.metric("Flood Risk",     f"{m['risk_pct']}%")

    st.divider()

    # ── Simulated 24h water-level trend ───────────────────────────────────────
    st.subheader("Water Level Trend — Last 24 Hours")
    hours = list(range(0, 25))
    base  = m["wl"]
    np.random.seed(42)
    levels = np.clip(
        base * 0.4 + np.linspace(0, base * 0.6, 25) + np.random.normal(0, 0.05, 25),
        0.3, 8
    )
    threshold_warn  = [3.5] * 25
    threshold_crit  = [5.0] * 25

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=levels, name="Water Level",
                             line=dict(color="#1D9E75", width=2.5), fill="tozeroy",
                             fillcolor="rgba(29,158,117,0.08)"))
    fig.add_trace(go.Scatter(x=hours, y=threshold_warn, name="Warning (3.5m)",
                             line=dict(color="#EF9F27", dash="dash", width=1.5)))
    fig.add_trace(go.Scatter(x=hours, y=threshold_crit, name="Critical (5.0m)",
                             line=dict(color="#E24B4A", dash="dot", width=1.5)))
    fig.update_layout(
        xaxis_title="Hour", yaxis_title="Water Level (m)",
        height=320, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

    # ── Response actions ───────────────────────────────────────────────────────
    ACTIONS = {
        "NORMAL":    ["Continue standard monitoring", "Schedule weekly sensor maintenance", "Review flood model parameters"],
        "WATCH":     ["Notify emergency management", "Increase sensor polling to 5-min intervals", "Pre-position flood barriers at zones B3, B7"],
        "WARNING":   ["Issue public alert via SMS and broadcast", "Activate emergency response teams", "Deploy mobile pumping units"],
        "EMERGENCY": ["Mandatory evacuation — zones R4, R5, R9", "Contact national disaster authority", "Open emergency shelters at City Hall & North School"],
    }
    st.subheader("Recommended Response Actions")
    for i, action in enumerate(ACTIONS[level], 1):
        color = "#E24B4A" if level == "EMERGENCY" else "#BA7517" if level == "WARNING" else "#1D9E75"
        st.markdown(f"**{i}.** {action}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Data Explorer":
    st.title("📊 Dataset Explorer")
    st.caption("Public dataset: Flood Prediction Factors — Kaggle (naiyakhalid)")
    st.link_button("View on Kaggle", "https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-factors")
    st.divider()

    df = load_data()
    _, _, _, _, _, df_clean, feature_cols = preprocess(df)

    # ── Overview ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",   f"{len(df_clean):,}")
    c2.metric("Features",     len(feature_cols))
    c3.metric("Flood Cases",  f"{df_clean[BINARY_TARGET].sum():,}")
    c4.metric("No-Flood Cases", f"{(df_clean[BINARY_TARGET]==0).sum():,}")

    st.subheader("Sample Data")
    st.dataframe(df_clean[feature_cols + [BINARY_TARGET, "FloodProbability"]].head(100),
                 use_container_width=True, height=260)

    st.divider()

    # ── Class balance ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Class Balance")
        vc = df_clean[BINARY_TARGET].value_counts().reset_index()
        vc.columns = ["Label", "Count"]
        vc["Label"] = vc["Label"].map({1: "Flood", 0: "No Flood"})
        fig_pie = px.pie(vc, names="Label", values="Count",
                         color="Label",
                         color_discrete_map={"Flood": "#E24B4A", "No Flood": "#1D9E75"})
        fig_pie.update_layout(height=300, margin=dict(t=20, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Flood Probability Distribution")
        fig_hist = px.histogram(df_clean, x="FloodProbability", nbins=40,
                                color_discrete_sequence=["#378ADD"])
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#E24B4A",
                           annotation_text="Threshold (0.5)")
        fig_hist.update_layout(height=300, margin=dict(t=20, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── Feature distribution ───────────────────────────────────────────────────
    st.subheader("Feature Distribution by Flood Label")
    selected_feature = st.selectbox("Select feature", feature_cols)
    fig_box = px.box(df_clean, x=BINARY_TARGET, y=selected_feature,
                     color=BINARY_TARGET,
                     color_discrete_map={0: "#1D9E75", 1: "#E24B4A"},
                     labels={BINARY_TARGET: "Flood (1) / No Flood (0)"})
    fig_box.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # ── Correlation heatmap ────────────────────────────────────────────────────
    st.subheader("Feature Correlation Heatmap")
    sample = df_clean[feature_cols].sample(min(5000, len(df_clean)), random_state=42)
    corr = sample.corr()
    fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, aspect="auto")
    fig_heat.update_layout(height=560, margin=dict(t=20, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Report":
    st.title("🤖 ML Model Report")
    st.divider()

    metrics = load_metrics()
    if metrics is None:
        st.warning("No trained model found. Go to the sidebar and click **Train Model Now**.")
        st.stop()

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.subheader(f"Best Model: {metrics['best_model_name']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",    f"{metrics['accuracy']*100:.2f}%")
    c2.metric("F1 Score",    f"{metrics['f1_score']:.4f}")
    c3.metric("ROC-AUC",     f"{metrics['roc_auc']:.4f}")
    c4.metric("5-Fold CV",   f"{metrics['cv_accuracy']*100:.2f}%")

    st.divider()

    # ── Model comparison table ─────────────────────────────────────────────────
    st.subheader("Model Comparison")
    rows = []
    for mname, mvals in metrics["all_models"].items():
        rows.append({
            "Model":       mname,
            "Accuracy":    f"{mvals['accuracy']*100:.2f}%",
            "F1 Score":    f"{mvals['f1']:.4f}",
            "ROC-AUC":     f"{mvals['auc']:.4f}",
            "CV Accuracy": f"{mvals['cv_acc']*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Report images ──────────────────────────────────────────────────────────
    img_col1, img_col2 = st.columns(2)
    for path, title, col in [
        ("reports/confusion_matrix.png",  "Confusion Matrix",       img_col1),
        ("reports/roc_curves.png",        "ROC Curves",             img_col2),
        ("reports/feature_importance.png","Feature Importance",     img_col1),
        ("reports/model_comparison.png",  "Model Comparison Chart", img_col2),
    ]:
        if os.path.exists(path):
            with col:
                st.subheader(title)
                st.image(path, use_column_width=True)
        else:
            with col:
                st.info(f"{title} — run training to generate this plot.")

    # ── ML Pipeline summary ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Pipeline Summary")
    st.markdown("""
| Step | Detail |
|------|--------|
| **Dataset** | Flood Prediction Factors (Kaggle — naiyakhalid) |
| **Features** | 20 risk-factor scores (scale 1–10) |
| **Target** | Binary flood label (FloodProbability ≥ 0.5 → Flood) |
| **Preprocessing** | StandardScaler + stratified 80/20 split |
| **Models** | Random Forest (200 trees) and XGBoost (200 estimators) |
| **Selection** | Highest ROC-AUC on held-out test set |
| **Evaluation** | Accuracy, F1, AUC-ROC, 5-fold cross-validation |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FLOOD PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Flood Predictor":
    st.title("🔮 Real-Time Flood Risk Predictor")
    st.caption("Adjust the risk factor sliders below to get an instant ML prediction.")
    st.divider()

    model, scaler, feature_cols = load_model()
    if model is None:
        st.warning("No trained model found. Go to the sidebar and click **Train Model Now**.")
        st.stop()

    # ── Feature sliders ────────────────────────────────────────────────────────
    DESCRIPTIONS = {
        "MonsoonIntensity":               "Intensity of monsoon / heavy rainfall season",
        "TopographyDrainage":             "Terrain's natural drainage capacity (lower = worse)",
        "RiverManagement":                "Quality of river management infrastructure",
        "Deforestation":                  "Extent of deforestation in the watershed",
        "Urbanization":                   "Level of urban development and impervious surfaces",
        "ClimateChange":                  "Climate-change-driven rainfall variability",
        "DamsQuality":                    "Structural quality of dams and reservoirs",
        "Siltation":                      "Sediment buildup blocking waterways",
        "AgriculturalPractices":          "Flood-risk impact of farming practices",
        "Encroachments":                  "Illegal encroachments on flood plains",
        "IneffectiveDisasterPreparedness":"Lack of emergency preparedness",
        "DrainageSystems":                "Capacity of city drainage systems",
        "CoastalVulnerability":           "Exposure to coastal or tidal flooding",
        "Landslides":                     "Landslide-triggered flood risk",
        "Watersheds":                     "Watershed degradation level",
        "DeterioratingInfrastructure":    "Aging or damaged flood protection infrastructure",
        "PopulationScore":                "Population density in flood-prone areas",
        "WetlandLoss":                    "Loss of natural flood-absorbing wetlands",
        "InadequatePlanning":             "Poor urban planning for flood resilience",
        "PoliticalFactors":               "Political/governance failures in flood management",
    }

    input_vals = {}
    col1, col2 = st.columns(2)
    for i, col in enumerate(feature_cols):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            input_vals[col] = st.slider(
                label=col,
                min_value=1, max_value=10, value=5,
                help=DESCRIPTIONS.get(col, col),
            )

    st.divider()

    # ── Predict ────────────────────────────────────────────────────────────────
    X_input = np.array([[input_vals[c] for c in feature_cols]])
    X_scaled = scaler.transform(X_input)
    prob  = model.predict_proba(X_scaled)[0][1]
    label = model.predict(X_scaled)[0]

    # Determine alert level
    if prob >= 0.80:
        alert_level = "EMERGENCY"
        alert_color = "#E24B4A"
        alert_icon  = "🔴"
        advice = "Immediate evacuation required. Contact emergency services."
    elif prob >= 0.60:
        alert_level = "WARNING"
        alert_color = "#BA7517"
        alert_icon  = "🟠"
        advice = "Flood conditions likely. Prepare barriers and notify residents."
    elif prob >= 0.40:
        alert_level = "WATCH"
        alert_color = "#EF9F27"
        alert_icon  = "🟡"
        advice = "Elevated risk. Monitor sensors closely and pre-position resources."
    else:
        alert_level = "NORMAL"
        alert_color = "#1D9E75"
        alert_icon  = "🟢"
        advice = "Low flood risk. Continue standard monitoring protocol."

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.subheader("Prediction Result")
        st.markdown(
            f"<h1 style='color:{alert_color};font-size:3rem;'>{alert_icon} {alert_level}</h1>",
            unsafe_allow_html=True,
        )
        st.metric("Flood Probability", f"{prob*100:.1f}%")
        st.caption(advice)

    with res_col2:
        st.subheader("Risk Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            delta={"reference": 50, "suffix": "%"},
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": alert_color},
                "steps": [
                    {"range": [0,  40], "color": "#e1f5ee"},
                    {"range": [40, 60], "color": "#faeeda"},
                    {"range": [60, 80], "color": "#fef3cd"},
                    {"range": [80,100], "color": "#fcebeb"},
                ],
                "threshold": {"line": {"color": "#E24B4A", "width": 3}, "value": 80},
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=20, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Top contributing features ──────────────────────────────────────────────
    st.divider()
    st.subheader("Top Risk Factors for This Prediction")
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": imps,
                                 "Your Input": [input_vals[c] for c in feature_cols]})
        feat_df["Weighted Risk"] = feat_df["Importance"] * feat_df["Your Input"]
        feat_df = feat_df.sort_values("Weighted Risk", ascending=False).head(8)

        fig_bar = px.bar(feat_df, x="Weighted Risk", y="Feature", orientation="h",
                         color="Weighted Risk",
                         color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"])
        fig_bar.update_layout(height=350, showlegend=False,
                              margin=dict(t=10, b=0), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)
