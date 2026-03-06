"""
Credit Card Fraud Detection — Monitoring Dashboard
====================================================
Streamlit dashboard with model metrics, confusion matrices,
ROC/PR curves, SHAP feature importance, and a live API test panel.

Run:
    streamlit run dashboard/app.py
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Bölüm seçin:",
    [
        "Model Metrikleri",
        "Confusion Matrix",
        "ROC & PR Eğrileri",
        "SHAP Feature Importance",
        "Canlı Test Paneli",
    ],
)

# ──────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_metrics():
    """Load the final model comparison CSV."""
    path = os.path.join(OUTPUT_DIR, "final_comparison.csv")
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_model_config():
    """Load model config (version, threshold, best params)."""
    import json
    config_path = os.path.join(MODEL_DIR, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_registry_version():
    """Read production model version from MLflow registry."""
    try:
        import mlflow
        mlruns_dir = os.path.join(BASE_DIR, "mlruns")
        tracking_uri = f"file:///{mlruns_dir.replace(os.sep, '/')}"
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias("CreditCardFraudDetector", "production")
        return str(mv.version)
    except Exception:
        return "?"  


@st.cache_data
def load_test_data():
    """Load a sample of the test parquet for SHAP / live panel."""
    path = os.path.join(DATA_DIR, "test.parquet")
    return pd.read_parquet(path)


@st.cache_resource
def load_scaler():
    """Load the fitted scaler for inverse-transforming sample data."""
    import joblib
    return joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))


COLS_TO_SCALE = [
    "Amount", "Time", "Time_in_day", "Amount_log",
    "Time_Amount", "Time_Amount_sq", "Amount_per_Time",
]


def inverse_transform_sample(row: pd.Series) -> pd.Series:
    """Inverse-transform scaled columns to get raw Time & Amount."""
    scaler = load_scaler()
    scaled = np.zeros((1, len(COLS_TO_SCALE)))
    for i, col in enumerate(COLS_TO_SCALE):
        if col in row.index:
            scaled[0, i] = row[col]
    raw = scaler.inverse_transform(scaled)
    result = row.copy()
    for i, col in enumerate(COLS_TO_SCALE):
        if col in result.index:
            result[col] = raw[0, i]
    return result


@st.cache_resource
def load_shap_artifacts():
    """Load model + compute SHAP values (cached)."""
    import shap
    import joblib

    model = joblib.load(os.path.join(MODEL_DIR, "optimized_model.joblib"))

    test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    features = [c for c in test.columns if c != "Class"]
    sample = test[features].sample(n=500, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    return shap_values, sample, features


# ──────────────────────────────────────────────────────────────────────
# 1. Model Metrikleri
# ──────────────────────────────────────────────────────────────────────

if page == "Model Metrikleri":
    st.title("📈 Model Performans Özeti")

    # Model version & config info
    config = load_model_config()
    registry_ver = load_registry_version()

    v1, v2, v3 = st.columns(3)
    v1.info(f"📦 **MLflow Registry Versiyon:** {registry_ver}")
    v2.info(f"🎯 **Threshold:** {config.get('best_threshold', '?')}")
    best_params = config.get("best_params", {})
    params_str = ", ".join(f"{k}={v}" for k, v in best_params.items()) if best_params else "?"
    v3.info(f"⚙️ **Optimized Params:** {params_str}")

    st.markdown("Tüm modellerin test seti üzerindeki karşılaştırmalı metrikleri.")

    df = load_metrics()

    # Highlight best model
    best_idx = df["PR-AUC"].idxmax()
    best_model = df.loc[best_idx, "Model"]
    st.success(f"🏆 En iyi model (PR-AUC): **{best_model}**")

    # Metric cards per model
    for _, row in df.iterrows():
        model_name = row["Model"]
        is_best = model_name == best_model

        st.subheader(f"{'⭐ ' if is_best else ''}{model_name}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}")
        c2.metric("PR-AUC", f"{row['PR-AUC']:.4f}")
        c3.metric("Accuracy", f"{row['Accuracy']:.4f}")
        c4.metric("Precision (Fraud)", f"{row['Precision (Fraud)']:.4f}")
        c5.metric("Recall (Fraud)", f"{row['Recall (Fraud)']:.4f}")
        c6.metric("F1 (Fraud)", f"{row['F1 (Fraud)']:.4f}")
        st.divider()

    # Full table
    st.subheader("Detaylı Tablo")
    st.dataframe(df.style.highlight_max(axis=0, subset=df.columns[1:]), width='stretch')


# ──────────────────────────────────────────────────────────────────────
# 2. Confusion Matrix
# ──────────────────────────────────────────────────────────────────────

elif page == "Confusion Matrix":
    st.title("🔢 Confusion Matrix")

    cm_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.startswith("cm_") and f.endswith(".png")]
    )

    if not cm_files:
        st.warning("outputs/ dizininde confusion matrix görseli bulunamadı.")
    else:
        labels = {f: f.replace("cm_", "").replace(".png", "").replace("_", " ") for f in cm_files}
        selected = st.selectbox("Model seçin:", cm_files, format_func=lambda x: labels[x])
        img_path = os.path.join(OUTPUT_DIR, selected)
        st.image(img_path, width='stretch')


# ──────────────────────────────────────────────────────────────────────
# 3. ROC & PR Eğrileri
# ──────────────────────────────────────────────────────────────────────

elif page == "ROC & PR Eğrileri":
    st.title("📉 ROC & Precision-Recall Eğrileri")

    tab1, tab2, tab3 = st.tabs(["Tüm Modeller", "Optimized Dahil", "Bireysel"])

    with tab1:
        col1, col2 = st.columns(2)
        roc_path = os.path.join(OUTPUT_DIR, "roc_combined.png")
        pr_path = os.path.join(OUTPUT_DIR, "pr_combined.png")
        if os.path.exists(roc_path):
            col1.image(roc_path, caption="ROC Curve — Tüm Modeller", width='stretch')
        if os.path.exists(pr_path):
            col2.image(pr_path, caption="PR Curve — Tüm Modeller", width='stretch')

    with tab2:
        col1, col2 = st.columns(2)
        roc_path = os.path.join(OUTPUT_DIR, "roc_final_combined.png")
        pr_path = os.path.join(OUTPUT_DIR, "pr_final_combined.png")
        if os.path.exists(roc_path):
            col1.image(roc_path, caption="ROC Curve — Optimized Dahil", width='stretch')
        if os.path.exists(pr_path):
            col2.image(pr_path, caption="PR Curve — Optimized Dahil", width='stretch')

    with tab3:
        col1, col2 = st.columns(2)
        roc_path = os.path.join(OUTPUT_DIR, "roc_individual.png")
        pr_path = os.path.join(OUTPUT_DIR, "pr_individual.png")
        if os.path.exists(roc_path):
            col1.image(roc_path, caption="ROC Curves — Bireysel", width='stretch')
        if os.path.exists(pr_path):
            col2.image(pr_path, caption="PR Curves — Bireysel", width='stretch')


# ──────────────────────────────────────────────────────────────────────
# 4. SHAP Feature Importance
# ──────────────────────────────────────────────────────────────────────

elif page == "SHAP Feature Importance":
    st.title("🧠 SHAP Global Feature Importance")
    st.markdown(
        "Optimized Random Forest modeli üzerinde 500 test örneği ile hesaplanmış "
        "SHAP değerleri. Hangi özellikler fraud kararını en çok etkiliyor?"
    )

    with st.spinner("SHAP değerleri hesaplanıyor (ilk yüklemede biraz sürebilir)..."):
        shap_values, sample, feature_names = load_shap_artifacts()

    # Handle different SHAP return formats:
    #   - Old API: list of [class_0, class_1], each (n_samples, n_features)
    #   - New API: ndarray (n_samples, n_features, n_classes)
    #   - Explanation object with .values attribute
    if isinstance(shap_values, list):
        sv = shap_values[1]  # fraud class
    elif hasattr(shap_values, "values"):
        sv = shap_values.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Bar plot — mean |SHAP|
    st.subheader("Ortalama |SHAP| Değerleri (Top 20)")
    mean_abs = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(20),
        mean_abs[sorted_idx][::-1],
        color="#ff6b6b",
    )
    ax.set_yticks(range(20))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Feature Importance (SHAP)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Beeswarm via shap library
    st.subheader("SHAP Beeswarm Plot")
    import shap

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv,
        sample.values if hasattr(sample, "values") else sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    st.pyplot(plt.gcf())
    plt.close("all")


# ──────────────────────────────────────────────────────────────────────
# 5. Canlı Test Paneli
# ──────────────────────────────────────────────────────────────────────

elif page == "Canlı Test Paneli":
    st.title("🧪 Canlı Fraud Testi")

    config = load_model_config()
    registry_ver = load_registry_version()
    st.caption(
        f"Model v{registry_ver} · Threshold: {config.get('best_threshold', '?')} · "
        f"API: `{API_URL}`"
    )
    st.markdown(
        f"Aşağıdaki alanları doldurup **Tahmin Et** butonuna basın. "
        f"API'ye (`{API_URL}/predict`) istek gönderilir ve sonuç ekranda gösterilir."
    )

    test_df = load_test_data()
    features = [c for c in test_df.columns if c != "Class"]

    # ── Sample loaders ──
    col_btn1, col_btn2 = st.columns(2)
    if "sample_values" not in st.session_state:
        st.session_state.sample_values = None

    with col_btn1:
        if st.button("🔴 Rastgele Fraud Yükle"):
            fraud_rows = test_df[test_df["Class"] == 1]
            sample = fraud_rows.sample(n=1, random_state=None).iloc[0]
            st.session_state.sample_values = inverse_transform_sample(sample)

    with col_btn2:
        if st.button("🟢 Rastgele Normal Yükle"):
            normal_rows = test_df[test_df["Class"] == 0]
            sample = normal_rows.sample(n=1, random_state=None).iloc[0]
            st.session_state.sample_values = inverse_transform_sample(sample)

    # ── Input form ──
    pca_features = [f"V{i}" for i in range(1, 29)]
    other_features = ["Time", "Amount"]

    sv = st.session_state.sample_values

    with st.form("predict_form"):
        st.subheader("PCA Bileşenleri (V1-V28)")
        cols = st.columns(4)
        values = {}
        for i, feat in enumerate(pca_features):
            default_val = float(sv[feat]) if sv is not None and feat in sv.index else 0.0
            values[feat] = cols[i % 4].number_input(feat, value=default_val, format="%.6f")

        st.subheader("İşlem Bilgileri")
        col_t, col_a = st.columns(2)
        default_time = float(sv["Time"]) if sv is not None and "Time" in sv.index else 0.0
        default_amount = float(sv["Amount"]) if sv is not None and "Amount" in sv.index else 0.0
        values["Time"] = col_t.number_input("Time (saniye)", value=default_time, format="%.2f")
        values["Amount"] = col_a.number_input("Amount ($)", value=default_amount, format="%.2f")

        submitted = st.form_submit_button("🚀 Tahmin Et", width='stretch')

    if submitted:
        payload = {"features": values}
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                prob = result["fraud_probability"]
                is_fraud = result["is_fraud"]
                risk = result["risk_level"]

                st.divider()
                st.subheader("Sonuç")
                r1, r2, r3 = st.columns(3)

                if is_fraud:
                    r1.error(f"🚨 **FRAUD**")
                else:
                    r1.success(f"✅ **NORMAL**")

                r2.metric("Fraud Olasılığı", f"{prob:.4f}")

                if risk == "HIGH":
                    r3.error(f"⚠️ Risk: **{risk}**")
                elif risk == "MEDIUM":
                    r3.warning(f"⚠️ Risk: **{risk}**")
                else:
                    r3.success(f"Risk: **{risk}**")

                with st.expander("API Yanıtı (JSON)"):
                    st.json(result)
            else:
                st.error(f"API Hatası: {resp.status_code} — {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                f"API'ye bağlanılamadı (`{API_URL}`). "
                "API sunucusunun çalıştığından emin olun: `uvicorn api.main:app --reload`"
            )
        except requests.exceptions.Timeout:
            st.error("API isteği zaman aşımına uğradı.")
