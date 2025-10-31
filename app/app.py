import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib
import streamlit as st

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="Flight Price Predictor ✈️",
    page_icon="💎",
    layout="wide",
)

# ========================
# CUSTOM STYLING (CSS)
# ========================
st.markdown(
    """
    <style>
    body { background: radial-gradient(circle at top left, #0f172a, #1e293b, #0f172a); color: white; font-family: 'Poppins', sans-serif; }
    .main { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; box-shadow: 0 10px 50px rgba(0,0,0,0.4); animation: fadeIn 1s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    h1, h2, h3 { text-align: center; color: #000000; font-weight: 600; }
    h1 span { background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton>button { background: linear-gradient(90deg, #2563eb, #7c3aed); color: white; border-radius: 10px; height: 3em; font-weight: 600; transition: 0.3s; width: 100%; border: none; }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 25px rgba(147, 197, 253, 0.6); }
    .prediction-box { text-align: center; background: rgba(255,255,255,0.12); padding: 25px; border-radius: 15px; margin-top: 40px; font-size: 1.3rem; font-weight: 600; color: #10b981; backdrop-filter: blur(6px); }
    .cards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 50px !important; margin-top: 30px; margin-bottom: 40px; padding: 10px; }
    .card { background: rgba(255,255,255,0.07); padding: 25px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); transition: all 0.3s ease; }
    .card:hover { transform: translateY(-6px); box-shadow: 0 10px 30px rgba(56, 189, 248, 0.3); }
    .card h3 { color: #93c5fd; font-weight: 700; margin-bottom: 10px; text-align: center; }
    .card p, .card ul { font-size: 0.95rem; line-height: 1.5; }
    .card ul { list-style: none; padding-left: 0; }
    .card ul li { margin-bottom: 5px; }
    hr { border: 1px solid rgba(255,255,255,0.15); margin: 30px 0; }
    .stPlotlyChart, .stPyplot { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; box-shadow: 0 0 30px rgba(0,0,0,0.2); margin-top: 25px; }
    div[data-testid="stMetric"] { background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 20px; padding: 25px 35px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; box-shadow: 0 0 25px rgba(56, 189, 248, 0.2); height: 150px; }
    div[data-testid="stMetric"] label { font-size: 1.3rem !important; font-weight: 600 !important; color: #93c5fd !important; }
    div[data-testid="stMetric"] div { font-size: 2rem !important; font-weight: 800 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================
# PATHS
# ========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (BASE_DIR / "../models").resolve()

# ========================
# LOAD MODEL & METADATA
# ========================
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = MODEL_DIR / "flight_price_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model_data = joblib.load(model_path)
    return model_data["model"], model_data["preprocessor"]


@st.cache_data(show_spinner=False)
def load_metadata():
    meta_path = MODEL_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"r2": 0.0, "mae": 0.0, "mse": 0.0}


try:
    model, preprocessor = load_model()
    metadata = load_metadata()
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

# ========================
# SHAP & LIME HELPERS
# ========================
@st.cache_resource(show_spinner=False)
def compute_shap_values(_model, X_sample):
    try:
        explainer = shap.Explainer(_model)
        shap_values = explainer(X_sample)
        return shap_values
    except Exception as exc:
        raise RuntimeError(f"SHAP hesaplanamadı: {exc}")


@st.cache_resource(show_spinner=False)
def explain_with_lime(_model, X_train, X_sample, feature_names):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        verbose=False,
        mode="regression",
    )
    exp = explainer.explain_instance(X_sample, _model.predict, num_features=8)
    return exp

# ========================
# HERO SECTION
# ========================
st.markdown("<h1>✈️ <span>Flight Price Predictor </span>✈️</h1>", unsafe_allow_html=True)
st.markdown("<h3>Using Machine Learning to estimate price for flights</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ========================
# INPUT FORM
# ========================
st.markdown("<h3 style='text-align:center;'>Enter Flight Details</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox("Airline", ["IndiGo", "Air India", "Vistara", "GO_FIRST", "AirAsia", "SpiceJet"])
    source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
    departure_time = st.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
    stops = st.number_input("Number of Stops", min_value=0, max_value=3, step=1)
with col2:
    destination_city = st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
    arrival_time = st.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
    flight_class = st.selectbox("Class", ["Economy", "Business"])
    duration = st.number_input("Flight Duration (hours)", min_value=0.5, max_value=10.0, step=0.5)
    days_left = st.number_input("Days Left Before Departure", min_value=1, max_value=60, step=1)

# ========================
# PREDICTION
# ========================
if st.button("💰 Predict Flight Price"):
    try:
        input_df = pd.DataFrame(
            [
                {
                    "airline": airline,
                    "source_city": source_city,
                    "departure_time": departure_time,
                    "arrival_time": arrival_time,
                    "destination_city": destination_city,
                    "class": flight_class,
                    "stops": stops,
                    "duration": duration,
                    "days_left": days_left,
                }
            ]
        )
        transformed = preprocessor.transform(input_df)
        pred = float(model.predict(transformed)[0])
        st.markdown(
            f"<div class='prediction-box'>Estimated Ticket Price: ₹{pred:,.2f}</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ========================
# METRICS SECTION
# ========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 📊 Model Performance Metrics")
st.caption("Metrics computed on the held-out test split (30%) after hyperparameter tuning.")

c1, c2, c3 = st.columns(3)
c1.metric("R² Score", f"{metadata.get('r2', 0):.3f}")
c2.metric("MAE", f"{metadata.get('mae', 0):,.0f}")
mse_value = float(metadata.get("mse", 0))
rmse_value = math.sqrt(mse_value)
c3.metric("RMSE", f"{rmse_value:,.0f}")

st.info("⚠️ This model is trained exclusively on Indian domestic flight data. Predictions for international routes may not be valid.")

# ========================
# SHAP & LIME SECTION
# ========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## 🧠 Model Explainability (SHAP Insights)")

@st.cache_data(show_spinner=False)
def load_eval_data():
    X_path = MODEL_DIR / "eval_X_trans.parquet"
    y_path = MODEL_DIR / "eval_y.parquet"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("Evaluation parquet files not found. Make sure eval_X_trans.parquet and eval_y.parquet exist inside models/.")
    X = pd.read_parquet(X_path, engine = "auto")
    y = pd.read_parquet(y_path, engine = "auto")
    return X, y

try:
    X_eval, _ = load_eval_data()
    X_sample = X_eval.sample(min(300, len(X_eval)), random_state=42).copy()

    def clean_feature_name(name: str) -> str:
        name = name.replace("cat_", "")
        name = name.replace("num_", "")
        name = name.replace("_class", "Class: ")
        name = name.replace("airline", "Airline: ")
        name = name.replace("source_city", "From: ")
        name = name.replace("destination_city", "To: ")
        name = name.replace("departure_time", "Dep Time: ")
        name = name.replace("arrival_time", "Arr Time: ")
        name = name.replace("days_left", "Days Left")
        name = name.replace("stops", "Stops")
        name = name.replace("duration", "Duration (h)")
        name = name.replace("_", "")
        return name

    X_sample.columns = [clean_feature_name(c) for c in X_sample.columns]

    shap_values = compute_shap_values(model, X_sample)

    tab_imp, tab_sum = st.tabs(["📊 Feature Importance", "🎨 SHAP Summary Plot"])

    with tab_imp:
        st.markdown("#### 🔹 Average Feature Importance")
        fig1, _ = plt.subplots(figsize=(12, 7))
        shap.summary_plot(shap_values.values, X_sample, plot_type="bar", show=False, color_bar=True)
        plt.tight_layout()
        plt.xlabel("")
        st.markdown(
            """
            <div style="overflow-x:auto; padding:10px; background:rgba(255,255,255,0.03); border-radius:10px;">
            """,
            unsafe_allow_html=True,
        )
        st.pyplot(fig1, use_container_width=False, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_sum:
        st.markdown("#### 🔹 SHAP Value Distribution")
        fig2, _ = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

except Exception as e:
    st.warning(f"SHAP analysis unavailable: {e}")

st.markdown("<hr>", unsafe_allow_html=True)

# ========================
# LIME SECTION
# ========================
st.markdown("## 🧩 LIME Explanation (Single Prediction Insight)")
try:
    X_eval, y_eval = load_eval_data()
    feature_names = list(X_eval.columns)
    X_array = np.array(X_eval)

    def clean_feature_name2(name: str) -> str:
        name = name.replace("cat_", "")
        name = name.replace("num_", "")
        name = name.replace("_class", "Class: ")
        name = name.replace("airline", "Airline: ")
        name = name.replace("source_city", "From: ")
        name = name.replace("destination_city", "To: ")
        name = name.replace("departure_time", "Dep Time: ")
        name = name.replace("arrival_time", "Arr Time: ")
        name = name.replace("days_left", "Days Left")
        name = name.replace("stops", "Stops")
        name = name.replace("duration", "Duration (h)")
        name = name.replace("_", "")
        return name

    clean_feature_names = [clean_feature_name2(c) for c in feature_names]

    sample_index = st.slider("🔍 Choose a flight sample for explanation", 0, len(X_eval) - 1, 10)
    X_instance = X_array[sample_index]

    lime_exp = explain_with_lime(model, X_eval, X_instance, clean_feature_names)

    st.markdown(
        """
        <div style=" background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 15px; padding: 20px; margin-top: 15px; box-shadow: 0 0 25px rgba(56, 189, 248, 0.2); ">
            <h4 style="color:#93c5fd; text-align:center;">🎨 Understanding the LIME Explanation</h4>
            <ul style="font-size:1.05rem; color:black; list-style:none;">
                <li>🟥 <b>Red bars</b> → Features that <b>increase</b> the predicted flight price.</li>
                <li>🟩 <b>Green bars</b> → Features that <b>decrease</b> the predicted flight price.</li>
                <li>💡 Each bar shows how much that feature pushed the model’s decision <b>up or down</b> for this specific flight.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"#### 🔹 Local Explanation ==> Flight #{sample_index}")
    fig_lime = lime_exp.as_pyplot_figure()
    plt.tight_layout()
    st.pyplot(fig_lime, clear_figure=True)

except Exception as e:
    st.warning(f"LIME explanation unavailable: {e}")

st.markdown("<hr>", unsafe_allow_html=True)

# ========================
# REAL vs PREDICTED GRAPH
# ========================
try:
    X_eval, y_eval = load_eval_data()
    y_pred_eval = model.predict(X_eval)
    df_eval = pd.DataFrame({"Actual": y_eval["price"].values, "Predicted": y_pred_eval})

    st.markdown("### 📈 Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_eval["Actual"], df_eval["Predicted"], alpha=0.4)
    mn, mx = df_eval["Actual"].min(), df_eval["Actual"].max()
    ax.plot([mn, mx], [mn, mx], linewidth=2)
    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")
    ax.set_title("Model Fit Visualization")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Visualization unavailable: {e}")

# ========================
# PROJECT OVERVIEW CARDS
# ========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>💡 Project Overview & Tools Used</h2>", unsafe_allow_html=True)

st.markdown("<div class='cards-grid'>", unsafe_allow_html=True)
colA, colB, colC = st.columns(3)
with colA:
    st.markdown(
        """
        <div class="card">
            <h3>📊 Data Cleaning</h3>
            <p>Used <b>Pandas</b> and <b>NumPy</b> for data preprocessing — handling missing values, encoding categorical variables, and normalizing flight attributes to ensure clean input for the model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colB:
    st.markdown(
        """
        <div class="card">
            <h3>🧠 Model Training</h3>
            <p>Trained a <b>Decision Tree Regressor</b> using <b>Scikit-learn</b> with extensive hyperparameter tuning via <b>RandomizedSearchCV</b>. Achieved reliable accuracy and balanced bias-variance performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colC:
    st.markdown(
        """
        <div class="card">
            <h3>🚀 Deployment</h3>
            <p>Deployed interactively using <b>Streamlit</b>. Enhanced with custom CSS animations, gradient styling, and responsive design for a modern, professional ML dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

colD, colE, colF = st.columns(3)
with colD:
    st.markdown(
        """
        <div class="card">
            <h3>🎨 SHAP Explainability</h3>
            <p>Implemented <b>SHAP (SHapley Additive exPlanations)</b> to visualize global feature importance — showing how each variable impacts model predictions across the entire dataset.</p>
            <ul style="color:darkgray;">
                <li>🟦 Blue bars → Stronger impact</li>
                <li>📈 Global view of model behavior</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colE:
    st.markdown(
        """
        <div class="card">
            <h3>🧩 LIME Interpretation</h3>
            <p>Integrated <b>LIME (Local Interpretable Model-Agnostic Explanations)</b> to explain individual predictions. Helps identify why a specific flight's price was high or low.</p>
            <ul style="color:darkgray;">
                <li>🟥 Red → Increased price</li>
                <li>🟩 Green → Decreased price</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colF:
    st.markdown(
        """
        <div class="card">
            <h3>📈 Visualization</h3>
            <p>Created high-quality plots using <b>Matplotlib</b> to show <b>Actual vs Predicted</b> values and performance trends. Added dynamic tabs for SHAP and LIME visualizations.</p>
            <p>Visuals are interactive and scrollable for a clear, immersive experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ========================
# FOOTER
# ========================
st.markdown("<hr>", unsafe_allow_html=True)

footer_html = """
<div style='text-align: center; padding: 20px 0;'>
    <p style='font-size: 22px; font-weight: bold; margin-bottom: 8px;'>
        🚀 Made by <span style='color: #4F8BF9;'>Batuhan Başoda</span>
    </p>
    <p style='font-size: 16px; margin-bottom: 12px;'>
        Built using the <b>Kaggle Flight Price Dataset</b> — developed into an <b>advanced end-to-end Machine Learning project</b>.<br>
    </p>
    <h2>ADVANCE MACHINE LEARNING PROJECT</h2>
    <p style='font-size: 15px;'>
        🔗 Connect with me:<br>
        <a href='https://www.linkedin.com/in/batuhan-ba%C5%9Foda-b78799377/' target='_blank' style='text-decoration: none; color: #0077B5; font-weight: 600;'>LinkedIn</a> ·
        <a href='https://www.kaggle.com/batuhanbasoda' target='_blank' style='text-decoration: none; color: #20BEFF; font-weight: 600;'>Kaggle</a> ·
        <a href='https://github.com/Batuhan-METU' target='_blank' style='text-decoration: none; color: #333; font-weight: 600;'>GitHub</a>
    </p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
