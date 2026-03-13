import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #162032 60%, #0d2137 100%);
    color: #e8f0fe;
}

/* Header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #ffffff;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    text-align: center;
    color: #7ca3c8;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
}

/* Card */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4da3e0;
    margin-bottom: 1rem;
}

/* Result boxes */
.result-safe {
    background: linear-gradient(135deg, #0a3d2b, #0e5c3e);
    border: 1px solid #1a9e68;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-risk {
    background: linear-gradient(135deg, #3d1010, #5c1a1a);
    border: 1px solid #c0392b;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #ffffff;
    margin: 0.5rem 0;
}
.result-prob {
    font-size: 1rem;
    color: rgba(255,255,255,0.65);
    font-weight: 300;
}

/* Probability bar */
.prob-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 10px;
    margin: 0.8rem 0;
    overflow: hidden;
}
.prob-bar-fill-safe {
    background: linear-gradient(90deg, #1abc9c, #27ae60);
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}
.prob-bar-fill-risk {
    background: linear-gradient(90deg, #e74c3c, #c0392b);
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* Input labels */
label {
    color: #a0c4e0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* Number inputs */
input[type=number] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* Slider thumb */
.stSlider > div > div > div > div {
    background-color: #4da3e0 !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08); }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1a6bbd, #2980b9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2.5rem;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    width: 100%;
    letter-spacing: 0.04em;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(41, 128, 185, 0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2276cc, #3498db);
    box-shadow: 0 6px 24px rgba(41, 128, 185, 0.5);
    transform: translateY(-1px);
}

/* Disclaimer */
.disclaimer {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.3);
    text-align: center;
    margin-top: 2rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "diabetes_prediction_model.pkl"
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found. Make sure `diabetes_prediction_model.pkl` is in the same folder as this app.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ── Preprocessing constants (fitted on Pima Indians dataset) ──────────────────
SCALER_PARAMS = {
    "Pregnancies":             {"mean": 3.845, "std": 3.369},
    "Glucose":                 {"mean": 121.7, "std": 30.4},
    "BMI":                     {"mean": 32.46, "std": 6.87},
    "DiabetesPedigreeFunction": {"mean": 0.472, "std": 0.331},
}
# Insulin thresholds (33rd & 66th percentile of non-zero insulin in dataset)
INSULIN_LOW_THRESH  = 105.0   # 33rd percentile
INSULIN_HIGH_THRESH = 166.0   # 66th percentile

def categorize_insulin(insulin_value):
    """Replicate the notebook's Insulin_Category logic."""
    if insulin_value == 0:
        return "Very_low"
    elif insulin_value <= INSULIN_LOW_THRESH:
        return "Low"
    elif insulin_value <= INSULIN_HIGH_THRESH:
        return "Medium"
    else:
        return "High"

def scale(value, feature):
    p = SCALER_PARAMS[feature]
    return (value - p["mean"]) / p["std"]

def preprocess(preg, glucose, bp, skin, bmi, dpf, age, insulin):
    insulin_cat = categorize_insulin(insulin)
    insulin_cat_low = 1.0 if insulin_cat == "Low" else 0.0

    features = np.array([[
        scale(preg,    "Pregnancies"),
        scale(glucose, "Glucose"),
        float(bp),
        float(skin),
        scale(bmi,     "BMI"),
        scale(dpf,     "DiabetesPedigreeFunction"),
        float(age),
        insulin_cat_low,
    ]])
    return features

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter patient metrics below to assess early-stage diabetes risk</div>', unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1,
                                   help="Number of times pregnant")
    glucose     = st.number_input("Glucose (mg/dL)", min_value=50, max_value=250, value=100,
                                   help="Plasma glucose concentration (2-hr oral glucose tolerance test)")
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=30, max_value=130, value=72,
                                      help="Diastolic blood pressure")
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=5, max_value=100, value=23,
                                      help="Triceps skin fold thickness")

with col2:
    bmi         = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                                   help="Body mass index")
    dpf         = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.5,
                                   value=0.35, step=0.01,
                                   help="A function scoring likelihood of diabetes based on family history")
    age         = st.number_input("Age (years)", min_value=18, max_value=100, value=30,
                                   help="Patient age in years")
    insulin     = st.number_input("Insulin (µU/mL)", min_value=0, max_value=850, value=80,
                                   help="2-Hour serum insulin level (0 = not measured)")

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Run Prediction →"):
    features = preprocess(pregnancies, glucose, blood_pressure, skin_thickness, bmi, dpf, age, insulin)
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    diabetic_prob    = probability[1] * 100
    nondiabetic_prob = probability[0] * 100

    st.markdown("---")
    st.markdown('<div class="card-title" style="text-align:center;font-size:0.7rem;letter-spacing:0.12em;color:#4da3e0;">PREDICTION RESULT</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-risk">
            <div style="font-size:2.5rem;">⚠️</div>
            <div class="result-label">Diabetic Risk Detected</div>
            <div class="result-prob">The model predicts a <strong>{diabetic_prob:.1f}%</strong> probability of diabetes.</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill-risk" style="width:{diabetic_prob}%"></div>
            </div>
            <div style="font-size:0.8rem;color:rgba(255,255,255,0.45);margin-top:0.5rem;">
                Consult a healthcare professional for clinical evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div style="font-size:2.5rem;">✅</div>
            <div class="result-label">Low Diabetes Risk</div>
            <div class="result-prob">The model predicts a <strong>{nondiabetic_prob:.1f}%</strong> probability of being non-diabetic.</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill-safe" style="width:{nondiabetic_prob}%"></div>
            </div>
            <div style="font-size:0.8rem;color:rgba(255,255,255,0.45);margin-top:0.5rem;">
                Maintain regular health check-ups and a healthy lifestyle.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Confidence breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Confidence Breakdown</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Non-Diabetic", f"{nondiabetic_prob:.1f}%")
    with col_b:
        st.metric("Diabetic", f"{diabetic_prob:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚕️ This tool is for educational and research purposes only.<br>
    It is not a substitute for professional medical advice, diagnosis, or treatment.<br>
    Always consult a qualified healthcare provider with any questions about a medical condition.
</div>
""", unsafe_allow_html=True)
