# Telecom Customer Churn Predictor — Streamlit App
# Run with: streamlit run app.py

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH       = "model/best_model_pipeline.pkl"
GENDER_OPTIONS   = ["Male", "Female"]
INTERNET_OPTIONS = ["DSL", "Fiber optic", "No"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTIONS  = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]

# ── Load Model ──
@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    return model

# ── Page Setup ──
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="centered",
)
st.title("📡 Telecom Customer Churn Predictor")
st.markdown(
    """
    <h3 style='font-size:18px;'>
    Fill in the customer's profile below and click <b>Predict Churn</b>
    to get an estimated churn probability.
    </h3>
    """,
    unsafe_allow_html=True,
)
st.divider()

try:
    load_model()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# ── Input Form ──
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account & Services")
    gender     = st.selectbox("Gender",           GENDER_OPTIONS,   index=None, placeholder="Choose...")
    internet   = st.selectbox("Internet Service", INTERNET_OPTIONS, index=None, placeholder="Choose...")
    contract   = st.selectbox("Contract",         CONTRACT_OPTIONS, index=None, placeholder="Choose...")
    payment    = st.selectbox("Payment Method",   PAYMENT_OPTIONS,  index=None, placeholder="Choose...")
    monthly_ch = st.number_input("Monthly Charges ($)", value=None, min_value=0.0, max_value=200.0,   step=0.5)
    tenure     = st.number_input("Tenure (months)",     value=None, min_value=0,   max_value=72,      step=1)
    total_ch   = st.number_input("Total Charges ($)",   value=None, min_value=0.0, max_value=10000.0, step=1.0)

with col2:
    st.subheader("Demographics & Billing")
    st.markdown("**Demographics**")
    senior     = st.checkbox("Senior citizen", value=False)
    partner    = st.checkbox("Has partner",    value=False)
    dependents = st.checkbox("Has dependents", value=False)
    st.markdown("**Billing**")
    paperless  = st.checkbox("Paperless billing", value=False)
    st.markdown("**Phone**")
    phone_service  = st.checkbox("Phone service",  value=False)
    multiple_lines = st.checkbox("Multiple lines", value=False)

with col3:
    st.subheader("Add-ons")
    online_sec    = st.checkbox("Online security",   value=False)
    online_back   = st.checkbox("Online backup",     value=False)
    dev_prot      = st.checkbox("Device protection", value=False)
    tech_sup      = st.checkbox("Tech support",      value=False)
    stream_tv     = st.checkbox("Streaming TV",      value=False)
    stream_movies = st.checkbox("Streaming movies",  value=False)

st.divider()

# ── Helpers: map checkboxes → model-expected strings ──
def yn(flag):
    return "Yes" if flag else "No"

def phone_val(has_phone, has_multiple):
    if not has_phone:
        return "No phone service"
    return "Yes" if has_multiple else "No"

def addon_val(has_internet, flag):
    if has_internet == "No" or has_internet is None:
        return "No internet service"
    return "Yes" if flag else "No"

# ── Prediction ──
if st.button("Predict Churn", type="primary", use_container_width=True):
    errors = []

    if gender is None:
        errors.append("Please select a value for: **Gender**")
    if internet is None:
        errors.append("Please select a value for: **Internet Service**")
    if contract is None:
        errors.append("Please select a value for: **Contract**")
    if payment is None:
        errors.append("Please select a value for: **Payment Method**")
    if tenure is None:
        errors.append("Please fill in: **Tenure**")
    if monthly_ch is None:
        errors.append("Please fill in: **Monthly Charges**")
    if total_ch is None:
        errors.append("Please fill in: **Total Charges**")

    if errors:
        for msg in errors:
            st.warning(msg)
    else:
        try:
            num_addons = sum([online_sec, online_back, dev_prot, tech_sup, stream_tv, stream_movies])
            is_new     = int(tenure <= 3)

            input_df = pd.DataFrame([{
                "gender":           gender,
                "SeniorCitizen":    int(senior),
                "Partner":          yn(partner),
                "Dependents":       yn(dependents),
                "tenure":           tenure,
                "PhoneService":     yn(phone_service),
                "MultipleLines":    phone_val(phone_service, multiple_lines),
                "InternetService":  internet,
                "OnlineSecurity":   addon_val(internet, online_sec),
                "OnlineBackup":     addon_val(internet, online_back),
                "DeviceProtection": addon_val(internet, dev_prot),
                "TechSupport":      addon_val(internet, tech_sup),
                "StreamingTV":      addon_val(internet, stream_tv),
                "StreamingMovies":  addon_val(internet, stream_movies),
                "Contract":         contract,
                "PaperlessBilling": yn(paperless),
                "PaymentMethod":    payment,
                "MonthlyCharges":   monthly_ch,
                "TotalCharges":     total_ch,
                "NumAddons":        num_addons,
                "IsNewCustomer":    is_new,
            }])

            model = load_model()
            if hasattr(model, "feature_names_in_"):
                expected_cols = list(model.feature_names_in_)
                provided_set = set(input_df.columns)
                input_df = input_df[expected_cols]

            prob = model.predict_proba(input_df)[0, 1]
            THRESHOLD = 0.40
            pred = int(prob >= THRESHOLD)
            st.divider()
            r1, r2 = st.columns([1, 2])

            with r1:
                if pred == 1:
                    st.error("⚠️ CHURN RISK DETECTED")
                else:
                    st.success("Customer Likely to Stay")
                st.metric("Churn Probability", f"{prob * 100:.1f}%")

            with r2:
                fig, ax = plt.subplots(figsize=(5, 2))
                colour = "#E84C4C" if prob >= THRESHOLD else "#4C9BE8"
                ax.barh(["Churn Risk"], [prob],       color=colour,    height=0.4)
                ax.barh(["Churn Risk"], [1 - prob], left=[prob], color="#E0E0E0", height=0.4)
                ax.set_xlim(0, 1)
                ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.2)
                ax.set_title("Churn Probability", fontsize=11)
                ax.set_xlabel("Probability")
                ax.spines[["top", "right", "left"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")