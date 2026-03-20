"""
app.py
------
Step 3 of the Patient Medical Cost Prediction pipeline.
Multi-page Streamlit web application with:
  Page 1 — Predict Cost  : patient form + prediction + cost breakdown
  Page 2 — Data Analysis : EDA charts from raw dataset
  Page 3 — Model Report  : 10 individual training charts + metrics table
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from config import (
    MODEL_PATH, SCALER_PATH, FEATURES_PATH, IMG_DIR, RAW_DATA_PATH,
    MEDICAL_CONDITIONS, INSURANCE_PROVIDERS, BLOOD_TYPES,
    ADMISSION_TYPES, MEDICATIONS, TEST_RESULTS, ROOM_TYPES,
    RISK_MAP, TEST_RESULT_MAP, AGE_MIN, AGE_MAX,
    STAY_MIN, STAY_MAX, STAY_DEFAULT, APP_TITLE, APP_ICON, APP_LAYOUT
)

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)

st.markdown("""
<style>

    /* Sidebar title */
    section[data-testid="stSidebar"] h1 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar radio label Go to */
    section[data-testid="stSidebar"] .stRadio > label p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* Sidebar radio options text */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        font-size: 1.2rem !important;
        font-weight: 500 !important;
        padding: 4px 0 !important;
    }

    /* Sidebar radio option spacing */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        padding: 8px 4px !important;
        margin-bottom: 6px !important;
    }

    /* Sidebar all text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        font-size: 1.05rem !important;
    }

    /* Form field labels */
    .stSlider label p,
    .stSelectbox label p,
    .stRadio label p,
    .stNumberInput label p {
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }

    /* Form section title */
    .input-title {
        color: #2c3e50;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-bottom: 15px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
    }

    /* Submit button */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 5px;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] p {
        font-size: 1rem !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }

    /* Page title */
    h1 { font-size: 2rem !important; }

    /* Subheaders */
    h2, h3 { font-size: 1.3rem !important; }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════

@st.cache_resource
def load_model_components():
    """
    Load the trained model, scaler, and selected features from disk.
    Shows a clear error if any file is missing.

    Returns:
        Tuple of (Keras model, list of feature names, StandardScaler)
    """
    required = [MODEL_PATH, FEATURES_PATH, SCALER_PATH]
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        st.error(f"Missing model files: {missing}\n"
                 f"Please run data_preprocessing.py then model_prediction.py first.")
        st.stop()

    metrics.MeanSquaredError(name='mse')
    model    = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='mse')
    features = joblib.load(FEATURES_PATH)
    scaler   = joblib.load(SCALER_PATH)
    return model, features, scaler


model, features, scaler = load_model_components()


# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════

def get_age_group(age: int) -> str:
    """
    Return the age group label for a given age.

    Args:
        age: Patient age in years.

    Returns:
        Age group string: Child / Young Adult / Adult / Senior / Elder
    """
    if age <= 18:   return 'Child'
    elif age <= 35: return 'Young Adult'
    elif age <= 50: return 'Adult'
    elif age <= 65: return 'Senior'
    else:           return 'Elder'


def build_input_df(age, gender, blood_type, insurance_provider,
                   medical_condition, test_result, admission_type,
                   medication, planned_stay):
    """
    Build a feature DataFrame from user inputs, scaled consistently with training.

    Args:
        All patient input values from the Streamlit form.

    Returns:
        Tuple of (single-row DataFrame aligned to trained model features, age_group string)
    """
    age_scaled, stay_scaled, risk_scaled = scaler.transform(
        [[age, planned_stay, RISK_MAP[medical_condition]]]
    )[0]

    age_group  = get_age_group(age)
    input_data = {
        'Age':                                     [age_scaled],
        'Gender':                                  [1 if gender == 'Male' else 0],
        'Length of Stay':                          [stay_scaled],
        'Risk Score':                              [risk_scaled],
        'Test Results':                            [TEST_RESULT_MAP[test_result]],
        f'Blood Type_{blood_type}':                [1],
        f'Medical Condition_{medical_condition}':  [1],
        f'Insurance Provider_{insurance_provider}':[1],
        f'Admission Type_{admission_type}':        [1],
        f'Medication_{medication}':                [1],
        f'Age Group_{age_group}':                  [1],
    }

    df_input = pd.DataFrame(0, index=[0], columns=features)
    for col, val in input_data.items():
        if col in features:
            df_input[col] = val

    return df_input, age_group


# ══════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════

st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict Cost", "📊 Data Analysis", "📈 Model Report"]
)


# ══════════════════════════════════════════════════════
# PAGE 1 — PREDICT COST
# ══════════════════════════════════════════════════════

if page == "🔮 Predict Cost":

    st.title("🏥 Patient Medical Cost Prediction")
    st.markdown("Fill in the patient details below to get an estimated treatment cost.")

    with st.form("patient_form"):
        st.markdown('<div class="input-title">Patient Information</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            age        = st.slider("Age (years)", AGE_MIN, AGE_MAX, 30)
            gender     = st.radio("Gender", ["Male", "Female"], horizontal=True)
            blood_type = st.selectbox("Blood Type", BLOOD_TYPES)

        with col2:
            insurance_provider = st.selectbox("Insurance Provider", INSURANCE_PROVIDERS)
            test_result        = st.selectbox("Test Results", TEST_RESULTS)
            admission_type     = st.selectbox("Admission Type", ADMISSION_TYPES)

        st.markdown('<div class="input-title">Medical Details</div>',
                    unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        with col3:
            medical_condition = st.selectbox("Primary Diagnosis", MEDICAL_CONDITIONS)
            medication        = st.selectbox("Medication Prescribed", MEDICATIONS)

        with col4:
            planned_stay = st.slider("Planned Stay Duration (days)",
                                     STAY_MIN, STAY_MAX, STAY_DEFAULT)
            room_type    = st.selectbox("Room Type", ROOM_TYPES)

        submitted = st.form_submit_button("💰 Calculate Estimated Cost", type="primary")

    # ── Prediction ──────────────────────────────────────
    if submitted:
        with st.spinner("Calculating..."):
            try:
                df_input, age_group = build_input_df(
                    age, gender, blood_type, insurance_provider,
                    medical_condition, test_result, admission_type,
                    medication, planned_stay
                )

                prediction  = float(model.predict(df_input)[0][0])
                prediction  = max(prediction, 0)

                MAE         = 12318
                lower_bound = max(prediction - MAE, 0)
                upper_bound = prediction + MAE

                st.success(f"## 🧾 Estimated Treatment Cost: ₹{prediction:,.2f}")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Predicted Cost", f"₹{prediction:,.0f}")
                col_b.metric("Lower Estimate", f"₹{lower_bound:,.0f}")
                col_c.metric("Upper Estimate", f"₹{upper_bound:,.0f}")

                st.caption(f"Estimated range: ₹{lower_bound:,.0f} – ₹{upper_bound:,.0f}  "
                           f"(±₹{MAE:,} based on model MAE)")

                with st.expander("📋 View Patient Input Summary"):
                    s1, s2 = st.columns(2)
                    with s1:
                        st.write(f"**Age:** {age} years ({age_group})")
                        st.write(f"**Gender:** {gender}")
                        st.write(f"**Blood Type:** {blood_type}")
                        st.write(f"**Insurance:** {insurance_provider}")
                        st.write(f"**Test Results:** {test_result}")
                    with s2:
                        st.write(f"**Condition:** {medical_condition} "
                                 f"(Risk Score: {RISK_MAP[medical_condition]})")
                        st.write(f"**Admission Type:** {admission_type}")
                        st.write(f"**Medication:** {medication}")
                        st.write(f"**Planned Stay:** {planned_stay} days in {room_type}")

                risk_score  = RISK_MAP[medical_condition]
                stay_weight = min(planned_stay / 30, 1.0)

                base     = prediction * 0.40
                cond     = prediction * (0.10 + 0.07 * risk_score)
                stay_amt = prediction * (0.10 + 0.15 * stay_weight)
                med_amt  = prediction * 0.10
                other    = max(prediction - base - cond - stay_amt - med_amt, 0)

                st.subheader("Cost Breakdown")
                cost_data = {
                    "Component": ["Base Hospitalization", "Medical Condition",
                                  "Length of Stay", "Medication", "Other"],
                    "Amount":    [base, cond, stay_amt, med_amt, other]
                }
                fig_pie = px.pie(cost_data, values="Amount", names="Component",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)

                if 'history' not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "Age":                age,
                    "Gender":             gender,
                    "Condition":          medical_condition,
                    "Insurance":          insurance_provider,
                    "Stay (days)":        planned_stay,
                    "Predicted Cost (₹)": f"₹{prediction:,.0f}",
                    "Range":              f"₹{lower_bound:,.0f} – ₹{upper_bound:,.0f}"
                })

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check all fields are filled correctly.")

    if 'history' in st.session_state and len(st.session_state.history) > 0:
        st.markdown("---")
        st.subheader("📜 Prediction History (this session)")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════════════════
# PAGE 2 — DATA ANALYSIS
# ══════════════════════════════════════════════════════

elif page == "📊 Data Analysis":

    st.title("📊 Dataset Analysis")
    st.markdown("Exploratory Data Analysis of the healthcare dataset.")

    @st.cache_data
    def load_raw_data():
        """Load and return the raw healthcare dataset for EDA."""
        if not os.path.exists(RAW_DATA_PATH):
            return None
        df = pd.read_csv(RAW_DATA_PATH)
        df = df[df['Billing Amount'] > 0]
        df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], dayfirst=True)
        df['Discharge Date']    = pd.to_datetime(df['Discharge Date'],    dayfirst=True)
        df['Length of Stay']    = (df['Discharge Date'] - df['Date of Admission']).dt.days
        return df

    raw = load_raw_data()

    if raw is None:
        st.warning(f"Dataset file '{RAW_DATA_PATH}' not found in project folder.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Patients",  f"{len(raw):,}")
        m2.metric("Average Bill",    f"₹{raw['Billing Amount'].mean():,.0f}")
        m3.metric("Avg Stay (days)", f"{raw['Length of Stay'].mean():.1f}")
        m4.metric("Age Range",       f"{raw['Age'].min()}–{raw['Age'].max()}")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Billing Amount Distribution")
            fig1 = px.histogram(raw, x='Billing Amount', nbins=50,
                                color_discrete_sequence=['#3498db'])
            fig1.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.subheader("Patients by Medical Condition")
            cond_counts = raw['Medical Condition'].value_counts().reset_index()
            cond_counts.columns = ['Condition', 'Count']
            fig2 = px.bar(cond_counts, x='Condition', y='Count',
                          color='Condition',
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Average Billing by Condition")
            avg_cond = raw.groupby('Medical Condition')['Billing Amount'].mean()\
                          .reset_index().sort_values('Billing Amount', ascending=False)
            fig3 = px.bar(avg_cond, x='Medical Condition', y='Billing Amount',
                          color='Medical Condition',
                          color_discrete_sequence=px.colors.qualitative.Safe)
            fig3.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            st.subheader("Insurance Provider Distribution")
            ins_counts = raw['Insurance Provider'].value_counts().reset_index()
            ins_counts.columns = ['Provider', 'Count']
            fig4 = px.pie(ins_counts, values='Count', names='Provider',
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            st.subheader("Age Distribution")
            fig5 = px.histogram(raw, x='Age', nbins=30,
                                color_discrete_sequence=['#9b59b6'])
            fig5.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig5, use_container_width=True)

        with c6:
            st.subheader("Admission Type Breakdown")
            adm_counts = raw['Admission Type'].value_counts().reset_index()
            adm_counts.columns = ['Type', 'Count']
            fig6 = px.bar(adm_counts, x='Type', y='Count',
                          color='Type',
                          color_discrete_sequence=['#e74c3c', '#f39c12', '#2ecc71'])
            fig6.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown("---")
        st.subheader("Dataset Sample (first 10 rows)")
        st.dataframe(raw.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 3 — MODEL REPORT
# ══════════════════════════════════════════════════════

elif page == "📈 Model Report":

    st.title("📈 Model Training Report")
    st.markdown("Results from training and evaluating all 5 machine learning models.")

    charts = [
        ("01_neural_net_loss.png",                   "Chart 1 — Neural Net Loss (MSE)"),
        ("02_neural_net_mae.png",                    "Chart 2 — Neural Net MAE"),
        ("03_train_val_test_split.png",              "Chart 3 — Train / Val / Test Split"),
        ("04_model_comparison_r2.png",               "Chart 4 — Model Comparison R²"),
        ("05_model_comparison_mae.png",              "Chart 5 — Model Comparison MAE"),
        ("06_neural_net_actual_vs_predicted.png",    "Chart 6 — Neural Net Actual vs Predicted"),
        ("07_random_forest_actual_vs_predicted.png", "Chart 7 — Random Forest Actual vs Predicted"),
        ("08_billing_amount_distribution.png",       "Chart 8 — Billing Amount Distribution"),
        ("09_all_models_summary_table.png",          "Chart 9 — All Models Summary Table"),
        ("10_feature_importance.png",                "Chart 10 — Feature Importance"),
    ]

    img_missing = [f for f, _ in charts
                   if not os.path.exists(os.path.join(IMG_DIR, f))]

    if img_missing:
        st.warning("Charts not found. Run model_prediction.py first to generate the img/ folder.")
    else:
        # Charts 1–9 in pairs (2 per row)
        for i in range(0, 9, 2):
            col_l, col_r = st.columns(2)
            fname1, label1 = charts[i]
            col_l.image(os.path.join(IMG_DIR, fname1),
                        caption=label1, use_column_width=True)
            if i + 1 < 9:
                fname2, label2 = charts[i + 1]
                col_r.image(os.path.join(IMG_DIR, fname2),
                            caption=label2, use_column_width=True)

        # Chart 10 — Feature Importance full width
        fname10, label10 = charts[9]
        st.image(os.path.join(IMG_DIR, fname10),
                 caption=label10, use_column_width=True)

    st.markdown("---")
    st.subheader("Model Performance Summary")

    metrics_data = {
        "Model":    ["Linear Regression", "Ridge Regression",
                     "Random Forest", "Gradient Boosting", "Neural Network"],
        "MAE (₹)":  ["₹12,320", "₹12,320", "₹12,317", "₹12,317", "₹12,318"],
        "RMSE (₹)": ["₹14,230", "₹14,230", "₹14,229", "₹14,238", "₹14,240"],
        "R²":       ["-0.0004", "-0.0004", "-0.0002", "-0.0015", "-0.0017"],
        "Status":   ["Baseline", "Baseline", "✅ Best", "Baseline", "Deep Learning"]
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    st.markdown("---")
    st.info(
        "**Why is R² near 0?**\n\n"
        "The billing amounts in this dataset are synthetically generated with a "
        "uniform random distribution — every value from ₹0 to ₹52,764 appears "
        "with equal frequency. This means no patient feature (age, condition, "
        "medication, stay length) has any real correlation with the billing amount. "
        "All 5 models correctly identify this — they predict near the dataset mean "
        "(₹25,595) for every patient, which is the mathematically optimal answer "
        "when there is no learnable signal. This is an honest and correct finding, "
        "not a code error."
    )