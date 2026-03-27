"""
app.py
------
Step 3 of the Patient Medical Cost Prediction pipeline.
Multi-page Streamlit web application.

Changes:
  - Branding: City General Hospital → Medical Cost AI
  - Billing header (was Invoice)
  - Patient No (was Invoice No)
  - Planned stay fully automatic — driven by condition (CONDITION_STAY_DAYS)
  - Room type fully automatic — driven by condition (CONDITION_ROOM_TYPE)
  - Live cost preview updates as fields change (before submit)
"""

import os
import csv
import datetime
import streamlit as st
import streamlit.components.v1 as components
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
    STAY_MIN, STAY_MAX, STAY_DEFAULT, APP_TITLE, APP_ICON, APP_LAYOUT,
    CONDITION_MEDICATIONS, CONDITION_STAY_DAYS, CONDITION_ROOM_TYPE
)

# ══════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════

PATIENTS_CSV = 'patients.csv'

PATIENT_COLUMNS = [
    'Patient ID', 'Registration Date', 'Registration Time',
    'Patient Name', 'Patient Age', 'Patient Gender',
    'Patient Phone', 'Patient Address',
    'Attender Name', 'Attender Relationship', 'Attender Phone', 'Attender Address',
    'Blood Type', 'Insurance Provider', 'Medical Condition',
    'Test Results', 'Admission Type', 'Medication', 'Medication (Dataset Encoded)',
    'Planned Stay (days)', 'Room Type',
    'Predicted Cost (INR)', 'Lower Estimate (INR)', 'Upper Estimate (INR)',
    'Base Hospitalization', 'Medical Condition Cost', 'Length of Stay Cost',
    'Medication Cost', 'Other Charges'
]

DRUG_CLASS_COLORS = {
    'DMARD':                            ('#dbeafe', '#1e40af'),
    'NSAID':                            ('#fef9c3', '#854d0e'),
    'COX-2 inhibitor':                  ('#fef9c3', '#854d0e'),
    'Corticosteroid':                   ('#fce7f3', '#9d174d'),
    'Oral corticosteroid':              ('#fce7f3', '#9d174d'),
    'Biologic':                         ('#f0fdf4', '#166534'),
    'SABA inhaler':                     ('#ede9fe', '#4c1d95'),
    'LABA inhaler':                     ('#ede9fe', '#4c1d95'),
    'ICS inhaler':                      ('#ede9fe', '#4c1d95'),
    'Leukotriene antagonist':           ('#e0f2fe', '#0c4a6e'),
    'Anticholinergic':                  ('#fce7f3', '#9d174d'),
    'Methylxanthine':                   ('#fef9c3', '#854d0e'),
    'Anthracycline chemo':              ('#fee2e2', '#7f1d1d'),
    'Taxane chemo':                     ('#fee2e2', '#7f1d1d'),
    'Platinum chemo':                   ('#fee2e2', '#7f1d1d'),
    'Alkylating chemo':                 ('#fee2e2', '#7f1d1d'),
    'Antimetabolite chemo':             ('#fee2e2', '#7f1d1d'),
    'Hormonal therapy':                 ('#fce7f3', '#9d174d'),
    'Aromatase inhibitor':              ('#fce7f3', '#9d174d'),
    'Targeted therapy':                 ('#f0fdf4', '#166534'),
    'Monoclonal antibody':              ('#f0fdf4', '#166534'),
    'Tyrosine kinase inhibitor':        ('#f0fdf4', '#166534'),
    'Biguanide':                        ('#e0f2fe', '#0c4a6e'),
    'Sulfonylurea':                     ('#fef9c3', '#854d0e'),
    'DPP-4 inhibitor':                  ('#dbeafe', '#1e40af'),
    'SGLT-2 inhibitor':                 ('#f0fdf4', '#166534'),
    'GLP-1 agonist':                    ('#f0fdf4', '#166534'),
    'Long-acting insulin':              ('#ede9fe', '#4c1d95'),
    'Rapid-acting insulin':             ('#ede9fe', '#4c1d95'),
    'Calcium channel blocker':          ('#fef9c3', '#854d0e'),
    'ACE inhibitor':                    ('#dbeafe', '#1e40af'),
    'ARB':                              ('#dbeafe', '#1e40af'),
    'Thiazide diuretic':                ('#e0f2fe', '#0c4a6e'),
    'Loop diuretic':                    ('#e0f2fe', '#0c4a6e'),
    'Beta-blocker':                     ('#fee2e2', '#7f1d1d'),
    'Lipase inhibitor':                 ('#fef9c3', '#854d0e'),
    'Appetite suppressant':             ('#fce7f3', '#9d174d'),
    'Anticonvulsant / weight loss':     ('#ede9fe', '#4c1d95'),
    'Combination':                      ('#f0fdf4', '#166534'),
    'Statin — for comorbid lipids':     ('#fef9c3', '#854d0e'),
    'SGLT-2 — for comorbid diabetes':   ('#f0fdf4', '#166534'),
    'If hypothyroid-related obesity':   ('#e0f2fe', '#0c4a6e'),
    'Off-label, insulin sensitiser':    ('#e0f2fe', '#0c4a6e'),
}


def parse_medication(full_name: str):
    if '(' in full_name and full_name.endswith(')'):
        drug = full_name[:full_name.rfind('(')].strip()
        cls  = full_name[full_name.rfind('(') + 1:-1].strip()
        return drug, cls
    return full_name, ''


def closest_dataset_medication(condition: str) -> str:
    mapping = {
        'Arthritis':    'Ibuprofen',
        'Asthma':       'Ibuprofen',
        'Cancer':       'Paracetamol',
        'Diabetes':     'Aspirin',
        'Hypertension': 'Aspirin',
        'Obesity':      'Paracetamol',
    }
    return mapping.get(condition, 'Paracetamol')


# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
    }
    section[data-testid="stSidebar"] h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.6rem !important; color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio > label p {
        font-size: 0.78rem !important; font-weight: 600 !important;
        color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 1.5px;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        font-size: 1.05rem !important; font-weight: 500 !important; color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        padding: 10px 8px !important; border-radius: 8px; margin-bottom: 4px !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] hr   { border-color: rgba(255,255,255,0.15) !important; }

    h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.1rem !important; color: #0f2027 !important;
        border-bottom: 3px solid #2c5364; padding-bottom: 10px; margin-bottom: 20px;
    }
    h2 { font-size: 1.3rem !important; color: #1e3a4a !important; }
    h3 { font-size: 1.05rem !important; color: #2c5364 !important; }

    .section-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-left: 4px solid #2c5364; border-radius: 8px;
        padding: 20px 24px; margin-bottom: 20px;
    }
    .section-title {
        font-family: 'DM Serif Display', serif; font-size: 1.12rem;
        color: #1e3a4a; font-weight: 700; margin-bottom: 16px;
    }
    .med-info-box {
        background: #f0f9ff; border: 1px solid #bae6fd;
        border-left: 4px solid #0284c7; border-radius: 8px;
        padding: 14px 18px; margin-top: 8px; margin-bottom: 4px;
    }
    .med-info-drug { font-size: 1.05rem; font-weight: 700; color: #0c4a6e; margin-bottom: 6px; }
    .med-info-row  { font-size: 0.88rem; color: #334155; margin-bottom: 4px; line-height: 1.5; }

    /* Auto-assigned info box */
    .auto-info-box {
        background: #f0fdf4; border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a; border-radius: 8px;
        padding: 14px 18px; margin-bottom: 16px;
        display: flex; flex-wrap: wrap; gap: 20px;
    }
    .auto-info-item { display: flex; flex-direction: column; }
    .auto-info-label { font-size: 0.70rem; color: #166534; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
    .auto-info-value { font-size: 1.0rem; font-weight: 700; color: #14532d; }

    /* Live preview box */
    .preview-box {
        background: #fffbeb; border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b; border-radius: 8px;
        padding: 16px 20px; margin-bottom: 20px;
    }
    .preview-title { font-size: 0.80rem; font-weight: 700; color: #92400e;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
    .preview-cost  { font-size: 1.6rem; font-weight: 700; color: #78350f; }
    .preview-range { font-size: 0.80rem; color: #92400e; margin-top: 3px; }
    .preview-rows  { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }
    .preview-row   { font-size: 0.82rem; color: #78350f; }

    .stButton > button {
        background: linear-gradient(135deg, #203a43, #2c5364) !important;
        color: white !important; border: none !important;
        padding: 14px 36px !important; border-radius: 8px !important;
        font-size: 1.05rem !important; font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(44,83,100,0.3);
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.82rem !important; color: #64748b !important;
        text-transform: uppercase; letter-spacing: 0.8px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important; font-weight: 700 !important; color: #0f2027 !important;
    }
    .pid-badge {
        display: inline-block;
        background: linear-gradient(135deg, #203a43, #2c5364);
        color: white; font-size: 1.45rem; font-weight: 700;
        font-family: 'DM Serif Display', serif; padding: 10px 28px;
        border-radius: 50px; letter-spacing: 3px; margin: 8px 0 20px 0;
        box-shadow: 0 4px 12px rgba(44,83,100,0.35);
    }
    .stSlider label p, .stSelectbox label p, .stRadio label p,
    .stNumberInput label p, .stTextInput label p, .stTextArea label p {
        font-size: 0.93rem !important; font-weight: 500 !important; color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════

@st.cache_resource
def load_model_components():
    required = [MODEL_PATH, FEATURES_PATH, SCALER_PATH]
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        st.error(f"Missing model files: {missing}\nRun data_preprocessing.py then model_prediction.py first.")
        st.stop()
    metrics.MeanSquaredError(name='mse')
    mdl   = load_model(MODEL_PATH, compile=False)
    mdl.compile(optimizer='adam', loss='mse')
    feats = joblib.load(FEATURES_PATH)
    scl   = joblib.load(SCALER_PATH)
    return mdl, feats, scl

model, features, scaler = load_model_components()


# ══════════════════════════════════════════════════════
# CSV HELPERS
# ══════════════════════════════════════════════════════

def init_csv():
    if not os.path.exists(PATIENTS_CSV):
        with open(PATIENTS_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=PATIENT_COLUMNS).writeheader()

def get_next_patient_id() -> str:
    init_csv()
    df = pd.read_csv(PATIENTS_CSV)
    if df.empty:
        return 'P001'
    return f'P{int(df["Patient ID"].iloc[-1][1:]) + 1:03d}'

def save_patient(record: dict):
    init_csv()
    with open(PATIENTS_CSV, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=PATIENT_COLUMNS).writerow(record)

def load_all_patients() -> pd.DataFrame:
    init_csv()
    return pd.read_csv(PATIENTS_CSV)

def get_patient_by_id(pid: str):
    df  = load_all_patients()
    row = df[df['Patient ID'].str.upper() == pid.upper()]
    return row.iloc[0] if not row.empty else None


# ══════════════════════════════════════════════════════
# PREDICTION HELPERS
# ══════════════════════════════════════════════════════

def get_age_group(age: int) -> str:
    if age <= 18:   return 'Child'
    elif age <= 35: return 'Young Adult'
    elif age <= 50: return 'Adult'
    elif age <= 65: return 'Senior'
    else:           return 'Elder'

def build_input_df(age, gender, blood_type, insurance_provider,
                   medical_condition, test_result, admission_type,
                   encoded_medication, planned_stay):
    age_sc, stay_sc, risk_sc = scaler.transform(
        [[age, planned_stay, RISK_MAP[medical_condition]]]
    )[0]
    age_group  = get_age_group(age)
    input_data = {
        'Age':                                      [age_sc],
        'Gender':                                   [1 if gender == 'Male' else 0],
        'Length of Stay':                           [stay_sc],
        'Risk Score':                               [risk_sc],
        'Test Results':                             [TEST_RESULT_MAP[test_result]],
        f'Blood Type_{blood_type}':                 [1],
        f'Medical Condition_{medical_condition}':   [1],
        f'Insurance Provider_{insurance_provider}': [1],
        f'Admission Type_{admission_type}':         [1],
        f'Medication_{encoded_medication}':         [1],
        f'Age Group_{age_group}':                   [1],
    }
    df_in = pd.DataFrame(0, index=[0], columns=features)
    for col, val in input_data.items():
        if col in features:
            df_in[col] = val
    return df_in, age_group

def compute_breakdown(prediction, medical_condition, planned_stay) -> dict:
    rs   = RISK_MAP[medical_condition]
    sw   = min(planned_stay / 30, 1.0)
    base = round(prediction * 0.40, 2)
    cond = round(prediction * (0.10 + 0.07 * rs), 2)
    stay = round(prediction * (0.10 + 0.15 * sw), 2)
    med  = round(prediction * 0.10, 2)
    oth  = round(max(prediction - base - cond - stay - med, 0), 2)
    return {'base': base, 'condition': cond, 'stay': stay, 'medication': med, 'other': oth}

def quick_predict(age, gender, blood_type, insurance_provider,
                  medical_condition, test_result, admission_type, planned_stay):
    """Quick prediction for live preview — returns (prediction, lower, upper, bd)."""
    try:
        encoded_med = closest_dataset_medication(medical_condition)
        df_in, _    = build_input_df(age, gender, blood_type, insurance_provider,
                                     medical_condition, test_result, admission_type,
                                     encoded_med, planned_stay)
        pred        = float(max(model.predict(df_in, verbose=0)[0][0], 0))
        MAE         = 12318
        low         = max(pred - MAE, 0)
        high        = pred + MAE
        bd          = compute_breakdown(pred, medical_condition, planned_stay)
        return pred, low, high, bd
    except Exception:
        return None, None, None, None


# ══════════════════════════════════════════════════════
# BILLING RENDERER  (was Invoice)
# Branding: Medical Cost AI
# Header:   BILLING  (was INVOICE)
# ID label: Patient No  (was Invoice No)
# ══════════════════════════════════════════════════════

def render_billing(patient_name, patient_id, reg_date, reg_time,
                   age, gender, blood_type, insurance_provider,
                   medical_condition, admission_type, medication_display,
                   planned_stay, room_type,
                   prediction, lower_bound, upper_bound, bd):

    MAE      = 12318
    subtotal = sum(bd.values())
    ins_adj  = 0.0
    net      = round(subtotal - ins_adj, 2)
    drug_name, drug_class = parse_medication(medication_display)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'DM Sans', sans-serif;
    background: #f1f5f9;
    padding: 16px;
    color: #1e293b;
  }}

  /* Print button */
  .print-bar {{
    display: flex; justify-content: flex-end;
    gap: 10px; margin-bottom: 12px;
  }}
  .btn-print {{
    background: linear-gradient(135deg, #0f2027, #2c5364);
    color: #ffffff; border: none;
    padding: 10px 24px; border-radius: 8px;
    font-size: 0.92rem; font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer; letter-spacing: 0.5px;
    display: flex; align-items: center; gap: 8px;
  }}
  .btn-print:hover {{ opacity: 0.9; }}

  /* A4 page */
  .page {{
    background: #ffffff;
    width: 210mm; min-height: 297mm;
    margin: 0 auto;
    border-radius: 4px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    overflow: hidden;
    display: flex; flex-direction: column;
  }}

  /* Header */
  .bill-header {{
    background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
    padding: 28px 36px 22px 36px;
    display: flex; justify-content: space-between; align-items: flex-start;
  }}
  .brand-name {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem; color: #ffffff; letter-spacing: 0.5px;
    margin-bottom: 3px;
  }}
  .brand-sub  {{ font-size: 0.70rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; }}
  .brand-addr {{ color: #94a3b8; font-size: 0.73rem; margin-top: 8px; line-height: 1.7; }}

  .bill-title-blk {{ text-align: right; }}
  .bill-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem; font-weight: 700;
    color: #ffffff; letter-spacing: 6px;
  }}
  .bill-patno {{ font-size: 0.85rem; color: #7dd3fc; margin-top: 6px; letter-spacing: 1px; }}

  /* Meta strip */
  .bill-meta {{
    display: flex; justify-content: space-between; flex-wrap: wrap;
    background: #f0f4f8; border-bottom: 2px solid #dde3ea;
  }}
  .meta-cell {{
    padding: 16px 24px; border-right: 1px solid #dde3ea;
    flex: 1; min-width: 150px;
  }}
  .meta-cell:last-child {{ border-right: none; }}
  .meta-lbl {{
    font-size: 0.63rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 1.4px;
    margin-bottom: 5px; font-weight: 600;
  }}
  .meta-val  {{ font-size: 0.88rem; color: #1e3a4a; font-weight: 500; line-height: 1.5; }}
  .meta-name {{ font-size: 0.98rem; font-weight: 700; color: #0f2027; }}
  .meta-pid  {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem; font-weight: 800;
    color: #0f2027; letter-spacing: 3px;
  }}

  /* Body */
  .bill-body {{ padding: 28px 36px; flex: 1; }}

  /* Line-items table */
  .bill-table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; }}
  .bill-table thead tr {{ background: #1e3a4a; }}
  .bill-table thead th {{
    padding: 11px 14px; font-size: 0.70rem;
    text-transform: uppercase; letter-spacing: 1.1px;
    font-weight: 600; color: #ffffff; text-align: left;
  }}
  .bill-table thead th:last-child {{ text-align: right; }}
  .bill-table tbody tr {{ border-bottom: 1px solid #eef2f7; }}
  .bill-table tbody tr:nth-child(even) {{ background: #f8fafc; }}
  .bill-table tbody td {{
    padding: 12px 14px; font-size: 0.88rem;
    color: #334155; vertical-align: top;
  }}
  .bill-table tbody td:last-child {{
    text-align: right; font-weight: 600;
    white-space: nowrap; color: #0f2027;
  }}
  .item-desc {{ display: block; font-size: 0.74rem; color: #64748b; margin-top: 3px; }}

  /* Totals */
  .totals-wrap {{ display: flex; justify-content: flex-end; margin-bottom: 8px; }}
  .totals-box  {{ width: 340px; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
  .tot-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 16px; font-size: 0.88rem; color: #475569;
    border-bottom: 1px solid #f1f5f9; background: #ffffff;
  }}
  .tot-row span:last-child {{ font-weight: 600; }}
  .tot-green {{ color: #16a34a !important; }}
  .tot-total {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 16px;
    background: linear-gradient(135deg, #0f2027, #2c5364);
  }}
  .tot-total span {{ color: #ffffff; font-size: 1.1rem; font-weight: 700; letter-spacing: 0.3px; }}
  .tot-range {{
    padding: 10px 16px; font-size: 0.74rem; color: #64748b;
    line-height: 1.6; background: #f8fafc; border-top: 1px solid #e2e8f0;
  }}

  /* Signature strip */
  .sig-strip {{
    display: flex; justify-content: space-between;
    margin-top: 36px; padding-top: 16px;
    border-top: 1px dashed #cbd5e1;
  }}
  .sig-box  {{ text-align: center; width: 160px; }}
  .sig-line {{ border-bottom: 1px solid #334155; margin-bottom: 6px; height: 36px; }}
  .sig-lbl  {{ font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }}

  /* Footer */
  .bill-footer {{
    background: #1e3a4a; padding: 14px 36px;
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 10px;
    margin-top: auto;
  }}
  .foot-note  {{ font-size: 0.72rem; color: #94a3b8; line-height: 1.6; }}
  .foot-stamp {{
    background: rgba(255,255,255,0.12); color: #e2e8f0;
    padding: 6px 16px; border-radius: 4px;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 1.8px; text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.2);
    white-space: nowrap;
  }}

  /* Print styles */
  @media print {{
    @page {{ size: A4 portrait; margin: 0; }}
    body {{
      background: #ffffff !important; padding: 0 !important;
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }}
    .print-bar {{ display: none !important; }}
    .page {{
      width: 210mm !important; min-height: 297mm !important;
      margin: 0 !important; border-radius: 0 !important; box-shadow: none !important;
    }}
    .bill-header, .bill-footer, .tot-total, .bill-table thead tr {{
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }}
  }}
</style>
</head>
<body>

<div class="print-bar">
  <button class="btn-print" onclick="window.print()">
    &#128438;&nbsp; Download / Print PDF
  </button>
</div>

<div class="page">

  <!-- HEADER -->
  <div class="bill-header">
    <div>
      <div class="brand-name">Medical Cost AI</div>
      <div class="brand-sub">Patient Billing System</div>
      <div class="brand-addr">
        AI-Powered Medical Cost Prediction Platform<br>
        support@medcostai.in &nbsp;|&nbsp; +91 44 2222 3333
      </div>
    </div>
    <div class="bill-title-blk">
      <div class="bill-title">BILLING</div>
      <div class="bill-patno">Patient No: #{patient_id}</div>
      <div class="bill-patno">Date: {reg_date} &nbsp; {reg_time}</div>
    </div>
  </div>

  <!-- META STRIP -->
  <div class="bill-meta">
    <div class="meta-cell">
      <div class="meta-lbl">Billed To</div>
      <div class="meta-name">{patient_name}</div>
      <div class="meta-val">Age {age} &nbsp;&middot;&nbsp; {gender}</div>
      <div class="meta-val">Blood Type: {blood_type}</div>
    </div>
    <div class="meta-cell">
      <div class="meta-lbl">Admission Details</div>
      <div class="meta-val">{admission_type} Admission</div>
      <div class="meta-val">Room Type: {room_type}</div>
      <div class="meta-val">Stay Duration: {planned_stay} day(s)</div>
    </div>
    <div class="meta-cell">
      <div class="meta-lbl">Insurance Provider</div>
      <div class="meta-val" style="font-weight:700;">{insurance_provider}</div>
      <div class="meta-lbl" style="margin-top:10px;">Primary Diagnosis</div>
      <div class="meta-val" style="font-weight:700;">{medical_condition}</div>
    </div>
    <div class="meta-cell">
      <div class="meta-lbl">Patient ID</div>
      <div class="meta-pid">{patient_id}</div>
      <div class="meta-lbl" style="margin-top:10px;">Medication</div>
      <div class="meta-val">{drug_name}</div>
    </div>
  </div>

  <!-- BODY -->
  <div class="bill-body">

    <table class="bill-table">
      <thead>
        <tr>
          <th style="width:36px;">#</th>
          <th>Description</th>
          <th style="width:50px;text-align:center;">Qty</th>
          <th style="width:130px;">Amount (&#8377;)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>Base Hospitalization
              <span class="item-desc">Room ({room_type}), nursing care, facility &amp; overhead charges</span></td>
          <td style="text-align:center;">1</td>
          <td>&#8377;{bd['base']:,.2f}</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Medical Condition &mdash; {medical_condition}
              <span class="item-desc">Risk level {RISK_MAP[medical_condition]}/3 &nbsp;&middot;&nbsp; Diagnostic &amp; condition-specific treatment costs</span></td>
          <td style="text-align:center;">1</td>
          <td>&#8377;{bd['condition']:,.2f}</td>
        </tr>
        <tr>
          <td>3</td>
          <td>Length of Stay Charges
              <span class="item-desc">{planned_stay} day(s) &nbsp;&middot;&nbsp; {admission_type} admission &nbsp;&middot;&nbsp; Daily ward charges</span></td>
          <td style="text-align:center;">{planned_stay}</td>
          <td>&#8377;{bd['stay']:,.2f}</td>
        </tr>
        <tr>
          <td>4</td>
          <td>Medication &mdash; {drug_name}
              <span class="item-desc">{drug_class} &nbsp;&middot;&nbsp; Prescription drugs &amp; administration charges</span></td>
          <td style="text-align:center;">1</td>
          <td>&#8377;{bd['medication']:,.2f}</td>
        </tr>
        <tr>
          <td>5</td>
          <td>Ancillary &amp; Other Charges
              <span class="item-desc">Laboratory tests, imaging, consumables &amp; miscellaneous services</span></td>
          <td style="text-align:center;">1</td>
          <td>&#8377;{bd['other']:,.2f}</td>
        </tr>
      </tbody>
    </table>

    <!-- Totals -->
    <div class="totals-wrap">
      <div class="totals-box">
        <div class="tot-row">
          <span>Subtotal</span>
          <span>&#8377;{subtotal:,.2f}</span>
        </div>
        <div class="tot-row">
          <span>Insurance Adjustment</span>
          <span class="tot-green">&#8722; &#8377;{ins_adj:,.2f}</span>
        </div>
        <div class="tot-total">
          <span>TOTAL PAYABLE</span>
          <span>&#8377;{net:,.2f}</span>
        </div>
        <div class="tot-range">
          Estimated range: &#8377;{lower_bound:,.0f} &ndash; &#8377;{upper_bound:,.0f}
          &nbsp;&nbsp;(&#177;&#8377;{MAE:,} based on model MAE)
        </div>
      </div>
    </div>

    <!-- Signature strip -->
    <div class="sig-strip">
      <div class="sig-box">
        <div class="sig-line"></div>
        <div class="sig-lbl">Patient / Attender Signature</div>
      </div>
      <div class="sig-box">
        <div class="sig-line"></div>
        <div class="sig-lbl">Authorised Signatory</div>
      </div>
      <div class="sig-box">
        <div class="sig-line"></div>
        <div class="sig-lbl">Accounts Department</div>
      </div>
    </div>

  </div>

  <!-- FOOTER -->
  <div class="bill-footer">
    <div class="foot-note">
      &#9877; This is a <strong>cost estimate</strong> generated by AI / ML model prediction.
      Final billing may vary based on actual treatment and procedures performed.<br>
      Queries: support@medcostai.in &nbsp;|&nbsp; Medical Cost AI Platform
    </div>
    <div class="foot-stamp">AI Cost Estimate</div>
  </div>

</div>
</body>
</html>"""

    components.html(html, height=1080, scrolling=True)


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

st.sidebar.title("🏥 MedCost AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "NAVIGATION",
    ["🔮 Predict Cost", "🗂️ Patient Records", "📊 Data Analysis", "📈 Model Report"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="font-size:0.73rem;color:#64748b;text-align:center;">'
    'Patient Medical Cost Prediction<br>&#169; 2025 Medical Cost AI</p>',
    unsafe_allow_html=True
)

init_csv()


# ══════════════════════════════════════════════════════
# PAGE 1 — PREDICT COST
# ══════════════════════════════════════════════════════

if page == "🔮 Predict Cost":

    st.title("🏥 Patient Medical Cost Prediction")
    st.markdown("Fill in all sections. Stay duration and room type are **assigned automatically** based on diagnosis. Cost preview updates live as you fill the form.")

    # ── Condition picker (outside form for reactivity) ──
    if 'selected_condition' not in st.session_state:
        st.session_state.selected_condition = MEDICAL_CONDITIONS[0]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 Select Primary Diagnosis First</div>', unsafe_allow_html=True)
    st.markdown("Stay duration, room type and medication list update automatically.")
    pre_condition = st.selectbox(
        "Primary Diagnosis", MEDICAL_CONDITIONS, key='pre_condition',
        index=MEDICAL_CONDITIONS.index(st.session_state.selected_condition)
    )
    st.session_state.selected_condition = pre_condition
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Auto-assigned values based on condition ─────────
    auto_stay      = CONDITION_STAY_DAYS[st.session_state.selected_condition]
    auto_room      = CONDITION_ROOM_TYPE[st.session_state.selected_condition]
    available_meds = CONDITION_MEDICATIONS[st.session_state.selected_condition]

    # Show auto-assigned info box
    st.markdown(
        f'<div class="auto-info-box">'
        f'<div class="auto-info-item">'
        f'<span class="auto-info-label">&#128197; Auto Planned Stay</span>'
        f'<span class="auto-info-value">{auto_stay} days</span>'
        f'</div>'
        f'<div class="auto-info-item">'
        f'<span class="auto-info-label">&#127963; Auto Room Type</span>'
        f'<span class="auto-info-value">{auto_room}</span>'
        f'</div>'
        f'<div class="auto-info-item">'
        f'<span class="auto-info-label">&#9888; Risk Level</span>'
        f'<span class="auto-info-value">{RISK_MAP[st.session_state.selected_condition]}/3</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Live preview (outside form — updates as user changes fields) ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Live Cost Preview</div>', unsafe_allow_html=True)
    st.markdown("Preview updates as you change Age, Gender, Insurance, Test Results, or Admission Type.")

    prev_col1, prev_col2, prev_col3, prev_col4 = st.columns(4)
    with prev_col1:
        prev_age    = st.slider("Age", AGE_MIN, AGE_MAX, 30, key='prev_age')
        prev_gender = st.radio("Gender", ["Male", "Female"], horizontal=True, key='prev_gender')
    with prev_col2:
        prev_blood    = st.selectbox("Blood Type",        BLOOD_TYPES,         key='prev_blood')
        prev_test     = st.selectbox("Test Results",      TEST_RESULTS,        key='prev_test')
    with prev_col3:
        prev_ins      = st.selectbox("Insurance Provider", INSURANCE_PROVIDERS, key='prev_ins')
        prev_admission = st.selectbox("Admission Type",   ADMISSION_TYPES,     key='prev_admission')
    with prev_col4:
        prev_med = st.selectbox(
            f"Medication for {st.session_state.selected_condition}",
            available_meds, key='prev_med'
        )

    # Run live prediction
    live_pred, live_low, live_high, live_bd = quick_predict(
        prev_age, prev_gender, prev_blood, prev_ins,
        st.session_state.selected_condition,
        prev_test, prev_admission, auto_stay
    )

    if live_pred is not None:
        st.markdown(
            f'<div class="preview-box">'
            f'<div class="preview-title">&#128200; Estimated Cost Preview</div>'
            f'<div class="preview-cost">&#8377;{live_pred:,.2f}</div>'
            f'<div class="preview-range">Range: &#8377;{live_low:,.0f} &ndash; &#8377;{live_high:,.0f}</div>'
            f'<div class="preview-rows">'
            f'<span class="preview-row">&#9679; Base: &#8377;{live_bd["base"]:,.0f}</span>'
            f'<span class="preview-row">&#9679; Condition: &#8377;{live_bd["condition"]:,.0f}</span>'
            f'<span class="preview-row">&#9679; Stay ({auto_stay}d): &#8377;{live_bd["stay"]:,.0f}</span>'
            f'<span class="preview-row">&#9679; Medication: &#8377;{live_bd["medication"]:,.0f}</span>'
            f'<span class="preview-row">&#9679; Other: &#8377;{live_bd["other"]:,.0f}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Registration form ───────────────────────────────
    with st.form("patient_form"):

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Patient Personal Details</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            patient_name  = st.text_input("Full Name *",      placeholder="e.g. Ramesh Kumar")
            patient_phone = st.text_input("Contact Number *", placeholder="e.g. 9876543210")
        with p2:
            patient_address = st.text_area("Address *", placeholder="Door No, Street, City, PIN", height=100)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👥 Attender / Guardian Details</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            attender_name         = st.text_input("Attender Full Name *",      placeholder="e.g. Priya Ramesh")
            attender_relationship = st.selectbox("Relationship to Patient",
                                                  ["Spouse", "Parent", "Child", "Sibling", "Friend", "Other"])
            attender_phone        = st.text_input("Attender Contact Number *", placeholder="e.g. 9876543211")
        with a2:
            attender_address = st.text_area("Attender Address *",
                                             placeholder="Same as patient or different address", height=120)
        st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            f"✅ **Auto-assigned for {st.session_state.selected_condition}:** "
            f"Stay = **{auto_stay} days** · Room = **{auto_room}** · Risk = **{RISK_MAP[st.session_state.selected_condition]}/3**  \n"
            f"The values from the Live Cost Preview above will be used for registration."
        )

        submitted = st.form_submit_button("💰 Register Patient & Generate Billing", type="primary")

    # Medication info panel
    if prev_med:
        drug_name_p, drug_class_p = parse_medication(prev_med)
        bg, fg = DRUG_CLASS_COLORS.get(drug_class_p, ('#f1f5f9', '#334155'))
        st.markdown(
            f'<div class="med-info-box">'
            f'<div class="med-info-drug">💊 {drug_name_p}</div>'
            f'<div class="med-info-row"><b>Drug class:</b> '
            f'<span style="background:{bg};color:{fg};font-size:0.80rem;font-weight:600;'
            f'padding:2px 10px;border-radius:20px;">{drug_class_p}</span></div>'
            f'<div class="med-info-row"><b>Prescribed for:</b> {st.session_state.selected_condition}</div>'
            f'<div class="med-info-row" style="color:#64748b;font-size:0.82rem;">'
            f'ℹ️ Consult your physician before starting or changing any medication.</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── After submit ──────────────────────────────────────
    if submitted:
        errors = []
        if not patient_name.strip():     errors.append("Patient Name is required.")
        if not patient_phone.strip():    errors.append("Patient Contact Number is required.")
        if not patient_address.strip():  errors.append("Patient Address is required.")
        if not attender_name.strip():    errors.append("Attender Name is required.")
        if not attender_phone.strip():   errors.append("Attender Contact Number is required.")
        if not attender_address.strip(): errors.append("Attender Address is required.")

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            with st.spinner("Registering patient and generating billing…"):
                try:
                    # Use values from live preview section
                    age             = prev_age
                    gender          = prev_gender
                    blood_type      = prev_blood
                    insurance_provider = prev_ins
                    test_result     = prev_test
                    admission_type  = prev_admission
                    medication_display = prev_med
                    planned_stay    = auto_stay
                    room_type       = auto_room

                    encoded_med = closest_dataset_medication(st.session_state.selected_condition)
                    df_in, age_group = build_input_df(
                        age, gender, blood_type, insurance_provider,
                        st.session_state.selected_condition,
                        test_result, admission_type, encoded_med, planned_stay
                    )
                    prediction  = float(max(model.predict(df_in)[0][0], 0))
                    MAE         = 12318
                    lower_bound = max(prediction - MAE, 0)
                    upper_bound = prediction + MAE
                    bd          = compute_breakdown(prediction, st.session_state.selected_condition, planned_stay)

                    patient_id = get_next_patient_id()
                    now        = datetime.datetime.now()
                    reg_date   = now.strftime("%d %b %Y")
                    reg_time   = now.strftime("%I:%M %p")

                    save_patient({
                        'Patient ID':                   patient_id,
                        'Registration Date':            reg_date,
                        'Registration Time':            reg_time,
                        'Patient Name':                 patient_name.strip(),
                        'Patient Age':                  age,
                        'Patient Gender':               gender,
                        'Patient Phone':                patient_phone.strip(),
                        'Patient Address':              patient_address.strip(),
                        'Attender Name':                attender_name.strip(),
                        'Attender Relationship':        attender_relationship,
                        'Attender Phone':               attender_phone.strip(),
                        'Attender Address':             attender_address.strip(),
                        'Blood Type':                   blood_type,
                        'Insurance Provider':           insurance_provider,
                        'Medical Condition':            st.session_state.selected_condition,
                        'Test Results':                 test_result,
                        'Admission Type':               admission_type,
                        'Medication':                   medication_display,
                        'Medication (Dataset Encoded)': encoded_med,
                        'Planned Stay (days)':          planned_stay,
                        'Room Type':                    room_type,
                        'Predicted Cost (INR)':         round(prediction, 2),
                        'Lower Estimate (INR)':         round(lower_bound, 2),
                        'Upper Estimate (INR)':         round(upper_bound, 2),
                        'Base Hospitalization':         bd['base'],
                        'Medical Condition Cost':       bd['condition'],
                        'Length of Stay Cost':          bd['stay'],
                        'Medication Cost':              bd['medication'],
                        'Other Charges':                bd['other'],
                    })

                    st.success("✅ Patient registered successfully!")

                    id_col, met_col = st.columns([1, 1])
                    with id_col:
                        st.markdown("**Assigned Patient ID**")
                        st.markdown(f'<div class="pid-badge">{patient_id}</div>', unsafe_allow_html=True)
                        st.caption("Use this ID in **Patient Records** to retrieve all details anytime.")
                    with met_col:
                        st.metric("Total Billing Amount", f"₹{prediction:,.2f}")
                        st.metric("Lower Estimate",  f"₹{lower_bound:,.0f}")
                        st.metric("Upper Estimate",  f"₹{upper_bound:,.0f}")

                    st.markdown("---")
                    st.subheader("🧾 Patient Billing")
                    st.caption("Click **Download / Print PDF** inside the billing to save as PDF.")
                    render_billing(
                        patient_name=patient_name.strip(),
                        patient_id=patient_id,
                        reg_date=reg_date, reg_time=reg_time,
                        age=age, gender=gender, blood_type=blood_type,
                        insurance_provider=insurance_provider,
                        medical_condition=st.session_state.selected_condition,
                        admission_type=admission_type,
                        medication_display=medication_display,
                        planned_stay=planned_stay, room_type=room_type,
                        prediction=prediction,
                        lower_bound=lower_bound, upper_bound=upper_bound,
                        bd=bd
                    )

                    st.subheader("Cost Breakdown — Visual")
                    pie_df = pd.DataFrame({
                        "Component": ["Base Hospitalization", "Medical Condition",
                                      "Length of Stay", "Medication", "Other Charges"],
                        "Amount":    [bd['base'], bd['condition'], bd['stay'],
                                      bd['medication'], bd['other']]
                    })
                    fig_pie = px.pie(
                        pie_df, values="Amount", names="Component",
                        color_discrete_sequence=['#0f2027','#1a3a4a','#2c5364','#3d7a8a','#5ba4b0']
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=380)
                    st.plotly_chart(fig_pie, use_container_width=True)

                except Exception as exc:
                    st.error(f"❌ Error: {exc}")
                    st.info("Check all fields and that model files exist.")


# ══════════════════════════════════════════════════════
# PAGE 2 — PATIENT RECORDS
# ══════════════════════════════════════════════════════

elif page == "🗂️ Patient Records":

    st.title("🗂️ Patient Records")
    st.markdown("Search a patient by ID, or browse and download all records.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Look Up Patient by ID</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns([4, 1])
    with sc1:
        lookup_id = st.text_input("Patient ID", placeholder="e.g. P001", label_visibility="collapsed")
    with sc2:
        do_search = st.button("Search", use_container_width=True)

    if do_search and lookup_id.strip():
        patient = get_patient_by_id(lookup_id.strip())
        if patient is None:
            st.error(f"No patient found with ID **{lookup_id.upper()}**.")
        else:
            st.success(f"✅ Found — **{patient['Patient Name']}** ({patient['Patient ID']})")
            st.markdown("---")

            tab_p, tab_m, tab_i = st.tabs(["👤 Personal Details", "🩺 Medical Details", "🧾 Billing"])

            with tab_p:
                tp1, tp2 = st.columns(2)
                with tp1:
                    st.markdown("**Patient**")
                    st.write(f"**Name:** {patient['Patient Name']}")
                    st.write(f"**Age:** {patient['Patient Age']}  |  **Gender:** {patient['Patient Gender']}")
                    st.write(f"**Blood Type:** {patient['Blood Type']}")
                    st.write(f"**Phone:** {patient['Patient Phone']}")
                    st.write(f"**Address:** {patient['Patient Address']}")
                with tp2:
                    st.markdown("**Attender**")
                    st.write(f"**Name:** {patient['Attender Name']}")
                    st.write(f"**Relationship:** {patient['Attender Relationship']}")
                    st.write(f"**Phone:** {patient['Attender Phone']}")
                    st.write(f"**Address:** {patient['Attender Address']}")
                st.caption(f"Registered on {patient['Registration Date']} at {patient['Registration Time']}")

            with tab_m:
                tm1, tm2 = st.columns(2)
                with tm1:
                    st.write(f"**Primary Diagnosis:** {patient['Medical Condition']}")
                    st.write(f"**Admission Type:** {patient['Admission Type']}")
                    st.write(f"**Room Type:** {patient['Room Type']}")
                    st.write(f"**Planned Stay:** {patient['Planned Stay (days)']} day(s)")
                with tm2:
                    st.write(f"**Insurance Provider:** {patient['Insurance Provider']}")
                    st.write(f"**Test Results:** {patient['Test Results']}")
                    med_full      = str(patient.get('Medication', ''))
                    d_name, d_cls = parse_medication(med_full)
                    bg, fg        = DRUG_CLASS_COLORS.get(d_cls, ('#f1f5f9', '#334155'))
                    st.markdown(
                        f"**Medication:** {d_name} &nbsp;"
                        f'<span style="background:{bg};color:{fg};font-size:0.76rem;font-weight:600;'
                        f'padding:2px 9px;border-radius:20px;">{d_cls}</span>',
                        unsafe_allow_html=True
                    )

            with tab_i:
                st.caption("Click **Download / Print PDF** inside the billing to save as PDF.")
                bd_r = {
                    'base':       float(patient['Base Hospitalization']),
                    'condition':  float(patient['Medical Condition Cost']),
                    'stay':       float(patient['Length of Stay Cost']),
                    'medication': float(patient['Medication Cost']),
                    'other':      float(patient['Other Charges']),
                }
                render_billing(
                    patient_name=patient['Patient Name'],
                    patient_id=patient['Patient ID'],
                    reg_date=patient['Registration Date'],
                    reg_time=patient['Registration Time'],
                    age=patient['Patient Age'],
                    gender=patient['Patient Gender'],
                    blood_type=patient['Blood Type'],
                    insurance_provider=patient['Insurance Provider'],
                    medical_condition=patient['Medical Condition'],
                    admission_type=patient['Admission Type'],
                    medication_display=str(patient.get('Medication', '')),
                    planned_stay=int(patient['Planned Stay (days)']),
                    room_type=patient['Room Type'],
                    prediction=float(patient['Predicted Cost (INR)']),
                    lower_bound=float(patient['Lower Estimate (INR)']),
                    upper_bound=float(patient['Upper Estimate (INR)']),
                    bd=bd_r
                )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 All Registered Patients")
    all_df = load_all_patients()

    if all_df.empty:
        st.info("No patients registered yet. Use **Predict Cost** to add the first patient.")
    else:
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Total Patients",     f"{len(all_df):,}")
        rm2.metric("Avg Predicted Cost", f"₹{all_df['Predicted Cost (INR)'].mean():,.0f}")
        rm3.metric("Highest Cost",       f"₹{all_df['Predicted Cost (INR)'].max():,.0f}")
        rm4.metric("Lowest Cost",        f"₹{all_df['Predicted Cost (INR)'].min():,.0f}")
        st.markdown("---")

        fc1, fc2 = st.columns(2)
        with fc1:
            name_filter = st.text_input("🔎 Filter by Name", placeholder="Type to search…")
        with fc2:
            cond_filter = st.selectbox("Filter by Condition", ["All"] + MEDICAL_CONDITIONS)

        display = all_df.copy()
        if name_filter.strip():
            display = display[display['Patient Name'].str.contains(
                name_filter.strip(), case=False, na=False)]
        if cond_filter != "All":
            display = display[display['Medical Condition'] == cond_filter]

        show_cols = [
            'Patient ID', 'Registration Date', 'Patient Name', 'Patient Age',
            'Patient Gender', 'Patient Phone', 'Medical Condition', 'Medication',
            'Insurance Provider', 'Planned Stay (days)',
            'Predicted Cost (INR)', 'Lower Estimate (INR)', 'Upper Estimate (INR)'
        ]
        show_cols = [c for c in show_cols if c in display.columns]
        st.dataframe(display[show_cols].reset_index(drop=True), use_container_width=True)
        st.caption(f"Showing {len(display)} of {len(all_df)} records.")

        st.download_button(
            label="⬇️ Download All Records as CSV",
            data=all_df.to_csv(index=False).encode('utf-8'),
            file_name="all_patients.csv",
            mime="text/csv"
        )


# ══════════════════════════════════════════════════════
# PAGE 3 — DATA ANALYSIS
# ══════════════════════════════════════════════════════

elif page == "📊 Data Analysis":

    st.title("📊 Dataset Analysis")
    st.markdown("Exploratory Data Analysis of the healthcare dataset.")

    @st.cache_data
    def load_raw_data():
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
        st.warning(f"Dataset file '{RAW_DATA_PATH}' not found.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Patients",  f"{len(raw):,}")
        d2.metric("Average Bill",    f"₹{raw['Billing Amount'].mean():,.0f}")
        d3.metric("Avg Stay (days)", f"{raw['Length of Stay'].mean():.1f}")
        d4.metric("Age Range",       f"{raw['Age'].min()}–{raw['Age'].max()}")
        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Billing Amount Distribution")
            st.plotly_chart(px.histogram(raw, x='Billing Amount', nbins=50,
                color_discrete_sequence=['#2c5364'])
                .update_layout(showlegend=False, height=340), use_container_width=True)
        with c2:
            st.subheader("Patients by Medical Condition")
            cc = raw['Medical Condition'].value_counts().reset_index()
            cc.columns = ['Condition', 'Count']
            st.plotly_chart(px.bar(cc, x='Condition', y='Count', color='Condition',
                color_discrete_sequence=px.colors.qualitative.Pastel)
                .update_layout(showlegend=False, height=340), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Avg Billing by Condition")
            ac = raw.groupby('Medical Condition')['Billing Amount'].mean()\
                    .reset_index().sort_values('Billing Amount', ascending=False)
            st.plotly_chart(px.bar(ac, x='Medical Condition', y='Billing Amount',
                color='Medical Condition',
                color_discrete_sequence=px.colors.qualitative.Safe)
                .update_layout(showlegend=False, height=340), use_container_width=True)
        with c4:
            st.subheader("Insurance Provider Distribution")
            ic = raw['Insurance Provider'].value_counts().reset_index()
            ic.columns = ['Provider', 'Count']
            st.plotly_chart(px.pie(ic, values='Count', names='Provider',
                color_discrete_sequence=px.colors.qualitative.Pastel)
                .update_layout(height=340), use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            st.subheader("Age Distribution")
            st.plotly_chart(px.histogram(raw, x='Age', nbins=30,
                color_discrete_sequence=['#203a43'])
                .update_layout(showlegend=False, height=340), use_container_width=True)
        with c6:
            st.subheader("Admission Type Breakdown")
            adm = raw['Admission Type'].value_counts().reset_index()
            adm.columns = ['Type', 'Count']
            st.plotly_chart(px.bar(adm, x='Type', y='Count', color='Type',
                color_discrete_sequence=['#e74c3c', '#f39c12', '#2ecc71'])
                .update_layout(showlegend=False, height=340), use_container_width=True)

        st.markdown("---")
        st.subheader("Dataset Sample (first 10 rows)")
        st.dataframe(raw.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 4 — MODEL REPORT
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

    img_missing = [f for f, _ in charts if not os.path.exists(os.path.join(IMG_DIR, f))]
    if img_missing:
        st.warning("Charts not found. Run model_prediction.py first to generate the img/ folder.")
    else:
        for i in range(0, 9, 2):
            cl, cr = st.columns(2)
            f1, l1 = charts[i]
            cl.image(os.path.join(IMG_DIR, f1), caption=l1, use_column_width=True)
            if i + 1 < 9:
                f2, l2 = charts[i + 1]
                cr.image(os.path.join(IMG_DIR, f2), caption=l2, use_column_width=True)
        f10, l10 = charts[9]
        st.image(os.path.join(IMG_DIR, f10), caption=l10, use_column_width=True)

    st.markdown("---")
    st.subheader("Model Performance Summary")
    st.dataframe(pd.DataFrame({
        "Model":    ["Linear Regression", "Ridge Regression", "Random Forest",
                     "Gradient Boosting", "Neural Network"],
        "MAE (₹)":  ["₹12,320", "₹12,320", "₹12,317", "₹12,317", "₹12,318"],
        "RMSE (₹)": ["₹14,230", "₹14,230", "₹14,229", "₹14,238", "₹14,240"],
        "R²":       ["-0.0004", "-0.0004", "-0.0002", "-0.0015", "-0.0017"],
        "Status":   ["Baseline", "Baseline", "✅ Best", "Baseline", "Deep Learning"]
    }), use_container_width=True)

    st.markdown("---")
    st.info(
        "**Why is R² near 0?**\n\n"
        "The billing amounts in this dataset are synthetically generated with a "
        "uniform random distribution — every value from ₹0 to ₹52,764 appears "
        "with equal frequency. No patient feature (age, condition, medication, stay) "
        "correlates with billing. All 5 models correctly identify this — they predict "
        "near the dataset mean (₹25,595), which is the optimal answer when there is "
        "no learnable signal. This is an honest finding, not a code error."
    )
  
