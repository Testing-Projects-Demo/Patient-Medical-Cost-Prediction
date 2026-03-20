"""
config.py
---------
Central configuration file for the Patient Medical Cost Prediction project.
All hardcoded values, file paths, and model parameters are defined here.
Change values here — all scripts update automatically.
"""

# ── File Paths ─────────────────────────────────────────
RAW_DATA_PATH       = 'healthcare_dataset.csv'
PROCESSED_DATA_PATH = 'processed_data.csv'
MODEL_PATH          = 'patient_cost_model.keras'
RF_MODEL_PATH       = 'rf_model.pkl'
SCALER_PATH         = 'scaler.pkl'
FEATURES_PATH       = 'selected_features.pkl'
IMG_DIR             = 'img'

# ── Dataset Values (exact values from dataset) ─────────
MEDICAL_CONDITIONS = ['Arthritis', 'Asthma', 'Cancer', 'Diabetes', 'Hypertension', 'Obesity']
INSURANCE_PROVIDERS = ['Aetna', 'Blue Cross', 'Cigna', 'Medicare', 'UnitedHealthcare']
BLOOD_TYPES = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
ADMISSION_TYPES = ['Elective', 'Emergency', 'Urgent']
MEDICATIONS = ['Aspirin', 'Ibuprofen', 'Lipitor', 'Paracetamol', 'Penicillin']
TEST_RESULTS = ['Normal', 'Inconclusive', 'Abnormal']
GENDERS = ['Male', 'Female']
ROOM_TYPES = ['General Ward', 'Semi-Private', 'Private', 'ICU']

# ── Risk Scores per Condition ──────────────────────────
RISK_MAP = {
    'Cancer':      3,
    'Diabetes':    2,
    'Hypertension':2,
    'Obesity':     1,
    'Arthritis':   1,
    'Asthma':      1
}

# ── Test Result Encoding ───────────────────────────────
TEST_RESULT_MAP = {
    'Normal':       0,
    'Inconclusive': 1,
    'Abnormal':     2
}

# ── Age Group Bins ─────────────────────────────────────
AGE_BINS   = [0, 18, 35, 50, 65, 100]
AGE_LABELS = ['Child', 'Young Adult', 'Adult', 'Senior', 'Elder']
AGE_MIN    = 13
AGE_MAX    = 89

# ── Columns to Drop Before Training ───────────────────
COLUMNS_TO_DROP = ['Name', 'Room Number', 'Doctor', 'Hospital',
                   'Date of Admission', 'Discharge Date']

# ── Columns to Scale ───────────────────────────────────
SCALE_COLUMNS = ['Age', 'Length of Stay', 'Risk Score']

# ── One-Hot Encoding Columns ───────────────────────────
ONEHOT_COLUMNS = [
    'Blood Type', 'Medical Condition', 'Insurance Provider',
    'Admission Type', 'Medication', 'Age Group'
]

# ── Model Training Parameters ──────────────────────────
RANDOM_STATE     = 42
TEST_SIZE        = 0.30
VAL_SIZE         = 0.50
BATCH_SIZE       = 32
MAX_EPOCHS       = 100
LEARNING_RATE    = 0.001
EARLY_STOP_PATIENCE   = 10
REDUCE_LR_PATIENCE    = 5
REDUCE_LR_FACTOR      = 0.5
MIN_LR               = 1e-6
NUM_FEATURES         = 20

# ── Neural Network Architecture ────────────────────────
NN_LAYERS    = [128, 64, 32, 16]
DROPOUT_RATE = [0.3, 0.2, 0.1]

# ── Random Forest Parameters ───────────────────────────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 8

# ── Gradient Boosting Parameters ──────────────────────
GB_N_ESTIMATORS  = 200
GB_LEARNING_RATE = 0.05
GB_MAX_DEPTH     = 4

# ── App Settings ───────────────────────────────────────
APP_TITLE    = "Patient Medical Cost Prediction"
APP_ICON     = "🏥"
APP_LAYOUT   = "wide"
STAY_MIN     = 1
STAY_MAX     = 30
STAY_DEFAULT = 3