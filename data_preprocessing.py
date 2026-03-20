"""
data_preprocessing.py
---------------------
Step 1 of the Patient Medical Cost Prediction pipeline.
Loads raw healthcare data, cleans it, engineers features,
encodes categories, scales numeric columns, and saves
processed_data.csv + scaler.pkl for model training.
"""

import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH,
    COLUMNS_TO_DROP, ONEHOT_COLUMNS, SCALE_COLUMNS,
    AGE_BINS, AGE_LABELS, RISK_MAP, TEST_RESULT_MAP
)

# ── Logging setup ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw healthcare CSV and perform basic cleaning.

    Args:
        path: File path to the raw CSV dataset.

    Returns:
        Cleaned DataFrame with duplicates and nulls removed,
        and negative billing amounts filtered out.
    """
    log.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    before = len(df)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df[df['Billing Amount'] > 0]
    log.info(f"After cleaning: {len(df):,} rows  (removed {before - len(df):,} rows)")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns.

    Adds:
        - Length of Stay  : days between admission and discharge
        - Age Group       : categorical bins (Child / Young Adult / Adult / Senior / Elder)
        - Risk Score      : numeric severity score based on medical condition
        - Test Results    : encoded as 0=Normal / 1=Inconclusive / 2=Abnormal

    Args:
        df: Cleaned DataFrame from load_data().

    Returns:
        DataFrame with new feature columns added.
    """
    log.info("Engineering features...")

    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], dayfirst=True)
    df['Discharge Date']    = pd.to_datetime(df['Discharge Date'],    dayfirst=True)
    df['Length of Stay']    = (df['Discharge Date'] - df['Date of Admission']).dt.days

    df['Age Group']  = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS)
    df['Risk Score'] = df['Medical Condition'].map(RISK_MAP).fillna(0)
    df['Test Results'] = df['Test Results'].map(TEST_RESULT_MAP).fillna(0)

    log.info(f"Length of Stay range : {df['Length of Stay'].min()}–{df['Length of Stay'].max()} days")

    return df


def encode_and_scale(df: pd.DataFrame):
    """
    Encode categorical variables and scale numeric features.

    Steps:
        1. Drop columns not useful for prediction
        2. Label encode Gender  (Male=1, Female=0)
        3. One-hot encode all categorical columns
        4. StandardScale: Age, Length of Stay, Risk Score

    Args:
        df: DataFrame after feature engineering.

    Returns:
        Tuple of (processed DataFrame, fitted StandardScaler)
    """
    log.info("Encoding and scaling features...")

    df = df.drop(COLUMNS_TO_DROP, axis=1)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    df = pd.get_dummies(df, columns=ONEHOT_COLUMNS)
    log.info(f"After one-hot encoding: {df.shape[1]} columns")

    scaler = StandardScaler()
    df[SCALE_COLUMNS] = scaler.fit_transform(df[SCALE_COLUMNS])
    log.info(f"Scaled columns: {SCALE_COLUMNS}")

    return df, scaler


def save_outputs(df: pd.DataFrame, scaler: StandardScaler) -> None:
    """
    Save the processed dataset and fitted scaler to disk.

    Args:
        df     : Fully processed DataFrame ready for model training.
        scaler : Fitted StandardScaler to be reused in app.py for prediction.
    """
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    log.info(f"Processed data saved → {PROCESSED_DATA_PATH}  shape={df.shape}")

    joblib.dump(scaler, SCALER_PATH)
    log.info(f"Scaler saved         → {SCALER_PATH}")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    """Run the full data preprocessing pipeline."""
    log.info("=" * 50)
    log.info("  DATA PREPROCESSING STARTED")
    log.info("=" * 50)

    df         = load_data(RAW_DATA_PATH)
    df         = engineer_features(df)
    df, scaler = encode_and_scale(df)
    save_outputs(df, scaler)

    log.info("=" * 50)
    log.info("  DATA PREPROCESSING COMPLETE")
    log.info("=" * 50)


if __name__ == '__main__':
    main()
