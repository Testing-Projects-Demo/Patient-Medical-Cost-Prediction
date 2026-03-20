"""
model_prediction.py
-------------------
Step 2 of the Patient Medical Cost Prediction pipeline.
Loads processed data, selects top features, trains and compares
5 ML models, evaluates on the test set, and saves 10 individual
chart images into the img/ folder.
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config import (
    PROCESSED_DATA_PATH, MODEL_PATH, RF_MODEL_PATH,
    FEATURES_PATH, RANDOM_STATE, TEST_SIZE, VAL_SIZE,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, MIN_LR, NUM_FEATURES,
    NN_LAYERS, DROPOUT_RATE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    GB_N_ESTIMATORS, GB_LEARNING_RATE, GB_MAX_DEPTH
)

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Create img/ folder ─────────────────────────────────
IMG_DIR = 'img'
os.makedirs(IMG_DIR, exist_ok=True)


def save_chart(fig, filename: str) -> None:
    """
    Save a matplotlib figure as a PNG into the img/ folder and close it.

    Args:
        fig      : matplotlib Figure object
        filename : filename without path, e.g. '01_neural_net_loss.png'
    """
    path = os.path.join(IMG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Chart saved → {path}")


# ══════════════════════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════════════════════

def load_and_prepare(path: str):
    """
    Load processed CSV, impute NaN, and select top features using SelectKBest.

    Args:
        path: Path to processed_data.csv

    Returns:
        Tuple of (X DataFrame, y Series, selected feature names)
    """
    log.info(f"Loading processed data from: {path}")
    df = pd.read_csv(path)
    X  = df.drop('Billing Amount', axis=1)
    y  = df['Billing Amount']
    log.info(f"Dataset: {df.shape[0]:,} rows | mean=₹{y.mean():,.0f} | std=₹{y.std():,.0f}")

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    imputer   = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_cols    = X.columns[~pd.isnull(X).all()]
    X         = pd.DataFrame(X_imputed, columns=X_cols)

    selector          = SelectKBest(score_func=f_regression, k=NUM_FEATURES)
    X_selected        = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    X                 = pd.DataFrame(X_selected, columns=selected_features)
    log.info(f"Top {NUM_FEATURES} features selected")

    return X, y, selected_features


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split data into train (70%), validation (15%), test (15%) sets.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    log.info(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Train Linear Regression, Ridge, Random Forest, and Gradient Boosting.

    Args:
        X_train, X_test: Feature splits
        y_train, y_test: Target splits

    Returns:
        Dict mapping model name to results (mae, rmse, r2, pred, model)
    """
    log.info("Training baseline models...")
    results = {}

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression':  Ridge(alpha=1.0),
        'Random Forest':     RandomForestRegressor(
                                 n_estimators=RF_N_ESTIMATORS,
                                 max_depth=RF_MAX_DEPTH,
                                 random_state=RANDOM_STATE,
                                 n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(
                                 n_estimators=GB_N_ESTIMATORS,
                                 learning_rate=GB_LEARNING_RATE,
                                 max_depth=GB_MAX_DEPTH,
                                 random_state=RANDOM_STATE)
    }

    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[name] = {
            'mae':   mean_absolute_error(y_test, pred),
            'rmse':  np.sqrt(mean_squared_error(y_test, pred)),
            'r2':    r2_score(y_test, pred),
            'pred':  pred,
            'model': m
        }
        log.info(f"{name:<22} R²={results[name]['r2']:.4f}  MAE=₹{results[name]['mae']:,.0f}")

    return results


def build_neural_network(input_dim: int) -> Sequential:
    """
    Build a deep neural network for regression with Dropout and BatchNormalization.

    Args:
        input_dim: Number of input features.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        Dense(NN_LAYERS[0], input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(DROPOUT_RATE[0]),
        Dense(NN_LAYERS[1], activation='relu'),
        BatchNormalization(),
        Dropout(DROPOUT_RATE[1]),
        Dense(NN_LAYERS[2], activation='relu'),
        Dropout(DROPOUT_RATE[2]),
        Dense(NN_LAYERS[3], activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='mse', metrics=['mae'])
    return model


def train_neural_network(model, X_train, X_val, y_train, y_val, X_test, y_test):
    """
    Train neural network with EarlyStopping and ReduceLROnPlateau callbacks.

    Args:
        model                  : Compiled Keras model
        X_train, X_val, X_test : Feature splits
        y_train, y_val, y_test : Target splits

    Returns:
        Tuple of (training history, results dict)
    """
    log.info("Training Neural Network...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR,
                          patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=0)
    ]
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=callbacks, verbose=1)

    pred = np.maximum(model.predict(X_test).flatten(), 0)
    results = {
        'mae':   mean_absolute_error(y_test, pred),
        'rmse':  np.sqrt(mean_squared_error(y_test, pred)),
        'r2':    r2_score(y_test, pred),
        'pred':  pred,
        'model': model
    }
    log.info(f"Neural Network         R²={results['r2']:.4f}  MAE=₹{results['mae']:,.0f}")
    return history, results


def print_evaluation(all_results: dict, y: pd.Series) -> str:
    """
    Print a formatted evaluation summary and warn if R² is near 0.

    Args:
        all_results : Dict of all model results
        y           : Full target Series

    Returns:
        Name of the best performing model.
    """
    best = max(all_results, key=lambda k: all_results[k]['r2'])
    log.info("=" * 55)
    log.info("  FINAL EVALUATION — TEST SET")
    log.info("=" * 55)
    for name, r in all_results.items():
        marker = " ← BEST" if name == best else ""
        log.info(f"{name:<22} MAE=₹{r['mae']:,.0f}  R²={r['r2']:.4f}{marker}")
    log.info("=" * 55)

    if all_results[best]['r2'] < 0.1:
        log.warning("All R² scores near 0 — billing amounts appear randomly assigned.")
        log.warning(f"Models predict near dataset mean ≈ ₹{y.mean():,.0f}")

    return best


# ══════════════════════════════════════════════════════
# 10 INDIVIDUAL CHART FUNCTIONS
# ══════════════════════════════════════════════════════

def chart_01_loss(history) -> None:
    """Chart 1 — Neural Network training vs validation Loss (MSE) curve."""
    epochs = range(1, len(history.history['loss']) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history['loss'],     '#2980b9', lw=2, label='Train Loss')
    ax.plot(epochs, history.history['val_loss'], '#e74c3c', lw=2, ls='--', label='Val Loss')
    ax.set_title('Neural Net — Loss (MSE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)')
    ax.legend(); ax.grid(True, alpha=0.3)
    save_chart(fig, '01_neural_net_loss.png')


def chart_02_mae(history) -> None:
    """Chart 2 — Neural Network training vs validation MAE curve."""
    epochs = range(1, len(history.history['mae']) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history['mae'],     '#27ae60', lw=2, label='Train MAE')
    ax.plot(epochs, history.history['val_mae'], '#f39c12', lw=2, ls='--', label='Val MAE')
    ax.set_title('Neural Net — MAE', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (₹)')
    ax.legend(); ax.grid(True, alpha=0.3)
    save_chart(fig, '02_neural_net_mae.png')


def chart_03_split(X_train, X_val, X_test) -> None:
    """Chart 3 — Train / Validation / Test sample count bar chart."""
    sizes  = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
    labels = ['Train', 'Validation', 'Test']
    colors = ['#2980b9', '#f39c12', '#e74c3c']
    total  = sum(sizes)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, sizes, color=colors, width=0.5, edgecolor='white')
    ax.set_title('Train / Val / Test Split', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 150,
                f'{s:,}\n({s/total*100:.0f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    save_chart(fig, '03_train_val_test_split.png')


def chart_04_r2(all_results: dict, best_name: str) -> None:
    """Chart 4 — All models R² score horizontal bar comparison."""
    model_names = list(all_results.keys())
    r2_scores   = [all_results[m]['r2'] for m in model_names]
    colors      = ['#e74c3c' if m == best_name else '#3498db' for m in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(model_names, r2_scores, color=colors, edgecolor='white')
    ax.set_title('Model Comparison — R² Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('R² (higher = better)')
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, v in zip(bars, r2_scores):
        ax.text(max(v, 0) + 0.0001, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9)
    save_chart(fig, '04_model_comparison_r2.png')


def chart_05_mae_compare(all_results: dict, best_name: str) -> None:
    """Chart 5 — All models MAE horizontal bar comparison."""
    model_names = list(all_results.keys())
    mae_scores  = [all_results[m]['mae'] for m in model_names]
    colors      = ['#e74c3c' if m == best_name else '#2ecc71' for m in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(model_names, mae_scores, color=colors, edgecolor='white')
    ax.set_title('Model Comparison — MAE (₹)', fontsize=14, fontweight='bold')
    ax.set_xlabel('MAE — lower = better')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, v in zip(bars, mae_scores):
        ax.text(v + 50, bar.get_y() + bar.get_height()/2,
                f'₹{v:,.0f}', va='center', fontsize=9)
    save_chart(fig, '05_model_comparison_mae.png')


def chart_06_nn_scatter(all_results: dict, y_test) -> None:
    """Chart 6 — Neural Network Actual vs Predicted scatter plot."""
    pred = all_results['Neural Network']['pred']
    mn   = min(float(y_test.min()), float(pred.min()))
    mx   = max(float(y_test.max()), float(pred.max()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, pred, alpha=0.3, color='#8e44ad', s=10, edgecolors='none')
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_title('Neural Net — Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Amount (₹)'); ax.set_ylabel('Predicted Amount (₹)')
    ax.legend(); ax.grid(True, alpha=0.3)
    save_chart(fig, '06_neural_net_actual_vs_predicted.png')


def chart_07_rf_scatter(all_results: dict, y_test) -> None:
    """Chart 7 — Random Forest Actual vs Predicted scatter plot."""
    pred = all_results['Random Forest']['pred']
    mn   = min(float(y_test.min()), float(pred.min()))
    mx   = max(float(y_test.max()), float(pred.max()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, pred, alpha=0.3, color='#27ae60', s=10, edgecolors='none')
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_title('Random Forest — Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Amount (₹)'); ax.set_ylabel('Predicted Amount (₹)')
    ax.legend(); ax.grid(True, alpha=0.3)
    save_chart(fig, '07_random_forest_actual_vs_predicted.png')


def chart_08_distribution(y: pd.Series) -> None:
    """Chart 8 — Billing Amount distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(y.mean(), color='red', lw=2, ls='--', label=f'Mean ₹{y.mean():,.0f}')
    ax.set_title('Billing Amount Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Billing Amount (₹)'); ax.set_ylabel('Frequency')
    ax.legend(); ax.grid(True, alpha=0.3)
    save_chart(fig, '08_billing_amount_distribution.png')


def chart_09_table(all_results: dict, best_name: str) -> None:
    """Chart 9 — Summary metrics table for all models."""
    model_names = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    table_data = [[name,
                   f"₹{all_results[name]['mae']:,.0f}",
                   f"₹{all_results[name]['rmse']:,.0f}",
                   f"{all_results[name]['r2']:.4f}"]
                  for name in model_names]

    tbl = ax.table(cellText=table_data,
                   colLabels=['Model', 'MAE (₹)', 'RMSE (₹)', 'R²'],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 2.0)

    best_idx = model_names.index(best_name)
    for col in range(4):
        tbl[best_idx + 1, col].set_facecolor('#d5f5e3')
        tbl[best_idx + 1, col].set_text_props(fontweight='bold')

    ax.set_title('All Models — Summary', fontsize=14, fontweight='bold', pad=20)
    save_chart(fig, '09_all_models_summary_table.png')


def chart_10_feature_importance(all_results: dict) -> None:
    """Chart 10 — Random Forest feature importance bar chart."""
    rf_model    = all_results['Random Forest']['model']
    importances = rf_model.feature_importances_

    if hasattr(rf_model, 'feature_names_in_'):
        feat_names = list(rf_model.feature_names_in_)
    else:
        feat_names = [f'Feature {i}' for i in range(len(importances))]

    # Shorten long names
    short_names = [
        n.replace('Medical Condition_',   'Condition: ')
         .replace('Insurance Provider_',  'Insurance: ')
         .replace('Blood Type_',          'Blood: ')
         .replace('Admission Type_',      'Admission: ')
         .replace('Medication_',          'Medication: ')
         .replace('Age Group_',           'Age Group: ')
        for n in feat_names
    ]

    sorted_idx = np.argsort(importances)
    colors     = ['#e74c3c' if importances[i] == max(importances)
                  else '#3498db' for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(sorted_idx)),
            importances[sorted_idx],
            color=colors, edgecolor='white')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([short_names[i] for i in sorted_idx], fontsize=9)
    ax.set_title(
        'Feature Importance — Random Forest\n(most important feature in red)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Importance Score')
    ax.grid(True, alpha=0.3, axis='x')
    save_chart(fig, '10_feature_importance.png')


def save_models(nn_model, rf_model, selected_features) -> None:
    """
    Save trained models and selected features to disk.

    Args:
        nn_model          : Trained Keras neural network
        rf_model          : Trained Random Forest
        selected_features : List of selected feature names
    """
    nn_model.save(MODEL_PATH)
    joblib.dump(rf_model,          RF_MODEL_PATH)
    joblib.dump(selected_features, FEATURES_PATH)
    log.info(f"Neural Network saved  → {MODEL_PATH}")
    log.info(f"Random Forest saved   → {RF_MODEL_PATH}")
    log.info(f"Selected features     → {FEATURES_PATH}")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    """Run the full model training, evaluation, and chart generation pipeline."""
    log.info("=" * 55)
    log.info("  MODEL TRAINING STARTED")
    log.info("=" * 55)

    # Load and prepare
    X, y, selected_features = load_and_prepare(PROCESSED_DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train all models
    baseline_results       = train_baseline_models(X_train, X_test, y_train, y_test)
    nn_model               = build_neural_network(X_train.shape[1])
    history, nn_result     = train_neural_network(
                                 nn_model, X_train, X_val,
                                 y_train, y_val, X_test, y_test)
    all_results            = {**baseline_results, 'Neural Network': nn_result}
    best_name              = print_evaluation(all_results, y)

    # Save 10 individual charts into img/ folder
    log.info(f"Saving 10 charts to '{IMG_DIR}/' folder...")
    chart_01_loss(history)
    chart_02_mae(history)
    chart_03_split(X_train, X_val, X_test)
    chart_04_r2(all_results, best_name)
    chart_05_mae_compare(all_results, best_name)
    chart_06_nn_scatter(all_results, y_test)
    chart_07_rf_scatter(all_results, y_test)
    chart_08_distribution(y)
    chart_09_table(all_results, best_name)
    chart_10_feature_importance(all_results)
    log.info(f"All 10 charts saved to '{IMG_DIR}/' folder")

    # Save models
    save_models(nn_model, baseline_results['Random Forest']['model'], selected_features)

    log.info("=" * 55)
    log.info("  MODEL TRAINING COMPLETE")
    log.info("=" * 55)


if __name__ == '__main__':
    main()