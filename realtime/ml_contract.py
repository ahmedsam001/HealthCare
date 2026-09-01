"""
# STEP 1: Define the exact ML input, features, targets, preprocessing,
# and inference contract.
#
# This module is the single source of truth for:
#   - feature column names (must match ML.PATIENT_FEATURES_TARGETS_LONG)
#   - preprocessing steps (matches ML_BD1.ipynb: SimpleImputer + OneHotEncoder)
#   - model loading (reads the .joblib pipelines saved by ML_BD1.ipynb)
#   - inference API used by inference.py
#
# DO NOT retrain or alter the preprocessing here.
# DO NOT invent feature names — they are extracted from ML_BD1.ipynb.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config import TARGETS, MODEL_DIR, model_path


# ---------------------------------------------------------------------------
# Feature contract — exact column names from ML_BD1.ipynb
# ---------------------------------------------------------------------------

CATEGORICAL_COLS: list[str] = ["GENDER", "RACE", "ETHNICITY", "MARITAL"]

# Full numeric feature set (includes count features used in the saved models)
NUMERIC_COLS_FULL: list[str] = [
    "AGE_AT_INDEX",
    "AVG_HEIGHT",
    "AVG_BMI",
    "AVG_WEIGHT",
    "AVG_DIASTOLIC_BP",
    "AVG_GLUCOSE",
    "AVG_SYSTOLIC_BP",
    "AVG_CHOLESTEROL",
    "OBSERVATION_COUNT",
    "DISTINCT_CONDITION_COUNT",
    "DISTINCT_MEDICATION_COUNT",
    "TOTAL_ENCOUNTERS",
    "HAS_GLUCOSE_READING",
    "HAS_CHOLESTEROL_READING",
]

# Ablated set (leaky count features removed) — used for documentation / comparison
NUMERIC_COLS_ABLATED: list[str] = [
    "AGE_AT_INDEX",
    "AVG_HEIGHT",
    "AVG_BMI",
    "AVG_WEIGHT",
    "AVG_DIASTOLIC_BP",
    "AVG_GLUCOSE",
    "AVG_SYSTOLIC_BP",
    "AVG_CHOLESTEROL",
    "HAS_GLUCOSE_READING",
    "HAS_CHOLESTEROL_READING",
]

# All feature columns in the order the model expects them
ALL_FEATURE_COLS: list[str] = NUMERIC_COLS_FULL + CATEGORICAL_COLS

# Metadata columns present in ML.PATIENT_FEATURES_TARGETS_LONG
# (dropped before constructing X)
META_COLS: list[str] = ["PATIENT_ID", "TARGET_NAME", "LABEL", "INDEX_DATE"]


# ---------------------------------------------------------------------------
# Age safety rule (copied from ML_BD1.ipynb load_data function)
# AGE_AT_INDEX stored as days in some rows → convert to years if > 200
# ---------------------------------------------------------------------------

def apply_age_safety(df: pd.DataFrame) -> pd.DataFrame:
    """If AGE_AT_INDEX > 200, divide by 365.25 (day→year conversion)."""
    if "AGE_AT_INDEX" in df.columns:
        df = df.copy()
        df["AGE_AT_INDEX"] = df["AGE_AT_INDEX"].apply(
            lambda x: x / 365.25 if pd.notnull(x) and x > 200 else x
        )
    return df


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def load_model(target_name: str):
    """
    Load (and cache) the sklearn Pipeline saved by ML_BD1.ipynb for the
    given target.  Returns the fitted pipeline object.

    Raises FileNotFoundError if the .joblib file is not found.
    """
    if target_name in _model_cache:
        return _model_cache[target_name]

    path = model_path(target_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Copy saved_models/ from the ML training environment or run "
            "Models/Ml/ML_BD1.ipynb first."
        )

    pipeline = joblib.load(path)
    _model_cache[target_name] = pipeline
    return pipeline


def models_available() -> dict[str, bool]:
    """Return {target: True/False} showing which model files exist."""
    return {t: model_path(t).exists() for t in TARGETS}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_feature_row(feature_dict: dict) -> pd.DataFrame:
    """
    Build a one-row DataFrame with exactly the columns the model expects.
    Missing numeric features are filled with NaN (the imputer handles them).
    Missing categorical features are filled with 'unknown'.
    """
    row: dict = {}
    for col in NUMERIC_COLS_FULL:
        row[col] = feature_dict.get(col, np.nan)
    for col in CATEGORICAL_COLS:
        row[col] = feature_dict.get(col, "unknown")
    return apply_age_safety(pd.DataFrame([row]))


def run_inference(target_name: str, feature_dict: dict) -> Optional[float]:
    """
    Run the saved sklearn pipeline for *target_name* on the supplied
    feature dictionary.  Returns the probability of the positive class
    (0.0 – 1.0), or None if the model file is unavailable.

    Parameters
    ----------
    target_name  : one of TARGETS
    feature_dict : dict mapping feature column names to values
    """
    try:
        pipeline = load_model(target_name)
    except FileNotFoundError as exc:
        warnings.warn(str(exc))
        return None

    X = build_feature_row(feature_dict)
    prob = pipeline.predict_proba(X)[0, 1]
    return float(prob)


def run_all_targets(feature_dict: dict) -> dict[str, Optional[float]]:
    """
    Run inference for all 5 targets.
    Returns {target_name: probability_or_None}.
    """
    return {t: run_inference(t, feature_dict) for t in TARGETS}


# ---------------------------------------------------------------------------
# Self-test (run via: python ml_contract.py)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """
    Load one real patient row from Snowflake and run inference for all targets.
    Prints the ML contract schema and the inference results.
    """
    import snowflake.connector
    from config import snowflake_connect_kwargs, SCHEMA_ML

    print("=" * 60)
    print("STEP 1 SELF-TEST — ML Contract")
    print("=" * 60)

    print("\n[1/4] Feature contract")
    print(f"  Numeric ({len(NUMERIC_COLS_FULL)})  : {NUMERIC_COLS_FULL}")
    print(f"  Categorical ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")
    print(f"  Targets ({len(TARGETS)})       : {TARGETS}")

    print("\n[2/4] Model files")
    for t, exists in models_available().items():
        status = "✓ found" if exists else "✗ MISSING"
        print(f"  {status}  {model_path(t)}")

    print("\n[3/4] Loading one patient row from Snowflake ...")
    conn = snowflake.connector.connect(**snowflake_connect_kwargs())
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT * FROM {SCHEMA_ML}.PATIENT_FEATURES_TARGETS_LONG "
            f"WHERE TARGET_NAME = 'Diabetes' LIMIT 1"
        )
        cols = [d[0] for d in cur.description]
        row_values = cur.fetchone()
        if row_values is None:
            print("  No rows found in ML.PATIENT_FEATURES_TARGETS_LONG — skipping inference test.")
            return
        row = dict(zip(cols, row_values))
        print(f"  Patient ID : {row.get('PATIENT_ID')}")
        print(f"  INDEX_DATE : {row.get('INDEX_DATE')}")
        print(f"  LABEL      : {row.get('LABEL')}")
        cur.close()
    finally:
        conn.close()

    # Build feature dict (drop metadata columns)
    feature_dict = {k: v for k, v in row.items() if k not in META_COLS}

    print("\n[4/4] Running inference for all targets ...")
    results = run_all_targets(feature_dict)
    for target, prob in results.items():
        if prob is None:
            print(f"  {target:<30} : model file missing")
        else:
            print(f"  {target:<30} : {prob:.4f} ({prob * 100:.1f}% risk)")

    print("\nSTEP 1 DONE")


if __name__ == "__main__":
    _self_test()

