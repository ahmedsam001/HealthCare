"""
# STEP 9: Recalculate the ML features after the new event is added.
#
# This module replicates the feature logic from ML.PATIENT_FEATURES_TARGETS_LONG
# (inferred from column names and the ML contract in ML_BD1.ipynb).
#
# Features are computed by reading:
#   GOLD.PATIENTS_GOLD         — age, demographics
#   GOLD.OBSERVATIONS_GOLD     — vitals averages (pivoted table)
#   REALTIME.RT_OBSERVATIONS   — new vitals (long/unpivoted)
#   GOLD.CONDITIONS_GOLD       — distinct condition count
#   REALTIME.RT_CONDITIONS     — new conditions
#   GOLD.MEDICATIONS_GOLD      — distinct medication count
#   REALTIME.RT_MEDICATIONS    — new medications
#   GOLD.ENCOUNTERS_GOLD       — total encounter count
#   REALTIME.RT_ENCOUNTERS     — new encounters
#
# Temporal rule: INDEX_DATE for live inference = NOW()
# (we are predicting risk at this moment, using all available data)
#
# The output dict has exactly the keys in ml_contract.ALL_FEATURE_COLS.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import snowflake.connector

from config import snowflake_connect_kwargs, SCHEMA_GOLD
from ml_contract import ALL_FEATURE_COLS, NUMERIC_COLS_FULL, CATEGORICAL_COLS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_conn() -> snowflake.connector.SnowflakeConnection:
    return snowflake.connector.connect(**snowflake_connect_kwargs())


def _scalar(conn, sql: str, params=()) -> Optional[float]:
    """Execute SQL and return the first column of the first row (or None)."""
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    cur.close()
    return row[0] if row and row[0] is not None else None


def _int_scalar(conn, sql: str, params=()) -> int:
    val = _scalar(conn, sql, params)
    return int(val) if val is not None else 0


# ---------------------------------------------------------------------------
# Measurement description → Gold column mapping
# (matches TARGET_MEASUREMENTS in 03_Gold.ipynb)
# ---------------------------------------------------------------------------

MEASUREMENT_TO_GOLD_COL = {
    "Body Height":             '"Body Height"',
    "Body Mass Index":         '"Body Mass Index"',
    "Body Weight":             '"Body Weight"',
    "Diastolic Blood Pressure": '"Diastolic Blood Pressure"',
    "Glucose":                 '"GLUCOSE"',
    "Systolic Blood Pressure": '"Systolic Blood Pressure"',
    "Total Cholesterol":       '"Total Cholesterol"',
}

# Map from feature name to the measurement description used in RT_OBSERVATIONS
FEATURE_TO_MEASUREMENT = {
    "AVG_HEIGHT":         "Body Height",
    "AVG_BMI":            "Body Mass Index",
    "AVG_WEIGHT":         "Body Weight",
    "AVG_DIASTOLIC_BP":   "Diastolic Blood Pressure",
    "AVG_GLUCOSE":        "Glucose",
    "AVG_SYSTOLIC_BP":    "Systolic Blood Pressure",
    "AVG_CHOLESTEROL":    "Total Cholesterol",
}


def _avg_from_gold_and_rt(
    conn, patient_id: str, feature_name: str
) -> Optional[float]:
    """
    Average a vital measurement across both OBSERVATIONS_GOLD (pivoted)
    and RT_OBSERVATIONS (long).

    OBSERVATIONS_GOLD stores values as VARCHAR (may be "Not Recorded").
    Cast safely with TRY_CAST.
    """
    measurement = FEATURE_TO_MEASUREMENT[feature_name]
    gold_col    = MEASUREMENT_TO_GOLD_COL[measurement]

    # Historical average from pivoted Gold table
    sql_hist = f"""
        SELECT AVG(TRY_CAST({gold_col} AS FLOAT))
        FROM {SCHEMA_GOLD}.OBSERVATIONS_GOLD
        WHERE PATIENT_ID = %s
          AND {gold_col} != 'Not Recorded'
    """

    # Real-time average from long RT table
    sql_rt = """
        SELECT AVG(VALUE_NUMERIC)
        FROM REALTIME.RT_OBSERVATIONS
        WHERE PATIENT_ID = %s
          AND UPPER(DESCRIPTION) = UPPER(%s)
          AND VALUE_NUMERIC IS NOT NULL
    """

    hist_avg = _scalar(conn, sql_hist, (patient_id,))
    rt_avg   = _scalar(conn, sql_rt,   (patient_id, measurement))

    # Combine: weighted average if both sources have data
    vals = [v for v in [hist_avg, rt_avg] if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _has_reading_flag(conn, patient_id: str, measurement: str) -> int:
    """Return 1 if ANY numeric reading exists for this measurement, else 0."""
    gold_col = MEASUREMENT_TO_GOLD_COL.get(measurement)

    hist_count = 0
    if gold_col:
        sql = f"""
            SELECT COUNT(*) FROM {SCHEMA_GOLD}.OBSERVATIONS_GOLD
            WHERE PATIENT_ID = %s
              AND {gold_col} != 'Not Recorded'
              AND TRY_CAST({gold_col} AS FLOAT) IS NOT NULL
        """
        hist_count = _int_scalar(conn, sql, (patient_id,))

    rt_count = _int_scalar(conn, """
        SELECT COUNT(*) FROM REALTIME.RT_OBSERVATIONS
        WHERE PATIENT_ID = %s
          AND UPPER(DESCRIPTION) = UPPER(%s)
          AND VALUE_NUMERIC IS NOT NULL
    """, (patient_id, measurement))

    return 1 if (hist_count + rt_count) > 0 else 0


# ---------------------------------------------------------------------------
# STEP 9: Main feature computation
# ---------------------------------------------------------------------------

def compute_features(patient_id: str) -> dict:
    """
    Compute the complete feature dictionary for a patient.
    Returns a dict with all keys in ml_contract.ALL_FEATURE_COLS.
    Missing values are np.nan for numeric or 'unknown' for categorical.
    """
    conn = _get_conn()
    try:
        # ----- Demographics from PATIENTS_GOLD ---------------------------------
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT BIRTHDATE, GENDER, RACE, ETHNICITY, MARITAL
            FROM {SCHEMA_GOLD}.PATIENTS_GOLD
            WHERE PATIENT_ID = %s
            LIMIT 1
            """,
            (patient_id,),
        )
        pat_row = cur.fetchone()
        cur.close()

        if pat_row is None:
            raise ValueError(f"Patient '{patient_id}' not found in PATIENTS_GOLD.")

        birthdate, gender, race, ethnicity, marital = pat_row

        # AGE_AT_INDEX = age in years at NOW() (live prediction)
        today = date.today()
        if isinstance(birthdate, (date, datetime)):
            age_years = (today - (birthdate.date() if isinstance(birthdate, datetime)
                                  else birthdate)).days / 365.25
        else:
            age_years = np.nan

        # ----- Vital averages from Gold + RT -----------------------------------
        avg_height       = _avg_from_gold_and_rt(conn, patient_id, "AVG_HEIGHT")
        avg_bmi          = _avg_from_gold_and_rt(conn, patient_id, "AVG_BMI")
        avg_weight       = _avg_from_gold_and_rt(conn, patient_id, "AVG_WEIGHT")
        avg_diastolic_bp = _avg_from_gold_and_rt(conn, patient_id, "AVG_DIASTOLIC_BP")
        avg_glucose      = _avg_from_gold_and_rt(conn, patient_id, "AVG_GLUCOSE")
        avg_systolic_bp  = _avg_from_gold_and_rt(conn, patient_id, "AVG_SYSTOLIC_BP")
        avg_cholesterol  = _avg_from_gold_and_rt(conn, patient_id, "AVG_CHOLESTEROL")

        # ----- Flags -----------------------------------------------------------
        has_glucose      = _has_reading_flag(conn, patient_id, "Glucose")
        has_cholesterol  = _has_reading_flag(conn, patient_id, "Total Cholesterol")

        # ----- Count features --------------------------------------------------
        obs_count = _int_scalar(conn, f"""
            SELECT COUNT(*) FROM (
                SELECT ENCOUNTER_ID FROM {SCHEMA_GOLD}.OBSERVATIONS_GOLD
                WHERE PATIENT_ID = %s
                UNION ALL
                SELECT ENCOUNTER_ID FROM REALTIME.RT_OBSERVATIONS
                WHERE PATIENT_ID = %s
            )
        """, (patient_id, patient_id))

        distinct_conditions = _int_scalar(conn, f"""
            SELECT COUNT(DISTINCT CODE) FROM (
                SELECT CODE FROM {SCHEMA_GOLD}.CONDITIONS_GOLD WHERE PATIENT_ID = %s
                UNION ALL
                SELECT CODE FROM REALTIME.RT_CONDITIONS WHERE PATIENT_ID = %s
            )
        """, (patient_id, patient_id))

        distinct_medications = _int_scalar(conn, f"""
            SELECT COUNT(DISTINCT CODE) FROM (
                SELECT CODE FROM {SCHEMA_GOLD}.MEDICATIONS_GOLD WHERE PATIENT_ID = %s
                UNION ALL
                SELECT CODE FROM REALTIME.RT_MEDICATIONS WHERE PATIENT_ID = %s
            )
        """, (patient_id, patient_id))

        total_encounters = _int_scalar(conn, f"""
            SELECT COUNT(*) FROM (
                SELECT ENCOUNTER_ID FROM {SCHEMA_GOLD}.ENCOUNTERS_GOLD WHERE PATIENT_ID = %s
                UNION ALL
                SELECT ENCOUNTER_ID FROM REALTIME.RT_ENCOUNTERS WHERE PATIENT_ID = %s
            )
        """, (patient_id, patient_id))

    finally:
        conn.close()

    # ----- Assemble feature dict -------------------------------------------
    features: dict = {
        # Numeric
        "AGE_AT_INDEX":              age_years,
        "AVG_HEIGHT":                avg_height      if avg_height      is not None else np.nan,
        "AVG_BMI":                   avg_bmi         if avg_bmi         is not None else np.nan,
        "AVG_WEIGHT":                avg_weight      if avg_weight      is not None else np.nan,
        "AVG_DIASTOLIC_BP":          avg_diastolic_bp if avg_diastolic_bp is not None else np.nan,
        "AVG_GLUCOSE":               avg_glucose     if avg_glucose     is not None else np.nan,
        "AVG_SYSTOLIC_BP":           avg_systolic_bp if avg_systolic_bp is not None else np.nan,
        "AVG_CHOLESTEROL":           avg_cholesterol if avg_cholesterol is not None else np.nan,
        "OBSERVATION_COUNT":         float(obs_count),
        "DISTINCT_CONDITION_COUNT":  float(distinct_conditions),
        "DISTINCT_MEDICATION_COUNT": float(distinct_medications),
        "TOTAL_ENCOUNTERS":          float(total_encounters),
        "HAS_GLUCOSE_READING":       float(has_glucose),
        "HAS_CHOLESTEROL_READING":   float(has_cholesterol),
        # Categorical
        "GENDER":    gender    or "unknown",
        "RACE":      race      or "unknown",
        "ETHNICITY": ethnicity or "unknown",
        "MARITAL":   marital   or "UNKNOWN",
    }

    return features


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("STEP 9 SELF-TEST — Feature Engine")

    from patient_state import search_patient_by_name
    results = search_patient_by_name("Smith")
    if results.empty:
        print("No patients found — cannot run self-test.")
    else:
        pid = results.iloc[0]["PATIENT_ID"]
        print(f"Computing features for patient: {pid}")
        feats = compute_features(pid)
        for k, v in feats.items():
            print(f"  {k:<35}: {v}")
    print("STEP 9 DONE")

