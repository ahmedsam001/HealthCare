"""
test_features.py
================
Unit tests for realtime/feature_engine.py logic — specifically the parts
that can be tested without a live Snowflake connection (pure computation).

Tests that do require Snowflake are marked with @pytest.mark.integration
and are skipped unless SNOWFLAKE_PASSWORD is set.

Run (unit tests only):
    pytest tests/test_features.py -v -m "not integration"

Run (all tests — needs SNOWFLAKE_PASSWORD):
    pytest tests/test_features.py -v
"""
import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "realtime"))


# ---------------------------------------------------------------------------
# Pure-logic tests (no Snowflake)
# ---------------------------------------------------------------------------
class TestFeatureLogic:
    """Tests for computations that are pure Python, no DB required."""

    def test_age_calculation_typical(self):
        """Age calculation sanity: 30-year-old."""
        from datetime import date
        birthdate = date(1994, 1, 1)
        today = date(2026, 9, 1)
        age_years = (today - birthdate).days / 365.25
        assert 32 <= age_years <= 33

    def test_age_calculation_very_old(self):
        """Age calculation sanity: 90-year-old."""
        from datetime import date
        birthdate = date(1936, 1, 1)
        today = date(2026, 9, 1)
        age_years = (today - birthdate).days / 365.25
        assert 90 <= age_years <= 91

    def test_age_safety_rule_applied(self):
        """ml_contract age safety: if AGE_AT_INDEX > 200, divide by 365.25."""
        import pandas as pd
        from ml_contract import apply_age_safety
        df = pd.DataFrame({"AGE_AT_INDEX": [12000.0, 45.0, 365.25 * 2]})
        result = apply_age_safety(df)
        assert result["AGE_AT_INDEX"].iloc[0] == pytest.approx(12000 / 365.25, rel=1e-4)
        assert result["AGE_AT_INDEX"].iloc[1] == 45.0

    def test_age_safety_rule_not_applied_for_valid(self):
        """Values under 200 are not divided."""
        import pandas as pd
        from ml_contract import apply_age_safety
        df = pd.DataFrame({"AGE_AT_INDEX": [35.5, 0.0, np.nan]})
        result = apply_age_safety(df)
        assert result["AGE_AT_INDEX"].iloc[0] == 35.5
        assert result["AGE_AT_INDEX"].iloc[1] == 0.0
        assert math.isnan(result["AGE_AT_INDEX"].iloc[2])


# ---------------------------------------------------------------------------
# Feature contract tests
# ---------------------------------------------------------------------------
class TestFeatureContract:
    def test_all_feature_cols_defined(self):
        from ml_contract import ALL_FEATURE_COLS, NUMERIC_COLS_FULL, CATEGORICAL_COLS
        assert len(ALL_FEATURE_COLS) == len(NUMERIC_COLS_FULL) + len(CATEGORICAL_COLS)

    def test_required_numeric_features_present(self):
        from ml_contract import NUMERIC_COLS_FULL
        required = [
            "AGE_AT_INDEX", "AVG_HEIGHT", "AVG_BMI", "AVG_WEIGHT",
            "AVG_DIASTOLIC_BP", "AVG_GLUCOSE", "AVG_SYSTOLIC_BP",
            "AVG_CHOLESTEROL", "OBSERVATION_COUNT",
            "DISTINCT_CONDITION_COUNT", "DISTINCT_MEDICATION_COUNT",
            "TOTAL_ENCOUNTERS", "HAS_GLUCOSE_READING", "HAS_CHOLESTEROL_READING",
        ]
        for col in required:
            assert col in NUMERIC_COLS_FULL, f"Missing from NUMERIC_COLS_FULL: {col}"

    def test_required_categorical_features_present(self):
        from ml_contract import CATEGORICAL_COLS
        required = ["GENDER", "RACE", "ETHNICITY", "MARITAL"]
        for col in required:
            assert col in CATEGORICAL_COLS, f"Missing from CATEGORICAL_COLS: {col}"

    def test_targets_defined(self):
        from ml_contract import TARGETS
        assert "Diabetes" in TARGETS
        assert "Hypertension" in TARGETS
        assert "Coronary Heart Disease" in TARGETS
        assert "Stroke" in TARGETS
        assert "Asthma" in TARGETS
        assert len(TARGETS) == 5

    def test_build_feature_row_all_missing(self):
        """build_feature_row with empty dict should produce NaN for numeric cols."""
        import pandas as pd
        from ml_contract import build_feature_row, NUMERIC_COLS_FULL, CATEGORICAL_COLS
        row = build_feature_row({})
        for col in NUMERIC_COLS_FULL:
            assert math.isnan(row[col].iloc[0]), f"{col} should be NaN"
        for col in CATEGORICAL_COLS:
            assert row[col].iloc[0] == "unknown"

    def test_build_feature_row_with_values(self):
        """build_feature_row correctly picks up provided values."""
        from ml_contract import build_feature_row
        features = {
            "AGE_AT_INDEX": 55.0,
            "AVG_GLUCOSE":  110.0,
            "GENDER":       "M",
            "RACE":         "white",
        }
        row = build_feature_row(features)
        assert row["AGE_AT_INDEX"].iloc[0] == 55.0
        assert row["AVG_GLUCOSE"].iloc[0]  == 110.0
        assert row["GENDER"].iloc[0]       == "M"
        assert row["RACE"].iloc[0]         == "white"

    def test_build_feature_row_age_safety_applied(self):
        """Age > 200 in feature dict should be corrected by build_feature_row."""
        from ml_contract import build_feature_row
        features = {"AGE_AT_INDEX": 20000.0}  # stored as days
        row = build_feature_row(features)
        assert row["AGE_AT_INDEX"].iloc[0] == pytest.approx(20000 / 365.25, rel=1e-4)


# ---------------------------------------------------------------------------
# Integration tests — require live Snowflake
# ---------------------------------------------------------------------------
needs_snowflake = pytest.mark.skipif(
    not os.getenv("SNOWFLAKE_PASSWORD"),
    reason="SNOWFLAKE_PASSWORD not set — skipping Snowflake integration tests",
)


@needs_snowflake
class TestFeatureIntegration:
    @pytest.fixture(scope="class")
    def real_patient_id(self):
        from patient_state import search_patient_by_name
        results = search_patient_by_name("Smith")
        if results.empty:
            pytest.skip("No patients named 'Smith' found in DB")
        return results.iloc[0]["PATIENT_ID"]

    def test_compute_features_returns_all_cols(self, real_patient_id):
        from feature_engine import compute_features
        from ml_contract import ALL_FEATURE_COLS
        feats = compute_features(real_patient_id)
        for col in ALL_FEATURE_COLS:
            assert col in feats, f"Feature missing: {col}"

    def test_age_at_index_reasonable(self, real_patient_id):
        from feature_engine import compute_features
        feats = compute_features(real_patient_id)
        age = feats.get("AGE_AT_INDEX")
        if age is not None and not math.isnan(age):
            assert 0 < age < 130, f"Implausible age: {age}"

    def test_has_flags_are_binary(self, real_patient_id):
        from feature_engine import compute_features
        feats = compute_features(real_patient_id)
        assert feats["HAS_GLUCOSE_READING"] in {0.0, 1.0}
        assert feats["HAS_CHOLESTEROL_READING"] in {0.0, 1.0}

    def test_counts_are_non_negative(self, real_patient_id):
        from feature_engine import compute_features
        feats = compute_features(real_patient_id)
        assert feats["TOTAL_ENCOUNTERS"] >= 0
        assert feats["OBSERVATION_COUNT"] >= 0
        assert feats["DISTINCT_CONDITION_COUNT"] >= 0
        assert feats["DISTINCT_MEDICATION_COUNT"] >= 0

