"""
test_inference.py
=================
Unit and integration tests for realtime/inference.py and realtime/ml_contract.py.

Run (unit tests only — no Snowflake, no model files):
    pytest tests/test_inference.py -v -m "not integration and not requires_models"

Run (with model files, no Snowflake):
    pytest tests/test_inference.py -v -m "requires_models and not integration"

Run (full):
    pytest tests/test_inference.py -v
"""
import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "realtime"))


# ---------------------------------------------------------------------------
# Model availability checks
# ---------------------------------------------------------------------------
def _models_found() -> list[str]:
    try:
        from ml_contract import models_available
        return [t for t, ok in models_available().items() if ok]
    except Exception:
        return []


MODELS_AVAILABLE = len(_models_found()) > 0

requires_models = pytest.mark.skipif(
    not MODELS_AVAILABLE,
    reason="No model .joblib files found — skipping model-dependent tests",
)
needs_snowflake = pytest.mark.skipif(
    not os.getenv("SNOWFLAKE_PASSWORD"),
    reason="SNOWFLAKE_PASSWORD not set",
)


# ---------------------------------------------------------------------------
# Tests: risk_label / risk_color helpers
# ---------------------------------------------------------------------------
class TestRiskHelpers:
    def _import(self):
        from inference import _risk_label, _risk_color
        return _risk_label, _risk_color

    def test_high_risk_label(self):
        _risk_label, _ = self._import()
        assert "High" in _risk_label(0.75)

    def test_moderate_risk_label(self):
        _risk_label, _ = self._import()
        assert "Moderate" in _risk_label(0.55)

    def test_low_risk_label(self):
        _risk_label, _ = self._import()
        assert "Low" in _risk_label(0.20)

    def test_none_returns_unavailable(self):
        _risk_label, _ = self._import()
        label = _risk_label(None)
        assert "Unavailable" in label or "unavailable" in label.lower()

    def test_high_risk_color(self):
        _, _risk_color = self._import()
        assert _risk_color(0.8) == "red"

    def test_moderate_risk_color(self):
        _, _risk_color = self._import()
        assert _risk_color(0.5) == "orange"

    def test_low_risk_color(self):
        _, _risk_color = self._import()
        assert _risk_color(0.1) == "green"


# ---------------------------------------------------------------------------
# Tests: run_inference with missing model (should return None, not raise)
# ---------------------------------------------------------------------------
class TestInferenceMissingModel:
    def test_missing_model_returns_none(self, tmp_path, monkeypatch):
        """When a model file is missing, run_inference returns None gracefully."""
        from config import MODEL_DIR
        import ml_contract
        # Point model cache to an empty dir so no model loads
        monkeypatch.setattr(ml_contract, "_model_cache", {})
        import config as cfg
        monkeypatch.setattr(cfg, "MODEL_DIR", tmp_path)
        from ml_contract import run_inference
        result = run_inference("Diabetes", {"AGE_AT_INDEX": 55.0})
        assert result is None

    def test_run_all_targets_missing_returns_dict_of_none(self, tmp_path, monkeypatch):
        import ml_contract
        monkeypatch.setattr(ml_contract, "_model_cache", {})
        import config as cfg
        monkeypatch.setattr(cfg, "MODEL_DIR", tmp_path)
        from ml_contract import run_all_targets, TARGETS
        results = run_all_targets({})
        assert set(results.keys()) == set(TARGETS)
        assert all(v is None for v in results.values())


# ---------------------------------------------------------------------------
# Tests: with real model files
# ---------------------------------------------------------------------------
@requires_models
class TestInferenceWithModels:
    """Requires .joblib files in saved_models/."""

    FULL_FEATURE_DICT = {
        "AGE_AT_INDEX":              55.0,
        "AVG_HEIGHT":                170.0,
        "AVG_BMI":                   26.0,
        "AVG_WEIGHT":                75.0,
        "AVG_DIASTOLIC_BP":          80.0,
        "AVG_GLUCOSE":               95.0,
        "AVG_SYSTOLIC_BP":           120.0,
        "AVG_CHOLESTEROL":           180.0,
        "OBSERVATION_COUNT":         20.0,
        "DISTINCT_CONDITION_COUNT":   3.0,
        "DISTINCT_MEDICATION_COUNT":  2.0,
        "TOTAL_ENCOUNTERS":          15.0,
        "HAS_GLUCOSE_READING":        1.0,
        "HAS_CHOLESTEROL_READING":    1.0,
        "GENDER":                    "M",
        "RACE":                      "white",
        "ETHNICITY":                 "nonhispanic",
        "MARITAL":                   "M",
    }

    def test_run_inference_returns_float(self):
        from ml_contract import run_inference, TARGETS
        target = _models_found()[0]
        prob = run_inference(target, self.FULL_FEATURE_DICT)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_run_all_targets_returns_all(self):
        from ml_contract import run_all_targets, TARGETS
        results = run_all_targets(self.FULL_FEATURE_DICT)
        assert set(results.keys()) == set(TARGETS)

    def test_probabilities_in_range(self):
        from ml_contract import run_all_targets
        results = run_all_targets(self.FULL_FEATURE_DICT)
        for target, prob in results.items():
            if prob is not None:
                assert 0.0 <= prob <= 1.0, f"{target}: probability {prob} out of [0,1]"

    def test_model_handles_nan_features(self):
        """Model pipeline should handle NaN numeric inputs (imputer fills them)."""
        from ml_contract import run_inference
        sparse_features = {
            "AGE_AT_INDEX": 60.0,
            "GENDER": "F",
            "RACE": "white",
            "ETHNICITY": "nonhispanic",
            "MARITAL": "S",
            # all numeric vitals missing → NaN → imputed
        }
        target = _models_found()[0]
        prob = run_inference(target, sparse_features)
        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_compute_risk_result_structure(self):
        """compute_risk (using mocked features) returns the expected dict structure."""
        from ml_contract import run_all_targets, TARGETS
        from inference import _risk_label, _risk_color
        # Simulate what compute_risk does without hitting Snowflake
        raw = run_all_targets(self.FULL_FEATURE_DICT)
        for target, prob in raw.items():
            if prob is not None:
                assert 0.0 <= prob <= 1.0
                assert _risk_label(prob) != ""


# ---------------------------------------------------------------------------
# Integration tests: full compute_risk with real patient (needs Snowflake + models)
# ---------------------------------------------------------------------------
@needs_snowflake
@requires_models
class TestComputeRiskIntegration:
    @pytest.fixture(scope="class")
    def real_patient_id(self):
        from patient_state import search_patient_by_name
        results = search_patient_by_name("Smith")
        if results.empty:
            pytest.skip("No patients named 'Smith' in DB")
        return results.iloc[0]["PATIENT_ID"]

    def test_compute_risk_returns_all_targets(self, real_patient_id):
        from inference import compute_risk
        from ml_contract import TARGETS
        result = compute_risk(real_patient_id)
        assert result["patient_id"] == real_patient_id
        for target in TARGETS:
            assert target in result["scores"]

    def test_compute_risk_scores_in_range(self, real_patient_id):
        from inference import compute_risk
        result = compute_risk(real_patient_id)
        for target, info in result["scores"].items():
            prob = info.get("probability")
            if prob is not None:
                assert 0.0 <= prob <= 1.0, f"{target}: {prob}"

    def test_compute_risk_features_key_present(self, real_patient_id):
        from inference import compute_risk
        result = compute_risk(real_patient_id)
        assert "features" in result
        assert "AGE_AT_INDEX" in result["features"]

