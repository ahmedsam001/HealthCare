"""
# STEP 10: Run the existing trained ML model on the updated features.
#
# This module is the inference entry point:
#   1. Calls feature_engine.compute_features(patient_id) to get the
#      latest features (GOLD + REALTIME, no model retraining).
#   2. Calls ml_contract.run_all_targets(feature_dict) to score the
#      patient against all 5 XGBoost pipelines.
#   3. Returns a structured result dict for display in Streamlit.
#
# The existing saved models are loaded once and cached in ml_contract.py.
# DO NOT retrain here.
"""

from feature_engine import compute_features
from ml_contract import TARGETS, run_all_targets

# ---------------------------------------------------------------------------
# Risk labels for UI display
# ---------------------------------------------------------------------------

def _risk_label(prob):
    if prob is None:
        return "Model Unavailable"
    if prob >= 0.7:
        return "🔴 High Risk"
    if prob >= 0.4:
        return "🟡 Moderate Risk"
    return "🟢 Low Risk"


def _risk_color(prob):
    if prob is None:
        return "gray"
    if prob >= 0.7:
        return "red"
    if prob >= 0.4:
        return "orange"
    return "green"


# ---------------------------------------------------------------------------
# STEP 10: Main inference function
# ---------------------------------------------------------------------------

def compute_risk(patient_id):
    """
    Compute updated risk scores for all 5 targets for a given patient.
    Returns a dictionary of risk information.
    """
    # Step 1 — Updated features
    features = compute_features(patient_id)

    # Step 2 — Run all 5 models (uses cached pipelines)
    raw_scores = run_all_targets(features)

    # Step 3 — Annotate
    scores = {}
    models_found   = []
    models_missing = []
    
    for target, prob in raw_scores.items():
        scores[target] = {
            "probability": prob,
            "label":       _risk_label(prob),
            "color":       _risk_color(prob),
            "pct":         round(prob * 100, 1) if prob is not None else None,
        }
        if prob is not None:
            models_found.append(target)
        else:
            models_missing.append(target)

    return {
        "patient_id":     patient_id,
        "features":       features,
        "scores":         scores,
        "models_found":   models_found,
        "models_missing": models_missing,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("STEP 10 SELF-TEST — Inference")
    from patient_state import search_patient_by_name

    results = search_patient_by_name("Smith")
    if results.empty:
        print("No patients found — cannot run inference self-test.")
    else:
        pid = results.iloc[0]["PATIENT_ID"]
        print(f"Running inference for patient: {pid}")
        risk = compute_risk(pid)
        print(f"\nFeatures used:")
        for k, v in risk["features"].items():
            print(f"  {k:<35}: {v}")
        print(f"\nRisk Scores:")
        for target, info in risk["scores"].items():
            pct = info["pct"]
            label = info["label"]
            pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
            print(f"  {target:<30}: {pct_str:>8}  {label}")
        if risk["models_missing"]:
            print(f"\n⚠ Missing model files for: {risk['models_missing']}")
    print("\nSTEP 10 DONE")

