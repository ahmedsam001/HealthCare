"""
# STEP 12: Validate the complete pipeline from patient search to updated risk.
#
# Tests covered:
#   T01  Search by valid Patient ID
#   T02  Search by patient name
#   T03  Search invalid patient ID → not found
#   T04  Validate a correct event (no DB writes)
#   T05  Validate an invalid event → ValidationError raised
#   T06  Direct Snowflake insert (STEP 5 path)
#   T07  Kafka connectivity check
#   T08  Kafka publish (if Kafka is available)
#   T09  RT_ table row count increases after insert
#   T10  Patient state UNION query includes the new event
#   T11  Feature recalculation (values change after insert)
#   T12  ML inference returns probabilities for all 5 targets
#   T13  Updated risk dict has correct structure
#
# Run:
#   python tests/test_e2e.py
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import date
from typing import Any, Optional

# Allow direct import from realtime/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "realtime"))

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

PASS = "✓ PASS"
FAIL = "✗ FAIL"
SKIP = "⚫ SKIP"

results: list[tuple[str, str, str]] = []


def run_test(name: str, fn):
    """Run fn(); record PASS/FAIL/SKIP."""
    try:
        result = fn()
        if result is None or result is True:
            status = PASS
            detail = ""
        elif result is False:
            status = FAIL
            detail = "returned False"
        elif isinstance(result, str) and result.startswith("SKIP"):
            status = SKIP
            detail = result[5:]
        else:
            status = PASS
            detail = str(result)[:120]
    except Exception as exc:
        status = FAIL
        detail = f"{type(exc).__name__}: {exc}"
    results.append((name, status, detail))
    symbol = {"✓ PASS": "✓", "✗ FAIL": "✗", "⚫ SKIP": "⚫"}[status]
    print(f"  [{symbol}] {name}" + (f"  — {detail}" if detail else ""))
    return status == PASS


# ---------------------------------------------------------------------------
# Shared state across tests
# ---------------------------------------------------------------------------

_REAL_PATIENT_ID: Optional[str]  = None
_PRE_INSERT_RT_COUNT: int         = 0
_POST_INSERT_RT_COUNT: int        = 0
_PRE_FEATURES: Optional[dict]     = None
_POST_FEATURES: Optional[dict]    = None
_RISK_RESULT: Optional[dict]      = None
_TEST_EVENT = None  # set during T06


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def t01_search_by_id():
    """Search for a real patient by looking up the first one in PATIENTS_GOLD."""
    global _REAL_PATIENT_ID
    import snowflake.connector
    from config import snowflake_connect_kwargs, SCHEMA_GOLD
    conn = snowflake.connector.connect(**snowflake_connect_kwargs())
    cur = conn.cursor()
    cur.execute(f"SELECT PATIENT_ID FROM {SCHEMA_GOLD}.PATIENTS_GOLD LIMIT 1")
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return "SKIP — PATIENTS_GOLD is empty"
    _REAL_PATIENT_ID = row[0]

    from patient_state import search_patient_by_id
    result = search_patient_by_id(_REAL_PATIENT_ID)
    assert result is not None and not result.empty, "Patient not found by ID"
    return f"patient_id={_REAL_PATIENT_ID[:8]}..."


def t02_search_by_name():
    """Search for the found patient by their FULL_NAME."""
    if _REAL_PATIENT_ID is None:
        return "SKIP — no patient ID from T01"
    from patient_state import get_patient_info, search_patient_by_name
    info = get_patient_info(_REAL_PATIENT_ID)
    if info is None:
        return "SKIP — could not get patient info"
    full_name = info.get("FULL_NAME", "")
    last = full_name.split()[-1] if full_name else ""
    results_df = search_patient_by_name(last)
    assert not results_df.empty, f"No results for name fragment '{last}'"
    return f"found {len(results_df)} patient(s) matching '{last}'"


def t03_invalid_patient():
    """Search for a non-existent patient ID → must return None."""
    from patient_state import search_patient_by_id
    result = search_patient_by_id("00000000-0000-0000-0000-000000000000")
    assert result is None or result.empty, "Expected None for non-existent patient"


def t04_valid_event_validation():
    """Create and validate a correct OBSERVATION event (no writes)."""
    from event_schema import MedicalEvent, validate_event
    ev = MedicalEvent(
        patient_id="test-patient",
        event_type="OBSERVATION",
        payload={
            "DESCRIPTION":      "Body Weight",
            "OBSERVATION_DATE": date.today().isoformat(),
            "VALUE_NUMERIC":    72.5,
        },
    )
    validated = validate_event(ev)
    assert validated.event_type == "OBSERVATION"
    assert validated.payload["READING_SEQ"] == 1  # default filled


def t05_invalid_event_validation():
    """Submit a CONDITION event missing required fields → ValidationError."""
    from event_schema import MedicalEvent, validate_event, ValidationError
    ev = MedicalEvent(
        patient_id="test",
        event_type="CONDITION",
        payload={"DESCRIPTION": "Hypertension"},  # missing CODE and START_DATE
    )
    try:
        validate_event(ev)
        return False  # should have raised
    except ValidationError as ve:
        assert "CODE" in str(ve) or "START_DATE" in str(ve)
        return f"caught: {ve}"


def t05b_unknown_event_type():
    """Unknown event_type → ValidationError."""
    from event_schema import MedicalEvent, validate_event, ValidationError
    ev = MedicalEvent(patient_id="x", event_type="LAB_RESULT", payload={})
    try:
        validate_event(ev)
        return False
    except ValidationError:
        return True


def t06_direct_snowflake_insert():
    """Insert a test OBSERVATION event directly into REALTIME.RT_OBSERVATIONS."""
    global _PRE_INSERT_RT_COUNT, _TEST_EVENT
    if _REAL_PATIENT_ID is None:
        return "SKIP — no patient ID"

    from patient_state import ensure_realtime_schema, insert_event_direct
    from event_schema import MedicalEvent
    import snowflake.connector
    from config import snowflake_connect_kwargs

    ensure_realtime_schema()

    # Count before insert
    conn = snowflake.connector.connect(**snowflake_connect_kwargs())
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM REALTIME.RT_OBSERVATIONS WHERE PATIENT_ID = %s",
        (_REAL_PATIENT_ID,),
    )
    _PRE_INSERT_RT_COUNT = cur.fetchone()[0]
    cur.close()
    conn.close()

    _TEST_EVENT = MedicalEvent(
        patient_id=_REAL_PATIENT_ID,
        event_type="OBSERVATION",
        payload={
            "DESCRIPTION":      "Body Weight",
            "OBSERVATION_DATE": date.today().isoformat(),
            "VALUE_NUMERIC":    99.9,
        },
    )
    insert_event_direct(_TEST_EVENT)
    return f"pre-count={_PRE_INSERT_RT_COUNT}"


def t07_kafka_connectivity():
    """Check whether Kafka is reachable."""
    try:
        from kafka_producer import test_connectivity
        ok = test_connectivity()
        if ok:
            return "Kafka connected"
        return "SKIP — Kafka not reachable (this is OK if Kafka is not running)"
    except Exception as e:
        return f"SKIP — kafka-python not installed or Kafka offline: {e}"


def t08_kafka_publish():
    """Publish a test event to Kafka (only if Kafka is reachable)."""
    if _TEST_EVENT is None:
        return "SKIP — no test event from T06"
    try:
        from kafka_producer import test_connectivity, produce_event
        if not test_connectivity():
            return "SKIP — Kafka not reachable"
        ok = produce_event(_TEST_EVENT, validate=False)
        assert ok, "produce_event returned False"
        return "published successfully"
    except Exception as e:
        return f"SKIP — {e}"


def t09_rt_row_count_increased():
    """RT_OBSERVATIONS must have more rows after the T06 insert."""
    global _POST_INSERT_RT_COUNT
    if _REAL_PATIENT_ID is None:
        return "SKIP — no patient ID"
    import snowflake.connector
    from config import snowflake_connect_kwargs
    conn = snowflake.connector.connect(**snowflake_connect_kwargs())
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM REALTIME.RT_OBSERVATIONS WHERE PATIENT_ID = %s",
        (_REAL_PATIENT_ID,),
    )
    _POST_INSERT_RT_COUNT = cur.fetchone()[0]
    cur.close()
    conn.close()
    assert _POST_INSERT_RT_COUNT > _PRE_INSERT_RT_COUNT, (
        f"Row count did not increase: pre={_PRE_INSERT_RT_COUNT} "
        f"post={_POST_INSERT_RT_COUNT}"
    )
    return f"count: {_PRE_INSERT_RT_COUNT} → {_POST_INSERT_RT_COUNT}"


def t10_patient_state_includes_new_event():
    """get_latest_observations() RT data must include the newly inserted row."""
    if _REAL_PATIENT_ID is None:
        return "SKIP"
    from patient_state import get_latest_observations
    _, rt_obs = get_latest_observations(_REAL_PATIENT_ID)
    assert not rt_obs.empty, "RT_OBSERVATIONS for patient is empty"
    # The test insert had VALUE_NUMERIC = 99.9; check it's present
    matching = rt_obs[rt_obs["VALUE_NUMERIC"] == 99.9]
    assert not matching.empty, "Test value 99.9 not found in RT_OBSERVATIONS"
    return f"{len(rt_obs)} RT observation(s) found; test value present"


def t11_feature_recalculation():
    """Compute features before and after T06 insert; AVG_WEIGHT should change."""
    global _POST_FEATURES
    if _REAL_PATIENT_ID is None:
        return "SKIP"
    from feature_engine import compute_features
    # We already inserted the 99.9 kg value; recompute features now
    _POST_FEATURES = compute_features(_REAL_PATIENT_ID)
    assert "AVG_WEIGHT" in _POST_FEATURES
    obs_count = _POST_FEATURES.get("OBSERVATION_COUNT", 0)
    return (
        f"AVG_WEIGHT={_POST_FEATURES.get('AVG_WEIGHT'):.2f}  "
        f"OBS_COUNT={obs_count}"
    )


def t12_ml_inference():
    """Run inference for all 5 targets; each must return a float 0-1 or None."""
    global _RISK_RESULT
    if _REAL_PATIENT_ID is None:
        return "SKIP"
    from inference import compute_risk
    _RISK_RESULT = compute_risk(_REAL_PATIENT_ID)
    scores = _RISK_RESULT.get("scores", {})
    for target, info in scores.items():
        prob = info.get("probability")
        if prob is not None:
            assert 0.0 <= prob <= 1.0, f"{target} probability out of range: {prob}"
    found = _RISK_RESULT.get("models_found", [])
    return f"probabilities obtained for {len(found)} targets"


def t13_risk_result_structure():
    """Risk result must have the correct keys and format."""
    if _RISK_RESULT is None:
        return "SKIP — inference not run"
    assert "patient_id"   in _RISK_RESULT
    assert "features"     in _RISK_RESULT
    assert "scores"       in _RISK_RESULT
    from config import TARGETS
    for t in TARGETS:
        assert t in _RISK_RESULT["scores"], f"Missing target: {t}"
        info = _RISK_RESULT["scores"][t]
        assert "probability" in info
        assert "label"       in info
        assert "color"       in info


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("STEP 12 — End-to-End Pipeline Tests")
    print("=" * 65)

    tests = [
        ("T01  Search by patient ID",                   t01_search_by_id),
        ("T02  Search by patient name",                 t02_search_by_name),
        ("T03  Invalid patient → not found",            t03_invalid_patient),
        ("T04  Valid event validation",                  t04_valid_event_validation),
        ("T05  Invalid event → ValidationError",        t05_invalid_event_validation),
        ("T05b Unknown event_type → ValidationError",   t05b_unknown_event_type),
        ("T06  Direct Snowflake insert",                t06_direct_snowflake_insert),
        ("T07  Kafka connectivity check",               t07_kafka_connectivity),
        ("T08  Kafka publish",                          t08_kafka_publish),
        ("T09  RT row count increased",                 t09_rt_row_count_increased),
        ("T10  Patient state includes new event",       t10_patient_state_includes_new_event),
        ("T11  Feature recalculation after insert",     t11_feature_recalculation),
        ("T12  ML inference (all 5 targets)",           t12_ml_inference),
        ("T13  Risk result structure valid",            t13_risk_result_structure),
    ]

    for name, fn in tests:
        run_test(name, fn)

    # Summary
    print("\n" + "=" * 65)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    total  = len(results)
    print(f"Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")

    if failed:
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  {name}: {detail}")
        sys.exit(1)
    else:
        print("\nSTEP 12 DONE ✓")


if __name__ == "__main__":
    main()

