"""
test_event_schema.py
====================
Unit tests for realtime/event_schema.py

Run from project root:
    pytest tests/test_event_schema.py -v

No Snowflake connection required.
"""
import sys
import os

# Allow direct import from realtime/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "realtime"))

import pytest
from event_schema import (
    MedicalEvent, validate_event, ValidationError,
    SUPPORTED_EVENT_TYPES, REQUIRED_PAYLOAD_FIELDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_obs(**kwargs) -> MedicalEvent:
    payload = {
        "DESCRIPTION":      "Body Weight",
        "OBSERVATION_DATE": "2026-09-01",
        "VALUE_NUMERIC":    72.5,
    }
    payload.update(kwargs)
    return MedicalEvent(patient_id="test-pid", event_type="OBSERVATION",
                        payload=payload)


def make_encounter(**kwargs) -> MedicalEvent:
    payload = {
        "CODE":           "185345009",
        "DESCRIPTION":    "Encounter for symptom",
        "ENCOUNTER_DATE": "2026-09-01T10:00:00",
    }
    payload.update(kwargs)
    return MedicalEvent(patient_id="test-pid", event_type="ENCOUNTER",
                        payload=payload)


# ---------------------------------------------------------------------------
# Tests: event_id / event_timestamp auto-generation
# ---------------------------------------------------------------------------
class TestAutoGeneration:
    def test_event_id_is_uuid(self):
        e = MedicalEvent(patient_id="p1", event_type="OBSERVATION", payload={})
        assert len(e.event_id) == 36
        assert e.event_id.count("-") == 4

    def test_two_events_have_different_ids(self):
        e1 = MedicalEvent(patient_id="p1", event_type="OBSERVATION", payload={})
        e2 = MedicalEvent(patient_id="p1", event_type="OBSERVATION", payload={})
        assert e1.event_id != e2.event_id

    def test_event_timestamp_is_iso8601(self):
        e = MedicalEvent(patient_id="p1", event_type="OBSERVATION", payload={})
        assert "T" in e.event_timestamp  # ISO-8601 format


# ---------------------------------------------------------------------------
# Tests: validate_event — happy paths
# ---------------------------------------------------------------------------
class TestValidValidEvents:
    def test_valid_observation_numeric(self):
        e = make_obs()
        result = validate_event(e, fill_defaults=True)
        assert result.event_type == "OBSERVATION"

    def test_valid_observation_text(self):
        e = make_obs(VALUE_NUMERIC=None, VALUE_TEXT="Positive")
        result = validate_event(e, fill_defaults=True)
        assert result is not None

    def test_valid_encounter(self):
        e = make_encounter()
        result = validate_event(e, fill_defaults=True)
        assert result.event_type == "ENCOUNTER"

    def test_valid_condition(self):
        e = MedicalEvent(patient_id="p", event_type="CONDITION", payload={
            "CODE": "44054006", "DESCRIPTION": "Diabetes mellitus",
            "START_DATE": "2026-01-01",
        })
        result = validate_event(e, fill_defaults=True)
        assert result.payload.get("END_DATE") is not None  # default filled

    def test_valid_medication(self):
        e = MedicalEvent(patient_id="p", event_type="MEDICATION", payload={
            "CODE": "429", "DESCRIPTION": "Metformin",
            "START_DATE": "2026-01-01",
        })
        validate_event(e, fill_defaults=True)

    def test_valid_procedure(self):
        e = MedicalEvent(patient_id="p", event_type="PROCEDURE", payload={
            "CODE": "73761001", "DESCRIPTION": "Colonoscopy",
            "PROCEDURE_DATE": "2026-09-01T09:00:00",
        })
        validate_event(e, fill_defaults=True)

    def test_valid_immunization(self):
        e = MedicalEvent(patient_id="p", event_type="IMMUNIZATION", payload={
            "CODE": "08", "DESCRIPTION": "Influenza vaccine",
            "IMMUNIZATION_DATE": "2026-09-01",
        })
        validate_event(e, fill_defaults=True)

    def test_valid_allergy(self):
        e = MedicalEvent(patient_id="p", event_type="ALLERGY", payload={
            "CODE": "372687004", "DESCRIPTION": "Penicillin allergy",
            "START_DATE": "2026-01-01",
        })
        validate_event(e, fill_defaults=True)

    def test_valid_careplan(self):
        e = MedicalEvent(patient_id="p", event_type="CAREPLAN", payload={
            "CODE": "734163000", "DESCRIPTION": "Care plan for diabetes",
            "START_DATE": "2026-01-01",
        })
        validate_event(e, fill_defaults=True)


# ---------------------------------------------------------------------------
# Tests: validate_event — expected failures
# ---------------------------------------------------------------------------
class TestInvalidEvents:
    def test_unknown_event_type(self):
        e = MedicalEvent(patient_id="p", event_type="LAB_RESULT", payload={})
        with pytest.raises(ValidationError, match="Unknown event_type"):
            validate_event(e)

    def test_empty_patient_id(self):
        e = MedicalEvent(patient_id="", event_type="OBSERVATION", payload={
            "DESCRIPTION": "Weight", "OBSERVATION_DATE": "2026-09-01",
            "VALUE_NUMERIC": 70.0,
        })
        with pytest.raises(ValidationError, match="patient_id"):
            validate_event(e)

    def test_whitespace_patient_id(self):
        e = MedicalEvent(patient_id="   ", event_type="OBSERVATION", payload={
            "DESCRIPTION": "Weight", "OBSERVATION_DATE": "2026-09-01",
            "VALUE_NUMERIC": 70.0,
        })
        with pytest.raises(ValidationError, match="patient_id"):
            validate_event(e)

    def test_observation_missing_description(self):
        e = MedicalEvent(patient_id="p", event_type="OBSERVATION", payload={
            "OBSERVATION_DATE": "2026-09-01", "VALUE_NUMERIC": 70.0,
        })
        with pytest.raises(ValidationError, match="missing required"):
            validate_event(e)

    def test_observation_missing_date(self):
        e = MedicalEvent(patient_id="p", event_type="OBSERVATION", payload={
            "DESCRIPTION": "Weight", "VALUE_NUMERIC": 70.0,
        })
        with pytest.raises(ValidationError, match="missing required"):
            validate_event(e)

    def test_observation_no_value(self):
        e = MedicalEvent(patient_id="p", event_type="OBSERVATION", payload={
            "DESCRIPTION": "Weight", "OBSERVATION_DATE": "2026-09-01",
        })
        with pytest.raises(ValidationError, match="VALUE_NUMERIC or VALUE_TEXT"):
            validate_event(e)

    def test_encounter_missing_code(self):
        e = MedicalEvent(patient_id="p", event_type="ENCOUNTER", payload={
            "DESCRIPTION": "Visit", "ENCOUNTER_DATE": "2026-09-01T10:00:00",
        })
        with pytest.raises(ValidationError, match="missing required"):
            validate_event(e)

    def test_condition_missing_start_date(self):
        e = MedicalEvent(patient_id="p", event_type="CONDITION", payload={
            "CODE": "44054006", "DESCRIPTION": "Diabetes",
        })
        with pytest.raises(ValidationError, match="missing required"):
            validate_event(e)


# ---------------------------------------------------------------------------
# Tests: fill_defaults
# ---------------------------------------------------------------------------
class TestFillDefaults:
    def test_observation_gets_reading_seq(self):
        e = make_obs()
        result = validate_event(e, fill_defaults=True)
        assert result.payload.get("READING_SEQ") == 1

    def test_condition_gets_end_date(self):
        e = MedicalEvent(patient_id="p", event_type="CONDITION", payload={
            "CODE": "44054006", "DESCRIPTION": "Diabetes",
            "START_DATE": "2026-01-01",
        })
        result = validate_event(e, fill_defaults=True)
        assert "END_DATE" in result.payload

    def test_careplan_gets_careplan_id(self):
        e = MedicalEvent(patient_id="p", event_type="CAREPLAN", payload={
            "CODE": "xxx", "DESCRIPTION": "Care plan",
            "START_DATE": "2026-01-01",
        })
        result = validate_event(e, fill_defaults=True)
        assert result.payload.get("CAREPLAN_ID") is not None


# ---------------------------------------------------------------------------
# Tests: serialisation round-trip
# ---------------------------------------------------------------------------
class TestSerialisation:
    def test_to_dict_and_from_dict(self):
        e = make_encounter()
        d = e.to_dict()
        assert set(d.keys()) == {
            "event_id", "patient_id", "event_type", "event_timestamp", "payload"
        }
        e2 = MedicalEvent.from_dict(d)
        assert e2.event_id == e.event_id
        assert e2.patient_id == e.patient_id
        assert e2.event_type == e.event_type

    def test_all_supported_types_covered_in_required_fields(self):
        for t in SUPPORTED_EVENT_TYPES:
            assert t in REQUIRED_PAYLOAD_FIELDS, f"{t} missing from REQUIRED_PAYLOAD_FIELDS"

