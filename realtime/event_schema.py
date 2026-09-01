"""
# STEP 2: Define a standardized schema for new medical events.
#
# Every new medical event uses a common envelope:
#
#   event_id        : UUID (auto-generated if not supplied)
#   patient_id      : must match an existing PATIENTS_GOLD.PATIENT_ID
#   event_type      : one of SUPPORTED_EVENT_TYPES
#   event_timestamp : ISO-8601 string
#   payload         : dict — fields depend on event_type
#
# Payload field requirements match the corresponding Gold table columns
# (from 03_Gold.ipynb).  Only the minimum required fields are enforced;
# extra fields are passed through unchanged.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Supported event types (must match Gold entity names)
# ---------------------------------------------------------------------------

SUPPORTED_EVENT_TYPES: tuple[str, ...] = (
    "ENCOUNTER",
    "OBSERVATION",
    "CONDITION",
    "MEDICATION",
    "PROCEDURE",
    "IMMUNIZATION",
    "ALLERGY",
    "CAREPLAN",
)

# ---------------------------------------------------------------------------
# Minimum required payload fields per event type
# Field names match the Gold-layer column names from 03_Gold.ipynb
# ---------------------------------------------------------------------------

REQUIRED_PAYLOAD_FIELDS: dict[str, list[str]] = {
    "ENCOUNTER": [
        "CODE",
        "DESCRIPTION",
        "ENCOUNTER_DATE",        # ISO-8601 string, e.g. "2026-09-01T10:30:00"
    ],
    "OBSERVATION": [
        "DESCRIPTION",           # measurement name, e.g. "Body Weight"
        "OBSERVATION_DATE",      # ISO-8601 date string
        # At least one of VALUE_NUMERIC or VALUE_TEXT must be non-null
    ],
    "CONDITION": [
        "CODE",
        "DESCRIPTION",
        "START_DATE",            # ISO-8601
    ],
    "MEDICATION": [
        "CODE",
        "DESCRIPTION",
        "START_DATE",            # ISO-8601
    ],
    "PROCEDURE": [
        "CODE",
        "DESCRIPTION",
        "PROCEDURE_DATE",        # ISO-8601
    ],
    "IMMUNIZATION": [
        "CODE",
        "DESCRIPTION",
        "IMMUNIZATION_DATE",     # ISO-8601
    ],
    "ALLERGY": [
        "CODE",
        "DESCRIPTION",
        "START_DATE",            # ISO-8601
    ],
    "CAREPLAN": [
        "CODE",
        "DESCRIPTION",
        "START_DATE",            # ISO-8601
    ],
}

# Optional / nullable fields with sensible defaults applied during validation
OPTIONAL_PAYLOAD_DEFAULTS: dict[str, dict[str, Any]] = {
    "ENCOUNTER": {
        "REASONCODE": "N/A",
        "REASONDESCRIPTION": "Routine / Unspecified",
        "ENCOUNTER_ID": None,    # auto-generated if absent
    },
    "OBSERVATION": {
        "VALUE_NUMERIC": None,
        "VALUE_TEXT": None,
        "READING_SEQ": 1,
        "ENCOUNTER_ID": None,
    },
    "CONDITION": {
        "END_DATE": "9999-12-31T00:00:00",
        "ENCOUNTER_ID": None,
    },
    "MEDICATION": {
        "END_DATE": "9999-12-31T00:00:00",
        "REASONCODE": "N/A",
        "REASONDESCRIPTION": "Unspecified",
        "ENCOUNTER_ID": None,
    },
    "PROCEDURE": {
        "REASONCODE": "N/A",
        "REASONDESCRIPTION": "Unspecified",
        "ENCOUNTER_ID": None,
    },
    "IMMUNIZATION": {
        "ENCOUNTER_ID": None,
    },
    "ALLERGY": {
        "END_DATE": "9999-12-31T00:00:00",
        "ENCOUNTER_ID": None,
    },
    "CAREPLAN": {
        "CAREPLAN_ID": None,     # auto-generated if absent
        "END_DATE": "9999-12-31T00:00:00",
        "REASONCODE": "N/A",
        "REASONDESCRIPTION": "Unspecified",
        "ENCOUNTER_ID": None,
    },
}


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class MedicalEvent:
    """
    Standardized envelope for a new real-time medical event.

    Usage
    -----
    event = MedicalEvent(
        patient_id="some-uuid",
        event_type="OBSERVATION",
        payload={"DESCRIPTION": "Body Weight", "VALUE_NUMERIC": 72.5,
                 "OBSERVATION_DATE": "2026-09-01"},
    )
    validated = validate_event(event)   # raises ValidationError on bad input
    msg = event.to_dict()               # ready to publish to Kafka
    """
    patient_id:      str
    event_type:      str
    payload:         dict[str, Any]
    event_id:        str   = field(default_factory=lambda: str(uuid.uuid4()))
    event_timestamp: str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Serialize the event to a plain dict (JSON-serializable)."""
        return {
            "event_id":        self.event_id,
            "patient_id":      self.patient_id,
            "event_type":      self.event_type,
            "event_timestamp": self.event_timestamp,
            "payload":         self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MedicalEvent":
        """Deserialize from a plain dict (e.g. from a Kafka message)."""
        return cls(
            event_id        = d.get("event_id", str(uuid.uuid4())),
            patient_id      = d["patient_id"],
            event_type      = d["event_type"],
            event_timestamp = d.get("event_timestamp",
                                    datetime.now(timezone.utc).isoformat()),
            payload         = d.get("payload", {}),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when a MedicalEvent fails schema validation."""


def validate_event(event: MedicalEvent, *, fill_defaults: bool = True) -> MedicalEvent:
    """
    Validate a MedicalEvent and optionally fill in default payload fields.

    Parameters
    ----------
    event          : the event to validate
    fill_defaults  : if True, missing optional fields are filled with their
                     default values before returning

    Returns
    -------
    The validated (and optionally default-filled) MedicalEvent.

    Raises
    ------
    ValidationError if any required field is missing or invalid.
    """
    # 1. event_type check
    if event.event_type not in SUPPORTED_EVENT_TYPES:
        raise ValidationError(
            f"Unknown event_type '{event.event_type}'. "
            f"Supported: {SUPPORTED_EVENT_TYPES}"
        )

    # 2. patient_id presence
    if not event.patient_id or not event.patient_id.strip():
        raise ValidationError("patient_id must be a non-empty string.")

    # 3. Required payload fields
    required = REQUIRED_PAYLOAD_FIELDS.get(event.event_type, [])
    missing = [f for f in required if f not in event.payload or
               event.payload[f] is None or str(event.payload[f]).strip() == ""]
    if missing:
        raise ValidationError(
            f"Event type '{event.event_type}' is missing required payload "
            f"fields: {missing}"
        )

    # 4. OBSERVATION-specific: at least one value must be provided
    if event.event_type == "OBSERVATION":
        has_numeric = event.payload.get("VALUE_NUMERIC") is not None
        has_text    = event.payload.get("VALUE_TEXT") is not None
        if not has_numeric and not has_text:
            raise ValidationError(
                "OBSERVATION event must have VALUE_NUMERIC or VALUE_TEXT "
                "(or both) in the payload."
            )

    # 5. Fill optional defaults
    if fill_defaults:
        defaults = OPTIONAL_PAYLOAD_DEFAULTS.get(event.event_type, {})
        for key, default_value in defaults.items():
            if key not in event.payload or event.payload[key] is None:
                if key == "ENCOUNTER_ID" and default_value is None:
                    event.payload[key] = None   # stays None — writer handles
                elif key in ("CAREPLAN_ID",) and default_value is None:
                    event.payload[key] = str(uuid.uuid4())
                else:
                    event.payload[key] = default_value

    return event


def make_observation_event(
    patient_id: str,
    description: str,
    observation_date: str,
    value_numeric: float | None = None,
    value_text: str | None = None,
    encounter_id: str | None = None,
) -> MedicalEvent:
    """Convenience constructor for OBSERVATION events."""
    return MedicalEvent(
        patient_id=patient_id,
        event_type="OBSERVATION",
        payload={
            "DESCRIPTION":      description,
            "OBSERVATION_DATE": observation_date,
            "VALUE_NUMERIC":    value_numeric,
            "VALUE_TEXT":       value_text,
            "ENCOUNTER_ID":     encounter_id,
        },
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("STEP 2 SELF-TEST — Event Schema")

    # Valid event
    ev = MedicalEvent(
        patient_id="patient-uuid-123",
        event_type="OBSERVATION",
        payload={
            "DESCRIPTION":      "Body Weight",
            "OBSERVATION_DATE": "2026-09-01",
            "VALUE_NUMERIC":    72.5,
        },
    )
    validated = validate_event(ev)
    print("Valid event  :", validated.to_dict())

    # Invalid — missing required field
    try:
        bad = MedicalEvent(
            patient_id="patient-uuid-123",
            event_type="CONDITION",
            payload={"DESCRIPTION": "Hypertension"},  # missing CODE, START_DATE
        )
        validate_event(bad)
    except ValidationError as e:
        print("Expected error:", e)

    # Invalid — unknown event type
    try:
        unknown = MedicalEvent(
            patient_id="x",
            event_type="LAB_RESULT",
            payload={},
        )
        validate_event(unknown)
    except ValidationError as e:
        print("Expected error:", e)

    print("STEP 2 DONE")

