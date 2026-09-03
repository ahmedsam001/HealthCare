
from __future__ import annotations

import uuid
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
        "ENCOUNTER_DATE",       
    ],
    "OBSERVATION": [
        "DESCRIPTION",           # measurement name, e.g. "Body Weight"
        "OBSERVATION_DATE",    
        # At least one of VALUE_NUMERIC or VALUE_TEXT must be non-null
    ],
    "CONDITION": [
        "CODE",
        "DESCRIPTION",
        "START_DATE",           
    ],
    "MEDICATION": [
        "CODE",
        "DESCRIPTION",
        "START_DATE",           
    ],
    "PROCEDURE": [
        "CODE",
        "DESCRIPTION",
        "PROCEDURE_DATE",       
    ],
    "IMMUNIZATION": [
        "CODE",
        "DESCRIPTION",
        "IMMUNIZATION_DATE",    
    ],
    "ALLERGY": [
        "CODE",
        "DESCRIPTION",
        "START_DATE",           
    ],
    "CAREPLAN": [
        "CODE",
        "DESCRIPTION",
        "START_DATE",           
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

class MedicalEvent:
    """Standardized envelope for a new real-time medical event."""
    
    def __init__(
        self, 
        patient_id: str, 
        event_type: str, 
        payload: dict, 
        event_id: str = "", 
        event_timestamp: str = ""
    ):
        self.patient_id = patient_id
        self.event_type = event_type
        self.payload = payload
        
        # Automatically generate an ID if not provided
        if event_id == "":
            self.event_id = str(uuid.uuid4())
        else:
            self.event_id = event_id
            
        # Automatically set the current time if not provided
        if event_timestamp == "":
            self.event_timestamp = datetime.now(timezone.utc).isoformat()
        else:
            self.event_timestamp = event_timestamp

    def to_dict(self) -> dict:  #convert object to dictionary
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
            patient_id      = d["patient_id"],
            event_type      = d["event_type"],
            payload         = d.get("payload", {}),
            event_id        = d.get("event_id", ""),
            event_timestamp = d.get("event_timestamp", ""),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when a MedicalEvent fails schema validation."""


def validate_event(event: MedicalEvent, *, fill_defaults: bool = True) -> MedicalEvent:
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
    required_fields = REQUIRED_PAYLOAD_FIELDS.get(event.event_type, [])
    missing_fields = []
    
    for field_name in required_fields:
        # Check if field is completely missing
        if field_name not in event.payload:
            missing_fields.append(field_name)
        # Check if field is explicitly set to None
        elif event.payload[field_name] is None:
            missing_fields.append(field_name)
        # Check if field is an empty string
        elif str(event.payload[field_name]).strip() == "":
            missing_fields.append(field_name)

    if len(missing_fields) > 0:
        raise ValidationError(
            f"Event type '{event.event_type}' is missing required payload "
            f"fields: {missing_fields}"
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
            
            # If the key is missing or explicitly set to None, we need to fill it
            if key not in event.payload or event.payload[key] is None:
                
                # Special Case 1: Encounter ID
                if key == "ENCOUNTER_ID":
                    # We leave it as None so the database writer can handle it
                    event.payload[key] = None
                    
                # Special Case 2: Careplan ID
                elif key == "CAREPLAN_ID":
                    # Generate a new unique ID for the careplan
                    event.payload[key] = str(uuid.uuid4())
                    
                # Standard Case
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

