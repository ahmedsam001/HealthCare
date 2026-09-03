"""
# STEP 6: Publish validated medical events to Kafka.
#
# Topic: medical-events (configurable via KAFKA_TOPIC env var)
# Bootstrap: localhost:9092 (configurable via KAFKA_BOOTSTRAP_SERVERS env var)
# Serialization: JSON (UTF-8)
# Key: patient_id (for partition routing by patient)
#
# Usage:
#   from kafka_producer import produce_event
#   produce_event(validated_medical_event)
"""

from __future__ import annotations

import json
import time
from typing import Optional

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
from event_schema import MedicalEvent, validate_event


import threading

# ---------------------------------------------------------------------------
# Serializer Functions (Explicit for students)
# ---------------------------------------------------------------------------
def _serialize_value(value_dict: dict) -> bytes:
    """Convert a dictionary to a JSON byte string."""
    json_string = json.dumps(value_dict)
    return json_string.encode("utf-8")

def _serialize_key(key_string: str) -> bytes:
    """Convert a string key to a byte string."""
    if key_string is None:
        return None
    return key_string.encode("utf-8")


# ---------------------------------------------------------------------------
# Producer singleton (lazy initialization with thread safety)
# ---------------------------------------------------------------------------

_producer = None
_producer_lock = threading.Lock()

def _get_producer():
    """Lazily create and return the Kafka producer in a thread-safe way."""
    global _producer
    
    with _producer_lock:
        if _producer is None:
            try:
                from kafka import KafkaProducer
            except ImportError:
                raise ImportError(
                    "kafka-python is not installed. "
                    "Run: pip install kafka-python"
                )

            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=_serialize_value,
                key_serializer=_serialize_key,
                # Reliability settings
                acks="all",                 # wait for all in-sync replicas
                retries=3,
                request_timeout_ms=30_000,
                # Allow up to 64 KB messages (large payloads are rare in medical events)
                max_request_size=65_536,
            )
    return _producer


def produce_event(event: MedicalEvent, *, validate: bool = True) -> bool:
    """
    Validate and publish a MedicalEvent to the Kafka medical-events topic.

    Parameters
    ----------
    event    : MedicalEvent to publish
    validate : if True, validate + fill defaults before publishing

    Returns
    -------
    True on success, False on failure (with error logged to stderr).
    """
    if validate:
        event = validate_event(event, fill_defaults=True)

    try:
        producer = _get_producer()

        # Publish; keyed by patient_id for partition affinity
        future = producer.send(
            topic=KAFKA_TOPIC,
            key=event.patient_id,
            value=event.to_dict(),
        )

        # Block until the broker acknowledges (with 10s timeout)
        record_metadata = future.get(timeout=10)

        print(
            f"[Kafka] Published event_id={event.event_id} "
            f"patient={event.patient_id} type={event.event_type} "
            f"→ {KAFKA_TOPIC} "
            f"[partition={record_metadata.partition}, "
            f"offset={record_metadata.offset}]"
        )
        return True

    except Exception as exc:
        print(f"[Kafka ERROR] Failed to publish event: {exc}")
        return False


def produce_event_async(event: MedicalEvent, *, validate: bool = True) -> None:
    """
    Non-blocking publish. Errors are just printed to the console.
    This is simpler for students to understand than nested callbacks.
    """
    if validate:
        event = validate_event(event, fill_defaults=True)

    event_dict = event.to_dict()
    producer = _get_producer()
    
    try:
        producer.send(
            topic=KAFKA_TOPIC,
            key=event.patient_id,
            value=event_dict,
        )
    except Exception as exc:
        print(f"[Kafka ERROR Async] {exc}")


def close_producer() -> None:
    """Flush and close the Kafka producer (call on shutdown)."""
    global _producer
    with _producer_lock:
        if _producer is not None:
            _producer.flush()
            _producer.close()
            _producer = None


def test_connectivity() -> bool:
    """
    Try to connect to Kafka and return True if successful.
    Used by the Streamlit app to show a connection status indicator.
    """
    try:
        p = _get_producer()
        p.bootstrap_connected()
        return True
    except Exception as exc:
        print(f"[Kafka] Connection test failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("STEP 6 SELF-TEST — Kafka Producer")
    print(f"  Bootstrap: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"  Topic    : {KAFKA_TOPIC}")

    connected = test_connectivity()
    if not connected:
        print("  ⚠ Kafka not reachable. Start Kafka and retry.")
    else:
        print("  ✓ Kafka connected")

        test_event = MedicalEvent(
            patient_id="test-patient-001",
            event_type="OBSERVATION",
            payload={
                "DESCRIPTION":      "Body Weight",
                "OBSERVATION_DATE": "2026-09-01",
                "VALUE_NUMERIC":    72.5,
            },
        )

        ok = produce_event(test_event)
        if ok:
            print("  ✓ Test message published successfully")
        else:
            print("  ✗ Publish failed")

        close_producer()

    print("STEP 6 DONE")

