import json
from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
from event_schema import MedicalEvent, validate_event
from kafka import KafkaProducer

# ---------------------------------------------------------------------------
# Serializer Functions (Explicit for students)
# ---------------------------------------------------------------------------
def _serialize_value(value_dict):
    """Convert a dictionary to a JSON byte string."""
    json_string = json.dumps(value_dict)
    return json_string.encode("utf-8")

def _serialize_key(key_string):
    """Convert a string key to a byte string."""
    if key_string is None:
        return None
    return key_string.encode("utf-8")

# ---------------------------------------------------------------------------
# Producer singleton (lazy initialization)
# ---------------------------------------------------------------------------

_producer = None

def _get_producer():
    """Lazily create and return the Kafka producer."""
    global _producer
    
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=_serialize_value,
            key_serializer=_serialize_key,
            acks="all",
            retries=3,
        )
    return _producer

def produce_event(event, validate=True):
    """
    Validate and publish a MedicalEvent to the Kafka medical-events topic.
    Returns True on success, False on failure.
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

def produce_event_async(event, validate=True):
    """
    Non-blocking publish. Errors are just printed to the console.
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

def close_producer():
    """Flush and close the Kafka producer (call on shutdown)."""
    global _producer
    if _producer is not None:
        _producer.flush()
        _producer.close()
        _producer = None

def test_connectivity():
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
