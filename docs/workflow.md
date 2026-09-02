# Event Workflow

This document details the complete flow of an event through the system, from user interaction to risk prediction.

## Complete Event Flow
User → Streamlit → Event Validation → Kafka → Spark Structured Streaming → Snowflake REALTIME → Persistence Polling → Feature Computation → XGBoost Inference → Risk Display

## Details

### 1. Event Types
The system processes the following medical event types:
- ENCOUNTER
- OBSERVATION
- CONDITION
- MEDICATION
- PROCEDURE
- IMMUNIZATION
- ALLERGY
- CAREPLAN

### 2. Event Schema
Events follow a standard schema:
- `event_id`: Unique identifier for the event
- `patient_id`: Identifier for the patient
- `event_type`: One of the supported event types
- `payload`: JSON payload of the event details
- `timestamp`: Event creation time

### 3. Kafka Topic
Events are published to the `medical-events` topic.

### 4. Spark Processing
Spark Structured Streaming handles real-time ingestion:
- **JSON parse**: Extracts fields from the event payload.
- **Validate**: Ensures data integrity and schema compliance.
- **Transform**: Formats data for target tables.
- **Write**: Appends to `RT_*` tables in the Snowflake REALTIME schema.

### 5. Persistence Polling
The application polls Snowflake for `EVENT_ID` confirmation to ensure the event was persisted successfully. There is a maximum polling timeout of 30 seconds.

### 6. Feature Computation
Features are computed dynamically from the combined data of GOLD and REALTIME tables, representing the most up-to-date patient state.

### 7. Risk Prediction and Display
Predicted risk levels are categorized and displayed as follows:
- **Low**: < 0.4
- **Moderate**: 0.4 - 0.7
- **High**: >= 0.7
