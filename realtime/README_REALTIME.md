# Healthcare Real-Time Medical Event Pipeline

## Overview

This directory adds a **real-time event pipeline** on top of the existing batch pipeline. The historical Bronze/Silver/Gold/ML notebooks are **not modified**.

```
Historical:   STAGING → Bronze → Silver → Gold → ML.PATIENT_FEATURES → XGBoost models
                                                                              ↑
Real-time:  Streamlit → Kafka → Spark Streaming → REALTIME schema → Updated Features → Updated Risk → Streamlit
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| Kafka broker | any (tested with 3.x) |
| Snowflake | existing `HEALTHCARE_DB` with GOLD tables populated |
| Saved XGBoost models | `../saved_models/xgb_full_features_<target>.joblib` |

---

## Setup

### 1. Install dependencies

```bash
cd "NTI Prolect/realtime"
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export SNOWFLAKE_PASSWORD="yq8aQN9yu82zkRH"

# Optional overrides (defaults shown):
export SNOWFLAKE_ACCOUNT="zj90931.eu-central-2.aws"
export SNOWFLAKE_USER="ahmedSami"
export SNOWFLAKE_DATABASE="HEALTHCARE_DB"
export SNOWFLAKE_WAREHOUSE="HEALTHCARE_WH"
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export KAFKA_TOPIC="medical-events"
export MODEL_DIR="../saved_models"   # path to the .joblib files
```

### 3. Copy ML model files

The XGBoost pipelines were saved by `Models/Ml/ML_BD1.ipynb` to `saved_models/`.
Copy them here:

```bash
# If you ran the notebook on Google Colab, download them first, then:
ls ../saved_models/
# Should show:
#   xgb_full_features_diabetes.joblib
#   xgb_full_features_hypertension.joblib
#   xgb_full_features_coronary_heart_disease.joblib
#   xgb_full_features_stroke.joblib
#   xgb_full_features_asthma.joblib
```

### 4. Start Kafka (if running locally)

```bash
# Using a standard Kafka installation:
bin/zookeeper-server-start.sh config/zookeeper.properties &
bin/kafka-server-start.sh config/server.properties &

# Create the topic:
bin/kafka-topics.sh --create \
  --topic medical-events \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

---

## Running the Pipeline

### Terminal 1 — Spark Structured Streaming (Kafka consumer)

```bash
cd "NTI Prolect/realtime"
export SNOWFLAKE_PASSWORD="yq8aQN9yu82zkRH"
python spark_streaming.py
```

The Spark job will:
- Connect to Kafka (`medical-events` topic)
- Parse and validate each event
- Write to `HEALTHCARE_DB.REALTIME.RT_<TYPE>` tables in Snowflake

### Terminal 2 — Streamlit App

```bash
cd "NTI Prolect/realtime"
export SNOWFLAKE_PASSWORD="yq8aQN9yu82zkRH"
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Using the Streamlit App

1. **Search** for a patient by ID or name in the left sidebar.
2. **View** demographics, encounters, conditions, medications, observations, allergies.
3. Click **Refresh Risk Scores** to see current disease risk from the ML model.
4. In **Part B**, select an event type (e.g., OBSERVATION) and fill in the form.
5. Click **Submit Event** — the event is:
   - Validated against the schema
   - Published to Kafka
   - Written directly to REALTIME schema (immediate reflection)
   - Reflected in updated risk scores automatically

---

## Step-by-Step Testing

```bash
cd "NTI Prolect/realtime"
export SNOWFLAKE_PASSWORD="yq8aQN9yu82zkRH"
python test_e2e.py
```

Expected output (with Kafka offline — SKIP is acceptable):
```
T01  Search by patient ID                  ✓ PASS
T02  Search by patient name                ✓ PASS
T03  Invalid patient → not found           ✓ PASS
T04  Valid event validation                ✓ PASS
T05  Invalid event → ValidationError       ✓ PASS
T05b Unknown event_type → ValidationError  ✓ PASS
T06  Direct Snowflake insert               ✓ PASS
T07  Kafka connectivity check              ⚫ SKIP  (Kafka not running)
T08  Kafka publish                         ⚫ SKIP
T09  RT row count increased                ✓ PASS
T10  Patient state includes new event      ✓ PASS
T11  Feature recalculation after insert    ✓ PASS
T12  ML inference (all 5 targets)          ✓ PASS  (or ⚫ SKIP if no .joblib files)
T13  Risk result structure valid           ✓ PASS
```

---

## File Structure

```
realtime/
├── config.py           — Snowflake + Kafka config (env-var driven)
├── ml_contract.py      — STEP 1: ML feature contract + inference API
├── event_schema.py     — STEP 2: Event envelope + per-type validation
├── patient_state.py    — STEP 5+8: REALTIME DDL, direct insert, state queries
├── feature_engine.py   — STEP 9: ML feature recalculation (GOLD ∪ REALTIME)
├── inference.py        — STEP 10: Run existing saved models → risk scores
├── kafka_producer.py   — STEP 6: Kafka producer
├── spark_streaming.py  — STEP 7: Spark Structured Streaming consumer
├── app.py              — STEP 11: Streamlit UI
├── test_e2e.py         — STEP 12: End-to-end tests
├── requirements.txt    — Python dependencies
└── README_REALTIME.md  — This file
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│             HISTORICAL PIPELINE (unchanged)                │
│  STAGING → Bronze → Silver → Gold → ML Features → XGBoost │
└──────────────────────────┬─────────────────────────────────┘
                           │ reads GOLD tables (read-only)
                           ▼
┌────────────────────────────────────────────────────────────┐
│              REAL-TIME PIPELINE (new — this dir)           │
│                                                            │
│  Streamlit UI                                              │
│      │ validate (event_schema.py)                          │
│      │                                                     │
│      ├──→ Kafka Producer (kafka_producer.py)               │
│      │         │                                           │
│      │         ▼                                           │
│      │    Kafka [medical-events]                           │
│      │         │                                           │
│      │         ▼                                           │
│      │    Spark Streaming (spark_streaming.py)             │
│      │         │                                           │
│      │         ▼                                           │
│      │    Snowflake REALTIME.RT_* (append only)            │
│      │                                                     │
│      └──→ Direct Insert (patient_state.py) ─────┐          │
│                                                 │          │
│  Patient State = GOLD ∪ REALTIME ←─────────────┘          │
│      │ (patient_state.py)                                  │
│      ▼                                                     │
│  Updated Features (feature_engine.py)                      │
│      ▼                                                     │
│  Existing .joblib pipelines (inference.py + ml_contract)   │
│      ▼                                                     │
│  Updated Risk → Streamlit UI                               │
└────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `SNOWFLAKE_PASSWORD not set` | Export the env var before running |
| `Model file not found` | Copy `.joblib` files from ML training environment |
| `Kafka unavailable` | The app falls back to direct Snowflake insert automatically |
| `REALTIME schema missing` | Runs `ensure_realtime_schema()` at startup automatically |
| `Connection refused (Snowflake)` | Check `SNOWFLAKE_ACCOUNT` env var — use `zj90931.eu-central-2.aws` |

