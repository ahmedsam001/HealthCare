# Healthcare Real-Time Risk Monitoring Platform

> **Academic prototype** demonstrating a production-style healthcare data platform with batch ETL, real-time event streaming, and XGBoost-based disease risk prediction.

---

## Business Problem

Clinicians need a way to see how a patient's disease risk changes as new clinical events occur — in near real-time. Historical batch pipelines produce gold-quality data but only update periodically. This platform bridges that gap: every new medical event flows through Kafka → Spark → Snowflake, and a refreshed risk score is computed within seconds.

---

## Architecture

### Historical Pipeline

```
SyntheticMass / Synthea Dataset
        ↓
Snowflake STAGING
        ↓
PySpark Batch ETL  (01_Bronze.ipynb → 02_Silver.ipynb → 03_Gold.ipynb)
        ↓
BRONZE → SILVER → GOLD
        ↓
ML.PATIENT_FEATURES_TARGETS_LONG
        ↓
5 XGBoost Models Trained (ML_BD1.ipynb)
        ↓
saved_models/xgb_full_features_<target>.joblib
```

### Real-Time Pipeline

```
Streamlit UI (realtime/app.py)
        ↓
Kafka Topic: medical-events
        ↓
Spark Structured Streaming (realtime/spark_streaming.py)
        ↓
Validation + Transformation
        ↓
Snowflake REALTIME Schema (RT_ENCOUNTERS, RT_OBSERVATIONS, …)
        ↓
Patient State = GOLD + REALTIME (UNION queries)
        ↓
Feature Recalculation (realtime/feature_engine.py)
        ↓
5 Saved XGBoost Pipelines (realtime/ml_contract.py)
        ↓
Updated Risk Score → Streamlit
```

---

## Patient State Architecture

```
Patient State = GOLD (read-only, historical)
              UNION
              REALTIME (append-only, new events)
```

| Layer    | Schema       | Role                                      |
|----------|--------------|-------------------------------------------|
| GOLD     | GOLD.*       | Immutable historical records (never modified) |
| REALTIME | REALTIME.RT_* | New live events only                      |

Historical data is **never overwritten** by realtime events.

---

## ML Architecture

| Model | Target | File |
|-------|--------|------|
| XGBoost Pipeline | Diabetes | `xgb_full_features_diabetes.joblib` |
| XGBoost Pipeline | Hypertension | `xgb_full_features_hypertension.joblib` |
| XGBoost Pipeline | Coronary Heart Disease | `xgb_full_features_coronary_heart_disease.joblib` |
| XGBoost Pipeline | Stroke | `xgb_full_features_stroke.joblib` |
| XGBoost Pipeline | Asthma | `xgb_full_features_asthma.joblib` |

Each pipeline is a `sklearn.Pipeline` containing:
- `SimpleImputer` (numeric missing values)
- `OneHotEncoder` (categorical: GENDER, RACE, ETHNICITY, MARITAL)
- `XGBClassifier`

### Feature Contract (18 features)

**Numeric (14):**
`AGE_AT_INDEX`, `AVG_HEIGHT`, `AVG_BMI`, `AVG_WEIGHT`, `AVG_DIASTOLIC_BP`,
`AVG_GLUCOSE`, `AVG_SYSTOLIC_BP`, `AVG_CHOLESTEROL`, `OBSERVATION_COUNT`,
`DISTINCT_CONDITION_COUNT`, `DISTINCT_MEDICATION_COUNT`, `TOTAL_ENCOUNTERS`,
`HAS_GLUCOSE_READING`, `HAS_CHOLESTEROL_READING`

**Categorical (4):**
`GENDER`, `RACE`, `ETHNICITY`, `MARITAL`

> **Age safety rule:** If `AGE_AT_INDEX > 200`, divide by 365.25 (handles days-stored values).

---

## Technologies

| Component | Technology |
|-----------|-----------|
| Data warehouse | Snowflake |
| Batch ETL | PySpark 3.5.6 |
| Streaming | Kafka + Spark Structured Streaming |
| ML models | XGBoost + scikit-learn 1.6.1 |
| UI | Streamlit |
| Language | Python 3.12 |
| Data source | SyntheticMass / Synthea (synthetic EHR) |

---

## Project Structure

```
NTI Prolect/
├── data_ingestion/
│   └── data_ingestion_clean.ipynb   # Snowflake ingestion
│
├── Models/
│   ├── layer/
│   │   ├── 01_Bronze.ipynb          # Spark batch: raw → Bronze
│   │   ├── 02_Silver.ipynb          # Spark batch: Bronze → Silver
│   │   └── 03_Gold.ipynb            # Spark batch: Silver → Gold
│   └── Ml/
│       └── ML_BD1.ipynb             # XGBoost training (5 models)
│
├── realtime/
│   ├── app.py                       # Streamlit dashboard (5 tabs)
│   ├── config.py                    # Centralised configuration
│   ├── event_schema.py              # MedicalEvent dataclass + validation
│   ├── kafka_producer.py            # Kafka publish with direct-insert fallback
│   ├── spark_streaming.py           # Spark Structured Streaming consumer
│   ├── patient_state.py             # Snowflake queries (GOLD ∪ REALTIME)
│   ├── feature_engine.py            # Live feature recalculation
│   ├── inference.py                 # Risk computation entrypoint
│   ├── ml_contract.py               # Model loading + inference contract
│   ├── requirements.txt
│   └── README_REALTIME.md
│
├── saved_models/                    # XGBoost .joblib files (5 models)
│
├── tests/
│   ├── test_event_schema.py         # Schema validation unit tests
│   ├── test_features.py             # Feature contract + integration tests
│   └── test_inference.py            # Inference unit + integration tests
│
├── venv/                            # Python virtual environment
├── .env.example                     # Environment variable template
└── README.md                        # This file
```

---

## Setup

### Prerequisites

- Python 3.12
- Java 11+ (for PySpark/Spark)
- Apache Kafka (running locally on `localhost:9092`)
- Snowflake account with `HEALTHCARE_DB` populated

### 1. Create virtual environment

```bash
cd "NTI Prolect"
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r realtime/requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env and fill in SNOWFLAKE_PASSWORD
source .env   # or export manually
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SNOWFLAKE_PASSWORD` | ✅ Yes | — | Your Snowflake account password |
| `SNOWFLAKE_ACCOUNT` | No | `zj90931.eu-central-2.aws` | Snowflake account identifier |
| `SNOWFLAKE_USER` | No | `ahmedSami` | Snowflake username |
| `SNOWFLAKE_DATABASE` | No | `HEALTHCARE_DB` | Target database |
| `SNOWFLAKE_WAREHOUSE` | No | `HEALTHCARE_WH` | Compute warehouse |
| `SNOWFLAKE_ROLE` | No | `ACCOUNTADMIN` | Snowflake role |
| `KAFKA_BOOTSTRAP_SERVERS` | No | `localhost:9092` | Kafka broker address |
| `KAFKA_TOPIC` | No | `medical-events` | Kafka topic name |
| `MODEL_DIR` | No | `../saved_models` | Path to .joblib model files |

> ⚠️ **Never hardcode the password.** Always use the environment variable.

---

## Kafka Setup

```bash
# Start Zookeeper
$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties &

# Start Kafka broker
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties &

# Create topic (if not already created)
$KAFKA_HOME/bin/kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --replication-factor 1 \
  --partitions 1 \
  --topic medical-events
```

---

## Running the Application

### Terminal 1 — Spark Streaming Consumer

```bash
export SNOWFLAKE_PASSWORD="your_password"
source venv/bin/activate
cd realtime/
python spark_streaming.py
```

Expected output:
```
[Streaming] Spark 3.5.6 started.
[Streaming] Streaming query started. Waiting for data...
```

### Terminal 2 — Streamlit Dashboard

```bash
export SNOWFLAKE_PASSWORD="your_password"
source venv/bin/activate
cd realtime/
streamlit run app.py
```

Visit: **http://localhost:8501**

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 👤 Patient Overview | Demographics, 360 summary (Encounters/Conditions/Medications/Observations counts), clinical records |
| 🎯 Risk Assessment | Current disease risk scores (all 5 models) + Before/After comparison |
| 📅 Clinical Timeline | Chronological view of all events — Historical (blue) + Real-Time (orange) |
| ⚡ Live Event | Event submission form with live validation dashboard + processing status |
| 🖥 System Health | Kafka/Snowflake/Spark status, model file details, last event info, configuration |

---

## Testing

### Unit tests (no Snowflake required)

```bash
source venv/bin/activate
cd "NTI Prolect"
python -m pytest tests/ -v -m "not integration"
```

### Integration tests (require `SNOWFLAKE_PASSWORD`)

```bash
export SNOWFLAKE_PASSWORD="your_password"
python -m pytest tests/ -v
```

---

## Demo Scenario

1. Start Kafka and Spark Streaming (Terminals 1)
2. Open Streamlit (Terminal 2)
3. Search for patient by name (e.g., "Smith")
4. Go to **Risk Assessment** → click **Refresh Risk Scores** → baseline captured
5. Go to **Live Event** → submit an OBSERVATION (e.g., Glucose = 145)
6. Observe: validation passes, Kafka published, Snowflake written, risk updated
7. Return to **Risk Assessment** → before/after comparison shows
8. Check **Clinical Timeline** → new event appears with ⚡ REALTIME badge
9. Check **System Health** → last event details shown

---

## Risk Score Interpretation

> ⚠️ **Disclaimer:** These are model-predicted probabilities, **not clinical diagnoses**. They are outputs of an XGBoost classifier trained on synthetic (Synthea) data. They should not be used for actual medical decisions.

| Score | Tier |
|-------|------|
| ≥ 70% | 🔴 High Risk |
| 40–69% | 🟡 Moderate Risk |
| < 40% | 🟢 Low Risk |

---

## Known Limitations

1. **Spark status not polled** — the dashboard cannot confirm Spark has processed an event. It shows an honest "Awaiting Spark" status.
2. **Synthetic data** — trained on Synthea data, not real EHR data. Clinical validity is not assessed.
3. **No authentication** — the Streamlit app has no login/auth layer (academic prototype).
4. **Single-node Kafka** — not configured for production replication factor.
5. **scikit-learn version pinned** — models require exactly `scikit-learn==1.6.1`.

---

## Future Improvements

- Spark job status polling via REST API
- Patient risk trend charts over time
- Authentication / role-based access
- Alerting for high-risk threshold crossings
- Multi-tenant / multi-patient deployment
- FHIR-compliant event schema
- CI/CD pipeline with automated test runs
