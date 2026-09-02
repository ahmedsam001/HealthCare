# Healthcare Real-Time Risk Monitoring Platform

A production-style healthcare data platform that combines historical patient data with
real-time medical events to provide live disease risk predictions using machine learning.

## Architecture

```text
SyntheticMass (Synthea)
        ↓
  Snowflake STAGING
        ↓
  Spark Batch ETL
        ↓
  ┌─────┼─────┐
  ↓     ↓     ↓
BRONZE SILVER GOLD ─── ML Training ─── 5 XGBoost Models
                                              ↓
New Event ─→ Streamlit ─→ Kafka ─→ Spark Streaming ─→ REALTIME
                                                          ↓
                                              Patient State (GOLD ∪ REALTIME)
                                                          ↓
                                                   Feature Engine
                                                          ↓
                                                  Risk Prediction
                                                          ↓
                                                    Dashboard UI
```

## Disease Risk Models

| Target | Model | Features |
|--------|-------|----------|
| Diabetes | XGBoost | 14 numeric + 4 categorical |
| Hypertension | XGBoost | 14 numeric + 4 categorical |
| Coronary Heart Disease | XGBoost | 14 numeric + 4 categorical |
| Stroke | XGBoost | 14 numeric + 4 categorical |
| Asthma | XGBoost | 14 numeric + 4 categorical |

## Project Structure

```text
healthcare-risk-platform/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data_ingestion/
│   └── data_ingestion.ipynb         # Synthea → Snowflake STAGING
│
├── notebooks/
│   ├── 01_bronze.ipynb              # STAGING → BRONZE
│   ├── 02_silver.ipynb              # BRONZE → SILVER
│   ├── 03_gold.ipynb                # SILVER → GOLD + OBT
│   └── 04_ml_training.ipynb         # GOLD → XGBoost models
│
├── realtime/
│   ├── app.py                       # Streamlit dashboard (5 tabs)
│   ├── config.py                    # Centralized configuration
│   ├── event_schema.py              # Event validation & schema
│   ├── kafka_producer.py            # Kafka publisher
│   ├── spark_streaming.py           # Spark Structured Streaming consumer
│   ├── patient_state.py             # GOLD ∪ REALTIME queries
│   ├── feature_engine.py            # 18-feature computation
│   ├── ml_contract.py               # ML feature contract & model loading
│   └── inference.py                 # Risk prediction orchestration
│
├── models/
│   ├── diabetes/
│   │   ├── model.joblib
│   │   └── metadata.json
│   ├── hypertension/
│   │   ├── model.joblib
│   │   └── metadata.json
│   ├── coronary_heart_disease/
│   │   ├── model.joblib
│   │   └── metadata.json
│   ├── stroke/
│   │   ├── model.joblib
│   │   └── metadata.json
│   └── asthma/
│       ├── model.joblib
│       └── metadata.json
│
├── tests/
│   ├── test_event_schema.py
│   ├── test_features.py
│   ├── test_inference.py
│   └── test_e2e.py
│
└── docs/
    ├── architecture.md
    ├── workflow.md
    └── ml.md
```

## Quick Start

### Prerequisites

- Python 3.12+
- Apache Spark 3.5.x
- Apache Kafka
- Snowflake account with HEALTHCARE_DB

### Installation

```bash
git clone <repo-url>
cd healthcare-risk-platform

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your Snowflake credentials
source .env
```

### Running the System

**Terminal 1 — Kafka** (must be running first):
```bash
# Start Kafka broker (depends on your installation)
```

**Terminal 2 — Spark Streaming**:
```bash
export SNOWFLAKE_PASSWORD="<your_password>"
cd realtime
python spark_streaming.py
```

**Terminal 3 — Streamlit Dashboard**:
```bash
export SNOWFLAKE_PASSWORD="<your_password>"
cd realtime
streamlit run app.py
```

### Running Tests

```bash
export SNOWFLAKE_PASSWORD="<your_password>"

# Unit + integration tests
python -m pytest tests/ -v

# End-to-end pipeline test
python tests/test_e2e.py
```

## Data Pipeline

### Batch ETL (Historical Data)

1. **Data Ingestion**: Download Synthea dataset → load CSVs into Snowflake `STAGING`
2. **Bronze**: Raw staging with ingestion metadata
3. **Silver**: Cleaned, typed, deduplicated, standardized
4. **Gold**: PII-stripped, observations pivoted, OBT (One Big Table) constructed
5. **ML Training**: Feature engineering → XGBoost models for 5 diseases

### Real-Time Pipeline

1. **Event Submission**: User submits medical event via Streamlit
2. **Kafka Publishing**: Event published to `medical-events` topic
3. **Spark Streaming**: Consumes, validates, transforms, writes to `REALTIME.RT_*` tables
4. **Persistence Confirmation**: App polls Snowflake until event_id is confirmed (max 30s)
5. **Feature Computation**: 18 features computed from `GOLD ∪ REALTIME`
6. **Risk Prediction**: 5 XGBoost models produce disease probabilities
7. **Dashboard Update**: Before/after risk comparison displayed

## Key Design Decisions

- **GOLD is read-only**: Historical data is never modified by real-time events
- **REALTIME is append-only**: New events go to `RT_*` tables
- **Patient State = GOLD UNION REALTIME**: Queries combine both sources
- **Weighted averages**: Feature engine uses SUM/COUNT from both sources for correct combined averages
- **Persistence before risk**: Risk is never calculated until event is confirmed in Snowflake
- **Kafka is primary path**: Direct Snowflake insert is fallback only when Kafka is unavailable

## Documentation

- [Architecture](docs/architecture.md) — System components and data flow
- [Workflow](docs/workflow.md) — Complete event processing pipeline
- [ML Models](docs/ml.md) — Model training, features, and inference
