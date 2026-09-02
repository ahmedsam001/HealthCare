"""
# STEP 5: Centralized configuration for the real-time healthcare pipeline.
# All secrets are read from environment variables — never hardcoded.
# Set these before running:
#   SNOWFLAKE_PASSWORD
#   KAFKA_BOOTSTRAP_SERVERS  (optional, default: localhost:9092)
"""

import os

# ---------------------------------------------------------------------------
# Snowflake connection
# Use the same account/db as the batch layer notebooks (01_Bronze through
# 03_Gold). Override via env vars to switch environments.
# ---------------------------------------------------------------------------
SNOWFLAKE_ACCOUNT   = os.getenv("SNOWFLAKE_ACCOUNT",   "zj90931.eu-central-2.aws")
SNOWFLAKE_USER      = os.getenv("SNOWFLAKE_USER",       "ahmedSami")
SNOWFLAKE_PASSWORD  = os.getenv("SNOWFLAKE_PASSWORD",   "")   # MUST be set
SNOWFLAKE_DATABASE  = os.getenv("SNOWFLAKE_DATABASE",   "HEALTHCARE_DB")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE",  "HEALTHCARE_WH")
SNOWFLAKE_ROLE      = os.getenv("SNOWFLAKE_ROLE",       "ACCOUNTADMIN")

# Schema names (read-only — match the batch pipeline)
SCHEMA_GOLD    = "GOLD"
SCHEMA_SILVER  = "SILVER"
SCHEMA_ML      = "ML"

# Schema name for real-time appended events (never modify GOLD directly)
SCHEMA_REALTIME = "REALTIME"

# ---------------------------------------------------------------------------
# Kafka
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC",             "medical-events")

# ---------------------------------------------------------------------------
# ML model artifacts
# Saved by Models/Ml/ML_BD1.ipynb → saved_models/xgb_full_features_<target>.joblib
# Resolve relative to this file's location (realtime/ dir) → parent → saved_models/
# ---------------------------------------------------------------------------
import pathlib
_BASE = pathlib.Path(__file__).parent.parent  # NTI Prolect/
MODEL_DIR = pathlib.Path(os.getenv("MODEL_DIR", str(_BASE / "models")))

# ---------------------------------------------------------------------------
# Supported targets and their model file names
# (exact names used in ML_BD1.ipynb)
# ---------------------------------------------------------------------------
TARGETS = [
    "Diabetes",
    "Hypertension",
    "Coronary Heart Disease",
    "Stroke",
    "Asthma",
]


def model_path(target_name: str) -> pathlib.Path:
    """Return the expected .joblib path for a given target name."""
    subdir = target_name.lower().replace(' ', '_')
    return MODEL_DIR / subdir / "model.joblib"


def model_metadata_path(target_name: str) -> pathlib.Path:
    """Return the expected metadata.json path for a given target name."""
    subdir = target_name.lower().replace(' ', '_')
    return MODEL_DIR / subdir / "metadata.json"


def snowflake_connect_kwargs() -> dict:
    """Return keyword arguments for snowflake.connector.connect()."""
    if not SNOWFLAKE_PASSWORD:
        raise RuntimeError(
            "SNOWFLAKE_PASSWORD environment variable is not set. "
            "Export it before running the pipeline."
        )
    return {
        "account":   SNOWFLAKE_ACCOUNT,
        "user":      SNOWFLAKE_USER,
        "password":  SNOWFLAKE_PASSWORD,
        "database":  SNOWFLAKE_DATABASE,
        "warehouse": SNOWFLAKE_WAREHOUSE,
        "role":      SNOWFLAKE_ROLE,
    }

