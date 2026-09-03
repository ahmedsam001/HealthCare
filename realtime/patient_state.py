
"""
# STEP 5 (direct Snowflake) + STEP 8: Patient state queries.
#
# This module provides:
#   1. REALTIME schema / table creation (idempotent DDL).
#   2. Direct insert of a validated event into the REALTIME schema
#      (used for testing without Kafka in STEP 5).
#   3. Patient-state read queries that UNION GOLD + REALTIME tables
#      so historical records are NEVER modified.
#   4. Patient search by PATIENT_ID or FULL_NAME.
#
# Architecture rule preserved:
#   Historical GOLD data → read-only
#   New events          → append-only to REALTIME.RT_<TYPE>
#   Patient view        → UNION of both
"""

from __future__ import annotations
import streamlit as st

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import snowflake.connector

from config import (
    snowflake_connect_kwargs,
    SCHEMA_GOLD,
    SCHEMA_REALTIME,
)
from event_schema import MedicalEvent, validate_event, ValidationError


# ---------------------------------------------------------------------------
# DDL: REALTIME schema tables
# Each RT_ table mirrors the corresponding Gold table structure but adds
# EVENT_ID and INGESTION_TS for traceability.
# ---------------------------------------------------------------------------

REALTIME_DDL: dict[str, str] = {
    "RT_ENCOUNTERS": """
        CREATE TABLE IF NOT EXISTS REALTIME.RT_ENCOUNTERS (
            EVENT_ID            VARCHAR,
            PATIENT_ID          VARCHAR,
            ENCOUNTER_ID        VARCHAR,
            CODE                VARCHAR,
            DESCRIPTION         VARCHAR,
            REASONCODE          VARCHAR,
            REASONDESCRIPTION   VARCHAR,
            ENCOUNTER_DATE      TIMESTAMP_NTZ,
            INGESTION_TS        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "RT_OBSERVATIONS": """
        CREATE TABLE IF NOT EXISTS REALTIME.RT_OBSERVATIONS (
            EVENT_ID            VARCHAR,
            PATIENT_ID          VARCHAR,
            ENCOUNTER_ID        VARCHAR,
            DESCRIPTION         VARCHAR,
            OBSERVATION_DATE    DATE,
            VALUE_NUMERIC       FLOAT,
            VALUE_TEXT          VARCHAR,
            READING_SEQ         INT,
            INGESTION_TS        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "RT_CONDITIONS": """
        CREATE TABLE IF NOT EXISTS REALTIME.RT_CONDITIONS (
            EVENT_ID            VARCHAR,
            PATIENT_ID          VARCHAR,
            ENCOUNTER_ID        VARCHAR,
            CODE                VARCHAR,
            DESCRIPTION         VARCHAR,
            START_DATE          TIMESTAMP_NTZ,
            END_DATE            TIMESTAMP_NTZ,
            INGESTION_TS        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "RT_MEDICATIONS": """
        CREATE TABLE IF NOT EXISTS REALTIME.RT_MEDICATIONS (
            EVENT_ID            VARCHAR,
            PATIENT_ID          VARCHAR,
            ENCOUNTER_ID        VARCHAR,
            CODE                VARCHAR,
            DESCRIPTION         VARCHAR,
            REASONCODE          VARCHAR,
            REASONDESCRIPTION   VARCHAR,
            START_DATE          TIMESTAMP_NTZ,
            END_DATE            TIMESTAMP_NTZ,
            INGESTION_TS        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "RT_PROCEDURES": """
        CREATE TABLE IF NOT EXISTS REALTIME.RT_PROCEDURES (
            EVENT_ID            VARCHAR,
            PATIENT_ID          VARCHAR,
            ENCOUNTER_ID        VARCHAR,
            CODE                VARCHAR,
            DESCRIPTION         VARCHAR,
            REASONCODE          VARCHAR,
            REASONDESCRIPTION   VARCHAR,
            PROCEDURE_DATE      TIMESTAMP_NTZ,
            INGESTION_TS        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "RT_IMMUNIZATIONS": """
        CREATE TABLE IF NOT EXISTS REALTIME.RT_IMMUNIZATIONS (
            EVENT_ID            VARCHAR,
            PATIENT_ID          VARCHAR,
            ENCOUNTER_ID        VARCHAR,
            CODE                VARCHAR,
            DESCRIPTION         VARCHAR,
            IMMUNIZATION_DATE   TIMESTAMP_NTZ,
            INGESTION_TS        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "RT_ALLERGIES": """
        CREATE TABLE IF NOT EXISTS REALTIME.RT_ALLERGIES (
            EVENT_ID            VARCHAR,
            PATIENT_ID          VARCHAR,
            ENCOUNTER_ID        VARCHAR,
            CODE                VARCHAR,
            DESCRIPTION         VARCHAR,
            START_DATE          TIMESTAMP_NTZ,
            END_DATE            TIMESTAMP_NTZ,
            INGESTION_TS        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "RT_CAREPLANS": """
        CREATE TABLE IF NOT EXISTS REALTIME.RT_CAREPLANS (
            EVENT_ID            VARCHAR,
            CAREPLAN_ID         VARCHAR,
            PATIENT_ID          VARCHAR,
            ENCOUNTER_ID        VARCHAR,
            CODE                VARCHAR,
            DESCRIPTION         VARCHAR,
            REASONCODE          VARCHAR,
            REASONDESCRIPTION   VARCHAR,
            START_DATE          TIMESTAMP_NTZ,
            END_DATE            TIMESTAMP_NTZ,
            INGESTION_TS        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
        )
    """,
}


def _get_conn() -> snowflake.connector.SnowflakeConnection:
    return snowflake.connector.connect(**snowflake_connect_kwargs())


# ---------------------------------------------------------------------------
# Schema + table setup (idempotent — safe to call at startup)
# ---------------------------------------------------------------------------

def ensure_realtime_schema() -> None:
    """Create the REALTIME schema and all RT_ tables if they don't exist."""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS HEALTHCARE_DB.REALTIME")
        cur.execute("USE SCHEMA HEALTHCARE_DB.REALTIME")
        for table_name, ddl in REALTIME_DDL.items():
            cur.execute(ddl)
            print(f"  ✓ {table_name} ready")
        cur.close()
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# STEP 5: Direct Snowflake write (no Kafka, for testing)
# ---------------------------------------------------------------------------

def insert_event_direct(event: MedicalEvent) -> None:
    """
    Validate and insert a MedicalEvent directly into the REALTIME schema.
    This path is used for STEP 5 testing (bypasses Kafka/Spark).
    """
    # Validate first — raises ValidationError on bad input
    event = validate_event(event, fill_defaults=True)

    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("USE SCHEMA HEALTHCARE_DB.REALTIME")

        p = event.payload
        now = datetime.now(timezone.utc).isoformat()

        if event.event_type == "ENCOUNTER":
            enc_id = p.get("ENCOUNTER_ID") or str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO RT_ENCOUNTERS
                    (EVENT_ID, PATIENT_ID, ENCOUNTER_ID, CODE, DESCRIPTION,
                     REASONCODE, REASONDESCRIPTION, ENCOUNTER_DATE, INGESTION_TS)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (event.event_id, event.patient_id, enc_id,
                 p["CODE"], p["DESCRIPTION"],
                 p.get("REASONCODE", "N/A"),
                 p.get("REASONDESCRIPTION", "Routine / Unspecified"),
                 p["ENCOUNTER_DATE"], now),
            )

        elif event.event_type == "OBSERVATION":
            cur.execute(
                """
                INSERT INTO RT_OBSERVATIONS
                    (EVENT_ID, PATIENT_ID, ENCOUNTER_ID, DESCRIPTION,
                     OBSERVATION_DATE, VALUE_NUMERIC, VALUE_TEXT, READING_SEQ, INGESTION_TS)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (event.event_id, event.patient_id,
                 p.get("ENCOUNTER_ID"),
                 p["DESCRIPTION"], p["OBSERVATION_DATE"],
                 p.get("VALUE_NUMERIC"), p.get("VALUE_TEXT"),
                 p.get("READING_SEQ", 1), now),
            )

        elif event.event_type == "CONDITION":
            cur.execute(
                """
                INSERT INTO RT_CONDITIONS
                    (EVENT_ID, PATIENT_ID, ENCOUNTER_ID, CODE, DESCRIPTION,
                     START_DATE, END_DATE, INGESTION_TS)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (event.event_id, event.patient_id,
                 p.get("ENCOUNTER_ID"),
                 p["CODE"], p["DESCRIPTION"],
                 p["START_DATE"],
                 p.get("END_DATE", "9999-12-31"), now),
            )

        elif event.event_type == "MEDICATION":
            cur.execute(
                """
                INSERT INTO RT_MEDICATIONS
                    (EVENT_ID, PATIENT_ID, ENCOUNTER_ID, CODE, DESCRIPTION,
                     REASONCODE, REASONDESCRIPTION, START_DATE, END_DATE, INGESTION_TS)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (event.event_id, event.patient_id,
                 p.get("ENCOUNTER_ID"),
                 p["CODE"], p["DESCRIPTION"],
                 p.get("REASONCODE", "N/A"),
                 p.get("REASONDESCRIPTION", "Unspecified"),
                 p["START_DATE"],
                 p.get("END_DATE", "9999-12-31"), now),
            )

        elif event.event_type == "PROCEDURE":
            cur.execute(
                """
                INSERT INTO RT_PROCEDURES
                    (EVENT_ID, PATIENT_ID, ENCOUNTER_ID, CODE, DESCRIPTION,
                     REASONCODE, REASONDESCRIPTION, PROCEDURE_DATE, INGESTION_TS)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (event.event_id, event.patient_id,
                 p.get("ENCOUNTER_ID"),
                 p["CODE"], p["DESCRIPTION"],
                 p.get("REASONCODE", "N/A"),
                 p.get("REASONDESCRIPTION", "Unspecified"),
                 p["PROCEDURE_DATE"], now),
            )

        elif event.event_type == "IMMUNIZATION":
            cur.execute(
                """
                INSERT INTO RT_IMMUNIZATIONS
                    (EVENT_ID, PATIENT_ID, ENCOUNTER_ID, CODE, DESCRIPTION,
                     IMMUNIZATION_DATE, INGESTION_TS)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (event.event_id, event.patient_id,
                 p.get("ENCOUNTER_ID"),
                 p["CODE"], p["DESCRIPTION"],
                 p["IMMUNIZATION_DATE"], now),
            )

        elif event.event_type == "ALLERGY":
            cur.execute(
                """
                INSERT INTO RT_ALLERGIES
                    (EVENT_ID, PATIENT_ID, ENCOUNTER_ID, CODE, DESCRIPTION,
                     START_DATE, END_DATE, INGESTION_TS)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (event.event_id, event.patient_id,
                 p.get("ENCOUNTER_ID"),
                 p["CODE"], p["DESCRIPTION"],
                 p["START_DATE"],
                 p.get("END_DATE", "9999-12-31"), now),
            )

        elif event.event_type == "CAREPLAN":
            cp_id = p.get("CAREPLAN_ID") or str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO RT_CAREPLANS
                    (EVENT_ID, CAREPLAN_ID, PATIENT_ID, ENCOUNTER_ID, CODE, DESCRIPTION,
                     REASONCODE, REASONDESCRIPTION, START_DATE, END_DATE, INGESTION_TS)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (event.event_id, cp_id, event.patient_id,
                 p.get("ENCOUNTER_ID"),
                 p["CODE"], p["DESCRIPTION"],
                 p.get("REASONCODE", "N/A"),
                 p.get("REASONDESCRIPTION", "Unspecified"),
                 p["START_DATE"],
                 p.get("END_DATE", "9999-12-31"), now),
            )

        cur.close()
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Patient search
# ---------------------------------------------------------------------------

def _query_df(conn: snowflake.connector.SnowflakeConnection, sql: str,
              params=None) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=3600, show_spinner=False)
def search_patient_by_id(patient_id: str, cache_version: int = 0) -> Optional[pd.DataFrame]:
    """
    Look up a patient by exact PATIENT_ID from PATIENTS_GOLD.
    Returns a one-row DataFrame or None.
    """
    conn = _get_conn()
    try:
        sql = f"""
            SELECT * FROM {SCHEMA_GOLD}.PATIENTS_GOLD
            WHERE PATIENT_ID = %s
            LIMIT 1
        """
        df = _query_df(conn, sql, (patient_id,))
        return df if not df.empty else None
    finally:
        conn.close()


@st.cache_data(ttl=3600, show_spinner=False)
def search_patient_by_name(full_name: str, cache_version: int = 0) -> pd.DataFrame:
    """
    Search patients by FULL_NAME (case-insensitive, partial match).
    Returns a DataFrame (0 or more rows).
    """
    conn = _get_conn()
    try:
        sql = f"""
            SELECT * FROM {SCHEMA_GOLD}.PATIENTS_GOLD
            WHERE UPPER(FULL_NAME) ILIKE %s
            ORDER BY FULL_NAME
            LIMIT 50
        """
        return _query_df(conn, sql, (f"%{full_name.upper()}%",))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# STEP 8: Updated patient state (GOLD UNION REALTIME)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_patient_info(patient_id: str, cache_version: int = 0) -> Optional[dict]:
    """Return patient demographics from PATIENTS_GOLD."""
    conn = _get_conn()
    try:
        sql = f"""
            SELECT PATIENT_ID, BIRTHDATE, DEATHDATE, MARITAL, RACE,
                   ETHNICITY, GENDER, FULL_NAME
            FROM {SCHEMA_GOLD}.PATIENTS_GOLD
            WHERE PATIENT_ID = %s
            LIMIT 1
        """
        df = _query_df(conn, sql, (patient_id,))
        return df.iloc[0].to_dict() if not df.empty else None
    finally:
        conn.close()


@st.cache_data(ttl=600, show_spinner=False)
def get_encounters(patient_id: str, limit: int = 20, cache_version: int = 0) -> pd.DataFrame:
    """Return combined historical + real-time encounters for a patient."""
    conn = _get_conn()
    try:
        sql = f"""
            SELECT ENCOUNTER_ID, CODE, DESCRIPTION, REASONDESCRIPTION,
                   ENCOUNTER_DATE, 'historical' AS SOURCE
            FROM {SCHEMA_GOLD}.ENCOUNTERS_GOLD
            WHERE PATIENT_ID = %s
            UNION ALL
            SELECT ENCOUNTER_ID, CODE, DESCRIPTION, REASONDESCRIPTION,
                   ENCOUNTER_DATE, 'realtime' AS SOURCE
            FROM REALTIME.RT_ENCOUNTERS
            WHERE PATIENT_ID = %s
            ORDER BY ENCOUNTER_DATE DESC
            LIMIT {limit}
        """
        return _query_df(conn, sql, (patient_id, patient_id))
    finally:
        conn.close()


@st.cache_data(ttl=600, show_spinner=False)
def get_conditions(patient_id: str, cache_version: int = 0) -> pd.DataFrame:
    """Return combined historical + real-time conditions for a patient."""
    conn = _get_conn()
    try:
        sql = f"""
            SELECT CODE, DESCRIPTION, START_DATE, END_DATE, 'historical' AS SOURCE
            FROM {SCHEMA_GOLD}.CONDITIONS_GOLD
            WHERE PATIENT_ID = %s
            UNION ALL
            SELECT CODE, DESCRIPTION, START_DATE, END_DATE, 'realtime' AS SOURCE
            FROM REALTIME.RT_CONDITIONS
            WHERE PATIENT_ID = %s
            ORDER BY START_DATE DESC
        """
        return _query_df(conn, sql, (patient_id, patient_id))
    finally:
        conn.close()


@st.cache_data(ttl=600, show_spinner=False)
def get_medications(patient_id: str, cache_version: int = 0) -> pd.DataFrame:
    """Return combined historical + real-time medications for a patient."""
    conn = _get_conn()
    try:
        sql = f"""
            SELECT CODE, DESCRIPTION, REASONDESCRIPTION, START_DATE, END_DATE,
                   'historical' AS SOURCE
            FROM {SCHEMA_GOLD}.MEDICATIONS_GOLD
            WHERE PATIENT_ID = %s
            UNION ALL
            SELECT CODE, DESCRIPTION, REASONDESCRIPTION, START_DATE, END_DATE,
                   'realtime' AS SOURCE
            FROM REALTIME.RT_MEDICATIONS
            WHERE PATIENT_ID = %s
            ORDER BY START_DATE DESC
        """
        return _query_df(conn, sql, (patient_id, patient_id))
    finally:
        conn.close()


@st.cache_data(ttl=600, show_spinner=False)
def get_latest_observations(patient_id: str, limit: int = 10, cache_version: int = 0) -> pd.DataFrame:
    """
    Return the most recent observation rows from OBSERVATIONS_GOLD
    plus raw RT_OBSERVATIONS rows.

    Note: OBSERVATIONS_GOLD is a pivoted table (each measurement is a column).
    RT_OBSERVATIONS is in long/unpivoted form (DESCRIPTION, VALUE_NUMERIC).
    Both are returned; the caller decides how to display them.
    """
    conn = _get_conn()
    try:
        # Historical: pivoted Gold table (last N rows by date)
        sql_hist = f"""
            SELECT OBSERVATION_DATE,
                   "Body Height", "Body Mass Index", "Body Weight",
                   "Diastolic Blood Pressure", "GLUCOSE",
                   "Systolic Blood Pressure", "Total Cholesterol",
                   'historical' AS SOURCE
            FROM {SCHEMA_GOLD}.OBSERVATIONS_GOLD
            WHERE PATIENT_ID = %s
            ORDER BY OBSERVATION_DATE DESC
            LIMIT {limit}
        """
        hist = _query_df(conn, sql_hist, (patient_id,))

        # Real-time: unpivoted (long form)
        sql_rt = """
            SELECT OBSERVATION_DATE, DESCRIPTION, VALUE_NUMERIC, VALUE_TEXT,
                   'realtime' AS SOURCE
            FROM REALTIME.RT_OBSERVATIONS
            WHERE PATIENT_ID = %s
            ORDER BY OBSERVATION_DATE DESC
        """
        rt = _query_df(conn, sql_rt, (patient_id,))

        return hist, rt
    finally:
        conn.close()


@st.cache_data(ttl=600, show_spinner=False)
def get_allergies(patient_id: str, cache_version: int = 0) -> pd.DataFrame:
    """Return combined historical + real-time allergies."""
    conn = _get_conn()
    try:
        sql = f"""
            SELECT CODE, DESCRIPTION, START_DATE, END_DATE, 'historical' AS SOURCE
            FROM {SCHEMA_GOLD}.ALLERGIES_GOLD
            WHERE PATIENT_ID = %s
            UNION ALL
            SELECT CODE, DESCRIPTION, START_DATE, END_DATE, 'realtime' AS SOURCE
            FROM REALTIME.RT_ALLERGIES
            WHERE PATIENT_ID = %s
            ORDER BY START_DATE DESC
        """
        return _query_df(conn, sql, (patient_id, patient_id))
    finally:
        conn.close()


def patient_exists(patient_id: str) -> bool:
    """Check whether a patient_id exists in PATIENTS_GOLD."""
    conn = _get_conn()
    try:
        sql = f"""
            SELECT 1 FROM {SCHEMA_GOLD}.PATIENTS_GOLD
            WHERE PATIENT_ID = %s LIMIT 1
        """
        cur = conn.cursor()
        cur.execute(sql, (patient_id,))
        found = cur.fetchone() is not None
        cur.close()
        return found
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# STEP 11 UPGRADE: Unified clinical timeline (GOLD + REALTIME)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_timeline(patient_id: str, limit: int = 200, cache_version: int = 0) -> pd.DataFrame:
    """
    Return a merged, date-sorted clinical timeline for a patient.

    Combines all event types from GOLD and REALTIME into one DataFrame.

    Returns columns:
        EVENT_DATE, EVENT_TYPE, DESCRIPTION, DETAIL, SOURCE, ENCOUNTER_ID
    SOURCE is 'historical' or 'realtime'.
    """
    conn = _get_conn()
    frames = []
    
    # Define all queries clearly in a list of (SQL, params)
    g = SCHEMA_GOLD
    queries = [
        # --- HISTORICAL (GOLD) ---
        (f"""
            SELECT ENCOUNTER_DATE AS EVENT_DATE, 'ENCOUNTER' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(REASONDESCRIPTION,'') AS DETAIL,
                   'historical' AS SOURCE, ENCOUNTER_ID
            FROM {g}.ENCOUNTERS_GOLD WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        (f"""
            SELECT START_DATE AS EVENT_DATE, 'CONDITION' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(CODE,'') AS DETAIL,
                   'historical' AS SOURCE, NULL AS ENCOUNTER_ID
            FROM {g}.CONDITIONS_GOLD WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        (f"""
            SELECT START_DATE AS EVENT_DATE, 'MEDICATION' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(REASONDESCRIPTION,'') AS DETAIL,
                   'historical' AS SOURCE, NULL AS ENCOUNTER_ID
            FROM {g}.MEDICATIONS_GOLD WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        (f"""
            SELECT OBSERVATION_DATE AS EVENT_DATE, 'OBSERVATION' AS EVENT_TYPE,
                   'Vital Signs Reading' AS DESCRIPTION, '' AS DETAIL,
                   'historical' AS SOURCE, ENCOUNTER_ID
            FROM {g}.OBSERVATIONS_GOLD WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        (f"""
            SELECT PROCEDURE_DATE AS EVENT_DATE, 'PROCEDURE' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(REASONDESCRIPTION,'') AS DETAIL,
                   'historical' AS SOURCE, ENCOUNTER_ID
            FROM {g}.PROCEDURES_GOLD WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        (f"""
            SELECT IMMUNIZATION_DATE AS EVENT_DATE, 'IMMUNIZATION' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(CODE,'') AS DETAIL,
                   'historical' AS SOURCE, ENCOUNTER_ID
            FROM {g}.IMMUNIZATIONS_GOLD WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        (f"""
            SELECT START_DATE AS EVENT_DATE, 'ALLERGY' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(CODE,'') AS DETAIL,
                   'historical' AS SOURCE, ENCOUNTER_ID
            FROM {g}.ALLERGIES_GOLD WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        (f"""
            SELECT START_DATE AS EVENT_DATE, 'CAREPLAN' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(REASONDESCRIPTION,'') AS DETAIL,
                   'historical' AS SOURCE, ENCOUNTER_ID
            FROM {g}.CAREPLANS_GOLD WHERE PATIENT_ID = %s
        """, (patient_id,)),

        # --- REALTIME ---
        ("""
            SELECT ENCOUNTER_DATE AS EVENT_DATE, 'ENCOUNTER' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(REASONDESCRIPTION,'') AS DETAIL,
                   'realtime' AS SOURCE, ENCOUNTER_ID
            FROM REALTIME.RT_ENCOUNTERS WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        ("""
            SELECT START_DATE AS EVENT_DATE, 'CONDITION' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(CODE,'') AS DETAIL,
                   'realtime' AS SOURCE, ENCOUNTER_ID
            FROM REALTIME.RT_CONDITIONS WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        ("""
            SELECT START_DATE AS EVENT_DATE, 'MEDICATION' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(REASONDESCRIPTION,'') AS DETAIL,
                   'realtime' AS SOURCE, ENCOUNTER_ID
            FROM REALTIME.RT_MEDICATIONS WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        ("""
            SELECT OBSERVATION_DATE AS EVENT_DATE, 'OBSERVATION' AS EVENT_TYPE,
                   DESCRIPTION,
                   COALESCE(CAST(VALUE_NUMERIC AS VARCHAR), VALUE_TEXT, '') AS DETAIL,
                   'realtime' AS SOURCE, ENCOUNTER_ID
            FROM REALTIME.RT_OBSERVATIONS WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        ("""
            SELECT PROCEDURE_DATE AS EVENT_DATE, 'PROCEDURE' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(REASONDESCRIPTION,'') AS DETAIL,
                   'realtime' AS SOURCE, ENCOUNTER_ID
            FROM REALTIME.RT_PROCEDURES WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        ("""
            SELECT IMMUNIZATION_DATE AS EVENT_DATE, 'IMMUNIZATION' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(CODE,'') AS DETAIL,
                   'realtime' AS SOURCE, ENCOUNTER_ID
            FROM REALTIME.RT_IMMUNIZATIONS WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        ("""
            SELECT START_DATE AS EVENT_DATE, 'ALLERGY' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(CODE,'') AS DETAIL,
                   'realtime' AS SOURCE, ENCOUNTER_ID
            FROM REALTIME.RT_ALLERGIES WHERE PATIENT_ID = %s
        """, (patient_id,)),
        
        ("""
            SELECT START_DATE AS EVENT_DATE, 'CAREPLAN' AS EVENT_TYPE,
                   DESCRIPTION, COALESCE(REASONDESCRIPTION,'') AS DETAIL,
                   'realtime' AS SOURCE, ENCOUNTER_ID
            FROM REALTIME.RT_CAREPLANS WHERE PATIENT_ID = %s
        """, (patient_id,)),
    ]

    try:
        # Explicit loop over each query
        for sql, params in queries:
            try:
                df = _query_df(conn, sql, params)
                frames.append(df)
            except Exception as e:
                # If a specific table fails (e.g. doesn't exist yet), just skip it
                pass
    finally:
        conn.close()

    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(
            columns=["EVENT_DATE", "EVENT_TYPE", "DESCRIPTION", "DETAIL",
                     "SOURCE", "ENCOUNTER_ID"]
        )

    combined = pd.concat(non_empty, ignore_index=True)
    combined["EVENT_DATE"] = pd.to_datetime(combined["EVENT_DATE"], errors="coerce")
    combined = combined.dropna(subset=["EVENT_DATE"])
    combined = combined.sort_values("EVENT_DATE", ascending=False)
    combined = combined.head(limit).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("STEP 5/8 SELF-TEST — Patient State + Direct Snowflake Write")

    print("\n[1] Creating REALTIME schema and tables...")
    ensure_realtime_schema()
    print("    Done.")

    print("\n[2] Searching for a real patient by name...")
    results = search_patient_by_name("Smith")
    print(f"    Found {len(results)} patients matching 'Smith'")
    if not results.empty:
        pid = results.iloc[0]["PATIENT_ID"]
        print(f"    Using patient: {pid}")

        print("\n[3] Fetching patient info...")
        info = get_patient_info(pid)
        print(f"    {info}")

        print("\n[4] Fetching encounters (top 5)...")
        enc = get_encounters(pid, limit=5)
        print(enc[["ENCOUNTER_DATE","DESCRIPTION","SOURCE"]].to_string())

        print("\n[5] Direct-inserting a test OBSERVATION event...")
        from event_schema import MedicalEvent
        test_event = MedicalEvent(
            patient_id=pid,
            event_type="OBSERVATION",
            payload={
                "DESCRIPTION":      "Body Weight",
                "OBSERVATION_DATE": "2026-09-01",
                "VALUE_NUMERIC":    75.0,
            },
        )
        insert_event_direct(test_event)
        print("    Inserted successfully.")

        print("\n[6] Verifying RT_OBSERVATIONS row count for patient...")
        conn2 = _get_conn()
        cur2 = conn2.cursor()
        cur2.execute(
            "SELECT COUNT(*) FROM REALTIME.RT_OBSERVATIONS WHERE PATIENT_ID = %s",
            (pid,)
        )
        cnt = cur2.fetchone()[0]
        cur2.close()
        conn2.close()
        print(f"    RT_OBSERVATIONS rows for patient: {cnt}")

    print("\nSTEP 5/8 DONE")

