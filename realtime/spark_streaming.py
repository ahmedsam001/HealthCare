"""
# STEP 7: Consume and process medical events using Spark Structured Streaming.
#
# Pipeline:
#   Kafka (medical-events topic)
#       ↓
#   Spark Structured Streaming
#       ↓
#   Parse JSON (event envelope)
#       ↓
#   Validate event_type + required fields
#       ↓
#   Transform payload → match REALTIME table column names
#       ↓
#   Write to Snowflake REALTIME.RT_<TYPE> (append mode, foreachBatch)
#
# The Spark-Snowflake connector for PySpark 3.5:
# net.snowflake:spark-snowflake_2.12:3.2.1-spark_3.5
#
# Run:
#   python spark_streaming.py
"""

from __future__ import annotations

import json
import os
import sys

# Must be set before SparkSession is created
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages "
    "net.snowflake:snowflake-jdbc:4.0.2,"
    "net.snowflake:spark-snowflake_2.12:3.2.1-spark_3.5,"
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6 "
    "pyspark-shell"
)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, from_json, get_json_object, lit, current_timestamp,
    coalesce, to_timestamp, to_date, expr
)
from pyspark.sql.types import (
    StructType, StructField, StringType, MapType
)

from config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC,
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
    SNOWFLAKE_DATABASE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE,
)


# ---------------------------------------------------------------------------
# Snowflake options (same pattern as existing notebooks)
# ---------------------------------------------------------------------------

SF_OPTIONS = {
    "sfURL":      f"{SNOWFLAKE_ACCOUNT}.snowflakecomputing.com",
    "sfUser":     SNOWFLAKE_USER,
    "sfPassword": SNOWFLAKE_PASSWORD,
    "sfDatabase": SNOWFLAKE_DATABASE,
    "sfWarehouse": SNOWFLAKE_WAREHOUSE,
    "sfRole":     SNOWFLAKE_ROLE,
    "sfSchema":   "REALTIME",
}

SNOWFLAKE_FORMAT = "net.snowflake.spark.snowflake"

# ---------------------------------------------------------------------------
# Kafka message schema
# The outer envelope fields; payload is a nested JSON string
# ---------------------------------------------------------------------------

ENVELOPE_SCHEMA = StructType([
    StructField("event_id",        StringType(), True),
    StructField("patient_id",      StringType(), True),
    StructField("event_type",      StringType(), True),
    StructField("event_timestamp", StringType(), True),
    StructField("payload",         StringType(), True),  # raw JSON string
])

SUPPORTED_TYPES = {
    "ENCOUNTER", "OBSERVATION", "CONDITION", "MEDICATION",
    "PROCEDURE", "IMMUNIZATION", "ALLERGY", "CAREPLAN",
}

# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("HealthcareRealTimeStreaming")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/hc_streaming_ckpt")
        .getOrCreate()
    )

# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_to_snowflake(df: DataFrame, table_name: str) -> None:
    """Append a micro-batch DataFrame to a REALTIME Snowflake table."""
    if df.isEmpty():
        return
    (
        df.write
        .format(SNOWFLAKE_FORMAT)
        .options(**SF_OPTIONS)
        .option("dbtable", table_name)
        .mode("append")
        .save()
    )


# ---------------------------------------------------------------------------
# Per-event-type parsers
# Each function receives the parsed envelope DataFrame and returns a
# typed DataFrame ready for Snowflake.
# ---------------------------------------------------------------------------

def _parse_encounters(df: DataFrame) -> DataFrame:
    return df.select(
        col("event_id"),
        col("patient_id"),
        coalesce(
            get_json_object(col("payload"), "$.ENCOUNTER_ID"),
            col("event_id")
        ).alias("ENCOUNTER_ID"),
        get_json_object(col("payload"), "$.CODE").alias("CODE"),
        get_json_object(col("payload"), "$.DESCRIPTION").alias("DESCRIPTION"),
        coalesce(
            get_json_object(col("payload"), "$.REASONCODE"), lit("N/A")
        ).alias("REASONCODE"),
        coalesce(
            get_json_object(col("payload"), "$.REASONDESCRIPTION"),
            lit("Routine / Unspecified")
        ).alias("REASONDESCRIPTION"),
        to_timestamp(
            get_json_object(col("payload"), "$.ENCOUNTER_DATE")
        ).alias("ENCOUNTER_DATE"),
        current_timestamp().alias("INGESTION_TS"),
    ).filter(col("CODE").isNotNull())


def _parse_observations(df: DataFrame) -> DataFrame:
    return df.select(
        col("event_id"),
        col("patient_id"),
        get_json_object(col("payload"), "$.ENCOUNTER_ID").alias("ENCOUNTER_ID"),
        get_json_object(col("payload"), "$.DESCRIPTION").alias("DESCRIPTION"),
        to_date(
            get_json_object(col("payload"), "$.OBSERVATION_DATE")
        ).alias("OBSERVATION_DATE"),
        get_json_object(col("payload"), "$.VALUE_NUMERIC").cast("double").alias("VALUE_NUMERIC"),
        get_json_object(col("payload"), "$.VALUE_TEXT").alias("VALUE_TEXT"),
        coalesce(
            get_json_object(col("payload"), "$.READING_SEQ").cast("int"), lit(1)
        ).alias("READING_SEQ"),
        current_timestamp().alias("INGESTION_TS"),
    ).filter(col("DESCRIPTION").isNotNull())


def _parse_conditions(df: DataFrame) -> DataFrame:
    return df.select(
        col("event_id"),
        col("patient_id"),
        get_json_object(col("payload"), "$.ENCOUNTER_ID").alias("ENCOUNTER_ID"),
        get_json_object(col("payload"), "$.CODE").alias("CODE"),
        get_json_object(col("payload"), "$.DESCRIPTION").alias("DESCRIPTION"),
        to_timestamp(
            get_json_object(col("payload"), "$.START_DATE")
        ).alias("START_DATE"),
        coalesce(
            to_timestamp(get_json_object(col("payload"), "$.END_DATE")),
            lit("9999-12-31").cast("timestamp")
        ).alias("END_DATE"),
        current_timestamp().alias("INGESTION_TS"),
    ).filter(col("CODE").isNotNull())


def _parse_medications(df: DataFrame) -> DataFrame:
    return df.select(
        col("event_id"),
        col("patient_id"),
        get_json_object(col("payload"), "$.ENCOUNTER_ID").alias("ENCOUNTER_ID"),
        get_json_object(col("payload"), "$.CODE").alias("CODE"),
        get_json_object(col("payload"), "$.DESCRIPTION").alias("DESCRIPTION"),
        coalesce(
            get_json_object(col("payload"), "$.REASONCODE"), lit("N/A")
        ).alias("REASONCODE"),
        coalesce(
            get_json_object(col("payload"), "$.REASONDESCRIPTION"), lit("Unspecified")
        ).alias("REASONDESCRIPTION"),
        to_timestamp(
            get_json_object(col("payload"), "$.START_DATE")
        ).alias("START_DATE"),
        coalesce(
            to_timestamp(get_json_object(col("payload"), "$.END_DATE")),
            lit("9999-12-31").cast("timestamp")
        ).alias("END_DATE"),
        current_timestamp().alias("INGESTION_TS"),
    ).filter(col("CODE").isNotNull())


def _parse_procedures(df: DataFrame) -> DataFrame:
    return df.select(
        col("event_id"),
        col("patient_id"),
        get_json_object(col("payload"), "$.ENCOUNTER_ID").alias("ENCOUNTER_ID"),
        get_json_object(col("payload"), "$.CODE").alias("CODE"),
        get_json_object(col("payload"), "$.DESCRIPTION").alias("DESCRIPTION"),
        coalesce(
            get_json_object(col("payload"), "$.REASONCODE"), lit("N/A")
        ).alias("REASONCODE"),
        coalesce(
            get_json_object(col("payload"), "$.REASONDESCRIPTION"), lit("Unspecified")
        ).alias("REASONDESCRIPTION"),
        to_timestamp(
            get_json_object(col("payload"), "$.PROCEDURE_DATE")
        ).alias("PROCEDURE_DATE"),
        current_timestamp().alias("INGESTION_TS"),
    ).filter(col("CODE").isNotNull())


def _parse_immunizations(df: DataFrame) -> DataFrame:
    return df.select(
        col("event_id"),
        col("patient_id"),
        get_json_object(col("payload"), "$.ENCOUNTER_ID").alias("ENCOUNTER_ID"),
        get_json_object(col("payload"), "$.CODE").alias("CODE"),
        get_json_object(col("payload"), "$.DESCRIPTION").alias("DESCRIPTION"),
        to_timestamp(
            get_json_object(col("payload"), "$.IMMUNIZATION_DATE")
        ).alias("IMMUNIZATION_DATE"),
        current_timestamp().alias("INGESTION_TS"),
    ).filter(col("CODE").isNotNull())


def _parse_allergies(df: DataFrame) -> DataFrame:
    return df.select(
        col("event_id"),
        col("patient_id"),
        get_json_object(col("payload"), "$.ENCOUNTER_ID").alias("ENCOUNTER_ID"),
        get_json_object(col("payload"), "$.CODE").alias("CODE"),
        get_json_object(col("payload"), "$.DESCRIPTION").alias("DESCRIPTION"),
        to_timestamp(
            get_json_object(col("payload"), "$.START_DATE")
        ).alias("START_DATE"),
        coalesce(
            to_timestamp(get_json_object(col("payload"), "$.END_DATE")),
            lit("9999-12-31").cast("timestamp")
        ).alias("END_DATE"),
        current_timestamp().alias("INGESTION_TS"),
    ).filter(col("CODE").isNotNull())


def _parse_careplans(df: DataFrame) -> DataFrame:
    return df.select(
        col("event_id"),
        coalesce(
            get_json_object(col("payload"), "$.CAREPLAN_ID"),
            col("event_id")
        ).alias("CAREPLAN_ID"),
        col("patient_id"),
        get_json_object(col("payload"), "$.ENCOUNTER_ID").alias("ENCOUNTER_ID"),
        get_json_object(col("payload"), "$.CODE").alias("CODE"),
        get_json_object(col("payload"), "$.DESCRIPTION").alias("DESCRIPTION"),
        coalesce(
            get_json_object(col("payload"), "$.REASONCODE"), lit("N/A")
        ).alias("REASONCODE"),
        coalesce(
            get_json_object(col("payload"), "$.REASONDESCRIPTION"), lit("Unspecified")
        ).alias("REASONDESCRIPTION"),
        to_timestamp(
            get_json_object(col("payload"), "$.START_DATE")
        ).alias("START_DATE"),
        coalesce(
            to_timestamp(get_json_object(col("payload"), "$.END_DATE")),
            lit("9999-12-31").cast("timestamp")
        ).alias("END_DATE"),
        current_timestamp().alias("INGESTION_TS"),
    ).filter(col("CODE").isNotNull())


PARSERS = {
    "ENCOUNTER":    (_parse_encounters,   "RT_ENCOUNTERS"),
    "OBSERVATION":  (_parse_observations, "RT_OBSERVATIONS"),
    "CONDITION":    (_parse_conditions,   "RT_CONDITIONS"),
    "MEDICATION":   (_parse_medications,  "RT_MEDICATIONS"),
    "PROCEDURE":    (_parse_procedures,   "RT_PROCEDURES"),
    "IMMUNIZATION": (_parse_immunizations,"RT_IMMUNIZATIONS"),
    "ALLERGY":      (_parse_allergies,    "RT_ALLERGIES"),
    "CAREPLAN":     (_parse_careplans,    "RT_CAREPLANS"),
}


# ---------------------------------------------------------------------------
# foreachBatch handler
# ---------------------------------------------------------------------------

def _process_batch(batch_df: DataFrame, batch_id: int) -> None:
    """
    Process one micro-batch from Kafka.
    Splits by event_type and writes each subset to the matching RT_ table.
    """
    if batch_df.isEmpty():
        return

    print(f"[Streaming] Processing batch_id={batch_id} ...")

    for event_type, (parser_fn, table_name) in PARSERS.items():
        subset = batch_df.filter(col("event_type") == event_type)
        parsed = parser_fn(subset)
        count  = parsed.count()
        if count > 0:
            print(f"  → {event_type}: {count} rows → {table_name}")
            _write_to_snowflake(parsed, table_name)

    print(f"[Streaming] Batch {batch_id} complete.")


# ---------------------------------------------------------------------------
# Main streaming job
# ---------------------------------------------------------------------------

def run_streaming() -> None:
    """Start and block on the Spark Structured Streaming job."""
    spark = create_spark()
    print(f"[Streaming] Spark {spark.version} started.")
    print(f"[Streaming] Reading from Kafka: {KAFKA_BOOTSTRAP_SERVERS} / {KAFKA_TOPIC}")

    # Read from Kafka
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse the JSON value
    parsed_stream = (
        raw_stream
        .select(
            from_json(col("value").cast("string"), ENVELOPE_SCHEMA).alias("env")
        )
        .select("env.*")
        # Discard unknown event types early
        .filter(col("event_type").isin(*SUPPORTED_TYPES))
        # Discard events with no patient or event id
        .filter(col("patient_id").isNotNull())
        .filter(col("event_id").isNotNull())
    )

    # Write via foreachBatch to Snowflake
    query = (
        parsed_stream.writeStream
        .foreachBatch(_process_batch)
        .option("checkpointLocation", "/tmp/hc_streaming_ckpt")
        .trigger(processingTime="10 seconds")  # micro-batch every 10 s
        .start()
    )

    print("[Streaming] Streaming query started. Waiting for data...")
    query.awaitTermination()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not SNOWFLAKE_PASSWORD:
        print("ERROR: SNOWFLAKE_PASSWORD environment variable is not set.")
        print("Please set it using: export SNOWFLAKE_PASSWORD=\"...\"")
        sys.exit(1)

    print("STEP 7 — Spark Structured Streaming Consumer")
    run_streaming()

