# System Architecture

## Overview
The system employs a Snowflake data warehouse structured with a medallion architecture (Bronze, Silver, Gold). Real-time events are ingested through a Kafka message broker and processed via Spark Structured Streaming, complementing a PySpark batch ETL pipeline.

## Key Components

- **Snowflake Data Warehouse (Medallion Architecture)**
  - Bronze, Silver, Gold schemas for batch data
  - REALTIME schema with append-only `RT_*` tables for real-time events
- **PySpark Batch ETL Pipeline**: Processes bulk historical and batch data into the medallion schemas.
- **Kafka Message Broker**: Handles real-time event streaming.
- **Spark Structured Streaming Consumer**: Consumes real-time events from Kafka and writes to Snowflake.
- **Unified Patient State**: Computed as `GOLD UNION REALTIME` for a complete view.
- **Feature Engine**: Computes 18 ML features from the unified patient state.
- **Machine Learning**: 5 XGBoost models for disease risk prediction.
- **Streamlit Dashboard**: Provides the user interface for monitoring and interacting with the system.

## Architecture Diagram

```mermaid
flowchart TD
    User([User]) --> Streamlit[Streamlit Dashboard]
    Streamlit --> Kafka[Kafka Broker: medical-events]
    
    subgraph Batch Data Pipeline
        BatchSource[(Batch Data)] --> PySparkETL[PySpark ETL Pipeline]
        PySparkETL --> SnowflakeBronze[(Snowflake Bronze)]
        SnowflakeBronze --> SnowflakeSilver[(Snowflake Silver)]
        SnowflakeSilver --> SnowflakeGold[(Snowflake Gold)]
    end

    subgraph Real-Time Pipeline
        Kafka --> SparkStreaming[Spark Structured Streaming]
        SparkStreaming --> SnowflakeRealtime[(Snowflake REALTIME)]
    end

    SnowflakeGold --> UnifiedState{Patient State:\nGOLD U REALTIME}
    SnowflakeRealtime --> UnifiedState
    
    UnifiedState --> FeatureEngine[Feature Engine\n(18 Features)]
    FeatureEngine --> MLModels[5 XGBoost Models]
    MLModels --> Streamlit
```
