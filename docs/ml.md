# Machine Learning

This document covers the machine learning models and features used for disease risk prediction.

## Target Diseases
The system predicts the risk for 5 target diseases:
- Diabetes
- Hypertension
- Coronary Heart Disease
- Stroke
- Asthma

## Training Data
- **Source**: Synthea SyntheticMass
- **Location**: `ML.PATIENT_FEATURES_TARGETS_LONG` in Snowflake.

## Features (18 Total)

### Numeric Features (14)
- `AGE_AT_INDEX`
- `AVG_HEIGHT`
- `AVG_BMI`
- `AVG_WEIGHT`
- `AVG_DIASTOLIC_BP`
- `AVG_GLUCOSE`
- `AVG_SYSTOLIC_BP`
- `AVG_CHOLESTEROL`
- `OBSERVATION_COUNT`
- `DISTINCT_CONDITION_COUNT`
- `DISTINCT_MEDICATION_COUNT`
- `TOTAL_ENCOUNTERS`
- `HAS_GLUCOSE_READING`
- `HAS_CHOLESTEROL_READING`

### Categorical Features (4)
- `GENDER`
- `RACE`
- `ETHNICITY`
- `MARITAL`

## Model Architecture
- **Pipeline**: scikit-learn `Pipeline`
- **Preprocessing**: `ColumnTransformer` featuring `SimpleImputer` and `OneHotEncoder`
- **Algorithm**: `XGBClassifier`

## Training
- **Class Imbalance**: Handled dynamically using `scale_pos_weight`.
- **Threshold Tuning**: Optimized via Precision-Recall (PR) curve analysis.

## Model Artifacts
Models are saved as:
- `models/<disease>/model.joblib`
- `models/<disease>/metadata.json`

## Inference Workflow
1. `feature_engine.py` computes the 18 features from the unified patient state.
2. `ml_contract.py` loads the saved XGBoost models and metadata.
3. `inference.py` orchestrates the scoring process.

## Risk Display
The predicted probability is mapped to risk labels for the UI:
- **Low**: < 0.4
- **Moderate**: 0.4 - 0.7
- **High**: >= 0.7
