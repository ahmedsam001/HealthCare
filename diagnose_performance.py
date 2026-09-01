import os
import sys
import time
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.abspath("realtime"))

# Imports
from realtime.patient_state import (
    search_patient_by_name, get_patient_info, get_encounters,
    get_conditions, get_medications, get_latest_observations,
    get_allergies, get_timeline
)
from realtime.feature_engine import compute_features
from realtime.inference import compute_risk
from realtime.ml_contract import load_model, run_all_targets, TARGETS
from realtime.config import snowflake_connect_kwargs
import snowflake.connector

def log(layer, func, event, elapsed=None):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if elapsed is not None:
        print(f"[{ts}] [{layer}] [{func}] {event} elapsed={elapsed:.3f}s")
    else:
        print(f"[{ts}] [{layer}] [{func}] {event}")

def measure(layer, func_name, func, *args, **kwargs):
    log(layer, func_name, "START")
    t0 = time.time()
    res = func(*args, **kwargs)
    t1 = time.time()
    log(layer, func_name, "END", elapsed=(t1 - t0))
    return res, (t1 - t0)

def main():
    log("SYS", "main", "Starting diagnostic script")
    
    # Get a patient
    t0 = time.time()
    df = search_patient_by_name("Smith")
    t1 = time.time()
    log("DB", "search_patient_by_name", "END", elapsed=(t1-t0))
    
    if df.empty:
        print("No patient found, aborting.")
        return
        
    pid = df.iloc[0]["PATIENT_ID"]
    print(f"\n--- Diagnosing for patient {pid} ---\n")

    # 1. Simulate Patient Overview (Tab 1)
    print("=== TAB 1: Patient Overview (Simulated Rerun) ===")
    t_tab1_start = time.time()
    _, t_info = measure("DB", "get_patient_info", get_patient_info, pid)
    _, t_enc = measure("DB", "get_encounters", get_encounters, pid, limit=1000)
    _, t_cond = measure("DB", "get_conditions", get_conditions, pid)
    _, t_med = measure("DB", "get_medications", get_medications, pid)
    _, t_obs = measure("DB", "get_latest_observations", get_latest_observations, pid)
    _, t_alg = measure("DB", "get_allergies", get_allergies, pid)
    t_tab1_end = time.time()
    log("UI", "render_patient_overview", "TOTAL", elapsed=(t_tab1_end - t_tab1_start))
    
    # 2. Simulate Clinical Timeline (Tab 3)
    print("\n=== TAB 3: Clinical Timeline ===")
    t_tab3_start = time.time()
    _, t_timeline = measure("DB", "get_timeline", get_timeline, pid, limit=200)
    t_tab3_end = time.time()
    log("UI", "render_clinical_timeline", "TOTAL", elapsed=(t_tab3_end - t_tab3_start))

    # 3. Simulate Feature Engine
    print("\n=== FEATURE ENGINE (compute_features) ===")
    _, t_feats = measure("ML", "compute_features", compute_features, pid)

    # 4. Simulate ML Inference
    print("\n=== ML INFERENCE (models) ===")
    # First, measure loading time for one model
    t_load_start = time.time()
    model = load_model(TARGETS[0])
    t_load_end = time.time()
    log("ML", "load_model_single", "END", elapsed=(t_load_end - t_load_start))
    
    # Measure run_all_targets (using features just computed to avoid DB hits here)
    features = compute_features(pid) # do it again to get the dict cleanly, we already timed it
    t_run_all_start = time.time()
    run_all_targets(features)
    t_run_all_end = time.time()
    log("ML", "run_all_targets", "END", elapsed=(t_run_all_end - t_run_all_start))

    # 5. Full Risk Computation (compute_risk = feature_engine + inference)
    print("\n=== RISK COMPUTATION (compute_risk) ===")
    _, t_risk = measure("ML", "compute_risk", compute_risk, pid)

    # 6. Simulate System Health (Tab 5)
    print("\n=== TAB 5: System Health ===")
    t_sf_start = time.time()
    try:
        conn = snowflake.connector.connect(**snowflake_connect_kwargs())
        conn.close()
    except Exception:
        pass
    t_sf_end = time.time()
    log("DB", "snowflake_connect_test", "END", elapsed=(t_sf_end - t_sf_start))
    
    print("\n=== DIAGNOSTICS COMPLETE ===")

if __name__ == "__main__":
    main()

    print("\n=== SIMULATING STREAMLIT RERUN ===")
    
    # 1. Simulate Patient Overview (Tab 1)
    print("=== TAB 1: Patient Overview ===")
    t_tab1_start = time.time()
    _, t_info = measure("DB", "get_patient_info", get_patient_info, pid)
    _, t_enc = measure("DB", "get_encounters", get_encounters, pid, limit=1000)
    _, t_cond = measure("DB", "get_conditions", get_conditions, pid)
    _, t_med = measure("DB", "get_medications", get_medications, pid)
    _, t_obs = measure("DB", "get_latest_observations", get_latest_observations, pid)
    _, t_alg = measure("DB", "get_allergies", get_allergies, pid)
    t_tab1_end = time.time()
    log("UI", "render_patient_overview", "TOTAL", elapsed=(t_tab1_end - t_tab1_start))
    
    # 2. Simulate Clinical Timeline (Tab 3)
    print("\n=== TAB 3: Clinical Timeline ===")
    t_tab3_start = time.time()
    _, t_timeline = measure("DB", "get_timeline", get_timeline, pid, limit=200)
    t_tab3_end = time.time()
    log("UI", "render_clinical_timeline", "TOTAL", elapsed=(t_tab3_end - t_tab3_start))

    print("\n=== AFTER DIAGNOSTICS COMPLETE ===")

