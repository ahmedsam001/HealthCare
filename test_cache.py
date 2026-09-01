import sys, os, time
sys.path.insert(0, os.path.abspath("realtime"))
import streamlit as st
from realtime.patient_state import search_patient_by_name, get_patient_info, get_encounters

# Need a streamlit context to use cache? Streamlit caches work globally in 1.35+.
t0 = time.time()
pid = search_patient_by_name("Smith", cache_version=0).iloc[0]["PATIENT_ID"]
t1 = time.time()
print(f"search_patient_by_name (run 1): {t1-t0:.3f}s")

t0 = time.time()
search_patient_by_name("Smith", cache_version=0)
t1 = time.time()
print(f"search_patient_by_name (run 2): {t1-t0:.3f}s")

t0 = time.time()
get_encounters(pid, cache_version=0)
t1 = time.time()
print(f"get_encounters (run 1): {t1-t0:.3f}s")

t0 = time.time()
get_encounters(pid, cache_version=0)
t1 = time.time()
print(f"get_encounters (run 2): {t1-t0:.3f}s")

t0 = time.time()
get_encounters(pid, cache_version=1)
t1 = time.time()
print(f"get_encounters (run 3 with busted cache): {t1-t0:.3f}s")

