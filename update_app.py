import re

with open('realtime/app.py', 'r') as f:
    content = f.read()

# 1. Add _get_db_version() helper
db_version_code = """
def _get_db_version():
    return st.session_state.get("db_version", 0)

def _bump_db_version():
    st.session_state["db_version"] = _get_db_version() + 1
"""
content = content.replace("# ---------------------------------------------------------------------------", "# ---------------------------------------------------------------------------\n" + db_version_code, 1)

# 2. Inject cache_version=_get_db_version() into calls
replacements = [
    (r'get_patient_info\(patient_id\)', r'get_patient_info(patient_id, cache_version=_get_db_version())'),
    (r'get_encounters\(patient_id, limit=1000\)', r'get_encounters(patient_id, limit=1000, cache_version=_get_db_version())'),
    (r'get_conditions\(patient_id\)', r'get_conditions(patient_id, cache_version=_get_db_version())'),
    (r'get_medications\(patient_id\)', r'get_medications(patient_id, cache_version=_get_db_version())'),
    (r'get_latest_observations\(patient_id\)', r'get_latest_observations(patient_id, cache_version=_get_db_version())'),
    (r'get_allergies\(patient_id\)', r'get_allergies(patient_id, cache_version=_get_db_version())'),
    (r'get_timeline\(patient_id, limit=200\)', r'get_timeline(patient_id, limit=200, cache_version=_get_db_version())'),
]

for old, new in replacements:
    content = re.sub(old, new, content)
    
# For search functions
content = re.sub(r'search_patient_by_id\(pid\)', r'search_patient_by_id(pid, cache_version=_get_db_version())', content)
content = re.sub(r'search_patient_by_name\(name_input\.strip\(\)\)', r'search_patient_by_name(name_input.strip(), cache_version=_get_db_version())', content)

# 3. Bump db_version on successful event insert
content = re.sub(
    r'(st\.session_state\["last_event"\] = \{[^\}]+\})',
    r'\1\n\n        if ok:\n            _bump_db_version()',
    content
)

# 4. Cache System Health Snowflake connection test
health_cache_code = """
@st.cache_data(ttl=60, show_spinner=False)
def _test_snowflake_connection():
    from config import snowflake_connect_kwargs
    import snowflake.connector
    try:
        conn = snowflake.connector.connect(**snowflake_connect_kwargs())
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)
"""

content = content.replace("def render_system_health():", health_cache_code + "\ndef render_system_health():")

# Replace sidebar snowflake test
sidebar_old = """    try:
        from config import snowflake_connect_kwargs
        import snowflake.connector
        conn = snowflake.connector.connect(**snowflake_connect_kwargs())
        conn.close()
        st.sidebar.markdown("**Snowflake** &nbsp; 🟢 Connected")
    except Exception:
        st.sidebar.markdown("**Snowflake** &nbsp; 🔴 Error")"""

sidebar_new = """    sf_ok, sf_err = _test_snowflake_connection()
    if sf_ok:
        st.sidebar.markdown("**Snowflake** &nbsp; 🟢 Connected")
    else:
        st.sidebar.markdown("**Snowflake** &nbsp; 🔴 Error")"""

content = content.replace(sidebar_old, sidebar_new)

# Replace main tab snowflake test
tab5_old = """    try:
        from config import snowflake_connect_kwargs, SNOWFLAKE_ACCOUNT
        import snowflake.connector
        conn = snowflake.connector.connect(**snowflake_connect_kwargs())
        conn.close()
        col2.metric("Snowflake", "🟢 Connected", SNOWFLAKE_ACCOUNT)
    except Exception as e:
        col2.metric("Snowflake", "🔴 Error", str(e)[:40])"""

tab5_new = """    from config import SNOWFLAKE_ACCOUNT
    sf_ok, sf_err = _test_snowflake_connection()
    if sf_ok:
        col2.metric("Snowflake", "🟢 Connected", SNOWFLAKE_ACCOUNT)
    else:
        col2.metric("Snowflake", "🔴 Error", sf_err[:40])"""

content = content.replace(tab5_old, tab5_new)

with open('realtime/app.py', 'w') as f:
    f.write(content)

print("Updated app.py successfully.")
