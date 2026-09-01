import re

with open('realtime/patient_state.py', 'r') as f:
    content = f.read()

# Add import if missing
if 'import streamlit as st' not in content:
    content = 'import streamlit as st\n' + content

# Functions to update
funcs = [
    'search_patient_by_id',
    'search_patient_by_name',
    'get_patient_info',
    'get_encounters',
    'get_conditions',
    'get_medications',
    'get_latest_observations',
    'get_allergies',
    'get_timeline'
]

for func in funcs:
    # Match def func_name(args) -> ret:
    pattern = r'def ' + func + r'\((.*?)\)( -> [^:]+)?:'
    
    def repl(m):
        args = m.group(1)
        ret = m.group(2) or ''
        # Add cache_version if not present
        if 'cache_version' not in args:
            if args:
                args += ', cache_version: int = 0'
            else:
                args = 'cache_version: int = 0'
        
        # Decide TTL
        ttl = 3600 if func.startswith('search_') else 600
        
        return f'@st.cache_data(ttl={ttl}, show_spinner=False)\ndef {func}({args}){ret}:'
        
    content = re.sub(pattern, repl, content, count=1)

with open('realtime/patient_state.py', 'w') as f:
    f.write(content)

print("Updated patient_state.py successfully.")
