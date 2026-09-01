import re

with open('realtime/app.py', 'r') as f:
    content = f.read()

# Add fragment to render_risk_assessment
content = content.replace("def render_risk_assessment(patient_id: str):", "@st.fragment\ndef render_risk_assessment(patient_id: str):")

# Add fragment to render_live_event
content = content.replace("def render_live_event(patient_id: str):", "@st.fragment\ndef render_live_event(patient_id: str):")

# In render_live_event, after _bump_db_version(), add st.rerun()
# Wait, it currently looks like:
#         if ok:
#             _bump_db_version()
content = re.sub(
    r'(if ok:\n\s+_bump_db_version\(\))',
    r'\1\n            st.rerun()',
    content
)

with open('realtime/app.py', 'w') as f:
    f.write(content)

print("Added fragments to app.py successfully.")
