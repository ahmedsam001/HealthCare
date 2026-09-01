"""
Healthcare Real-Time Risk Monitor — Professional Streamlit Dashboard
====================================================================

Complete flow:
  Patient Search → Patient 360 → Risk Assessment (before/after)
      → Clinical Timeline → Live Event → System Health

Run from inside the realtime/ directory:
    streamlit run app.py

Environment variables:
    SNOWFLAKE_PASSWORD        (required)
    KAFKA_BOOTSTRAP_SERVERS   (optional, default: localhost:9092)
    MODEL_DIR                 (optional, default: ../saved_models)
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------

def _get_db_version():
    return st.session_state.get("db_version", 0)

def _bump_db_version():
    st.session_state["db_version"] = _get_db_version() + 1

# Page config — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Risk Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports from realtime/ package (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, TARGETS
from event_schema import (
    MedicalEvent, validate_event, ValidationError,
    SUPPORTED_EVENT_TYPES, REQUIRED_PAYLOAD_FIELDS,
)
from patient_state import (
    ensure_realtime_schema,
    search_patient_by_id,
    search_patient_by_name,
    get_patient_info,
    get_encounters,
    get_conditions,
    get_medications,
    get_latest_observations,
    get_allergies,
    get_timeline,
    insert_event_direct,
    patient_exists,
)
from inference import compute_risk
from ml_contract import models_available

# ---------------------------------------------------------------------------
# CSS — professional healthcare dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] { font-family: 'Inter', 'Helvetica Neue', sans-serif; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #1e2533;
    border: 1px solid #2e3a4e;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 4px 0;
}
div[data-testid="metric-container"] label {
    color: #8fa3bf !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-testid="metric-container"] div[data-testid="metric-value"] {
    font-size: 1.45rem !important;
    font-weight: 700;
    color: #e8edf4 !important;
}

/* Tab styling */
button[data-baseweb="tab"] { font-size: 0.88rem; padding: 10px 18px; }
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #3b82f6;
    color: #3b82f6;
}

/* Timeline card */
.timeline-card {
    border-left: 4px solid #3b82f6;
    padding: 10px 16px;
    margin: 8px 0;
    background: #1a2235;
    border-radius: 0 8px 8px 0;
}
.timeline-card.realtime { border-left-color: #f59e0b; background: #1f1a0e; }
.timeline-badge {
    display: inline-block; font-size: 0.72rem; font-weight: 700;
    padding: 2px 8px; border-radius: 12px; text-transform: uppercase;
    margin-right: 6px;
}
.badge-historical { background: #1e3a5f; color: #60a5fa; }
.badge-realtime   { background: #5c3a00; color: #f59e0b; }

/* Risk bar */
.risk-bar-wrap { width:100%; background:#2e3a4e; border-radius:6px; height:10px; }
.risk-bar-fill  { height:10px; border-radius:6px; transition: width 0.4s ease; }

/* Status pills */
.pill { display:inline-block; padding:3px 10px; border-radius:12px;
        font-size:0.75rem; font-weight:600; margin:2px; }
.pill-ok   { background:#0d4429; color:#34d399; }
.pill-warn { background:#451a03; color:#fbbf24; }
.pill-err  { background:#450a0a; color:#f87171; }
.pill-gray { background:#1e2533; color:#94a3b8; }

/* Section header */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #93c5fd;
    border-bottom: 1px solid #2e3a4e; padding-bottom: 6px; margin: 20px 0 12px;
    text-transform: uppercase; letter-spacing: 0.06em;
}

/* Validation row */
.val-row { display:flex; align-items:center; gap:8px;
           padding:5px 0; border-bottom:1px solid #1e2533; font-size:0.86rem; }
.val-icon { font-size:1rem; min-width:22px; }

/* Before/after table */
.ba-row { display:grid; grid-template-columns:1fr 1fr 1fr 1fr;
          padding:8px 4px; border-bottom:1px solid #1e2533; align-items:center; }
.ba-header { font-weight:700; color:#94a3b8; font-size:0.78rem;
             text-transform:uppercase; letter-spacing:0.06em; }
.ba-up   { color:#f87171; font-weight:700; }
.ba-down { color:#34d399; font-weight:700; }
.ba-same { color:#94a3b8; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVENT_TYPE_ICONS = {
    "ENCOUNTER":    "🏥",
    "OBSERVATION":  "🔬",
    "CONDITION":    "🩺",
    "MEDICATION":   "💊",
    "PROCEDURE":    "⚕️",
    "IMMUNIZATION": "💉",
    "ALLERGY":      "⚠️",
    "CAREPLAN":     "📋",
}

RISK_COLORS = {
    "low":      ("#34d399", "#0d4429"),
    "moderate": ("#fbbf24", "#451a03"),
    "high":     ("#f87171", "#450a0a"),
}


# ---------------------------------------------------------------------------
# Schema initialisation (once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing REALTIME schema…")
def _init_schema():
    try:
        ensure_realtime_schema()
        return True
    except Exception as e:
        return str(e)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _set_patient(patient_id: str):
    if st.session_state.get("selected_patient_id") != patient_id:
        st.session_state["selected_patient_id"] = patient_id
        st.session_state["risk_before"] = None
        st.session_state["risk_after"]  = None
        st.session_state["last_event"]  = None


def _clear_patient():
    for k in ["selected_patient_id", "risk_before", "risk_after", "last_event"]:
        st.session_state.pop(k, None)


# ---------------------------------------------------------------------------
# Event routing and persistence helpers
# ---------------------------------------------------------------------------

# Maps event_type → the REALTIME table that stores it
_EVENT_TYPE_TO_RT_TABLE: dict[str, str] = {
    "ENCOUNTER":    "REALTIME.RT_ENCOUNTERS",
    "OBSERVATION":  "REALTIME.RT_OBSERVATIONS",
    "CONDITION":    "REALTIME.RT_CONDITIONS",
    "MEDICATION":   "REALTIME.RT_MEDICATIONS",
    "PROCEDURE":    "REALTIME.RT_PROCEDURES",
    "IMMUNIZATION": "REALTIME.RT_IMMUNIZATIONS",
    "ALLERGY":      "REALTIME.RT_ALLERGIES",
    "CAREPLAN":     "REALTIME.RT_CAREPLANS",
}

_PERSISTENCE_TIMEOUT_S  = 30   # seconds to wait before giving up
_PERSISTENCE_POLL_S     = 1.5  # polling interval in seconds


def _event_persisted_in_snowflake(event_id: str, event_type: str) -> bool:
    """
    Return True if the given event_id exists in the appropriate REALTIME table.
    Uses a fresh connection each call (no cache) so we always see the latest state.
    """
    table = _EVENT_TYPE_TO_RT_TABLE.get(event_type.upper())
    if not table:
        return False
    import snowflake.connector
    from config import snowflake_connect_kwargs
    try:
        conn = snowflake.connector.connect(**snowflake_connect_kwargs())
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM {table} WHERE EVENT_ID = %s LIMIT 1", (event_id,))
        found = cur.fetchone() is not None
        cur.close()
        conn.close()
        return found
    except Exception:
        return False


def _publish_to_kafka(event: MedicalEvent) -> tuple[bool, str]:
    """Try Kafka only; do NOT fall back to direct insert here."""
    try:
        from kafka_producer import produce_event, test_connectivity
        if test_connectivity():
            ok = produce_event(event, validate=False)
            if ok:
                return True, "kafka"
            return False, "kafka_produce_failed"
        return False, "kafka_unavailable"
    except Exception as exc:
        return False, f"kafka_error:{exc}"


def _try_produce(event: MedicalEvent) -> tuple[bool, str]:
    """
    Primary: Kafka.
    Fallback: direct Snowflake insert (only when Kafka is unavailable).
    Never calls insert_event_direct when Kafka succeeds — avoids duplicate writes.
    """
    kafka_ok, kafka_reason = _publish_to_kafka(event)
    if kafka_ok:
        return True, "kafka"

    # Kafka unavailable or failed — fall back to direct insert
    try:
        insert_event_direct(event)
        return True, "direct"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Risk label / color helpers
# ---------------------------------------------------------------------------
def _risk_tier(prob: float | None) -> str:
    if prob is None:
        return "unavailable"
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "moderate"
    return "low"


def _risk_emoji(prob: float | None) -> str:
    t = _risk_tier(prob)
    return {"high": "🔴", "moderate": "🟡", "low": "🟢", "unavailable": "⚪"}[t]


# ---------------------------------------------------------------------------
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown("## 🏥 Patient Search")

    search_mode = st.sidebar.radio("Search by", ["Patient ID", "Full Name"],
                                   horizontal=True)

    if search_mode == "Patient ID":
        pid_input = st.sidebar.text_input("Patient ID (UUID)", key="pid_input")
        if st.sidebar.button("🔍 Search", key="btn_search_id"):
            pid = pid_input.strip()
            if pid:
                df = search_patient_by_id(pid, cache_version=_get_db_version())
                if df is not None and not df.empty:
                    _set_patient(pid)
                    name = df.iloc[0].get("FULL_NAME", pid)
                    st.sidebar.success(f"✓ {name}")
                else:
                    st.sidebar.error("Patient not found.")
            else:
                st.sidebar.warning("Enter a Patient ID.")

    else:
        name_input = st.sidebar.text_input("Full Name (partial)", key="name_input")
        if st.sidebar.button("🔍 Search", key="btn_search_name"):
            if name_input.strip():
                results = search_patient_by_name(name_input.strip(), cache_version=_get_db_version())
                if not results.empty:
                    st.session_state["name_search_results"] = results
                else:
                    st.sidebar.error("No patients found.")
                    st.session_state.pop("name_search_results", None)
            else:
                st.sidebar.warning("Enter a name.")

        if "name_search_results" in st.session_state:
            results = st.session_state["name_search_results"]
            options = {
                f"{r['FULL_NAME']}  ({r['PATIENT_ID'][:8]}…)": r["PATIENT_ID"]
                for _, r in results.iterrows()
            }
            chosen = st.sidebar.selectbox("Select patient", list(options.keys()),
                                          key="name_select")
            if st.sidebar.button("Select →", key="btn_select_name"):
                _set_patient(options[chosen])

    pid = st.session_state.get("selected_patient_id")
    if pid:
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f"<div style='background:#0d4429;border-radius:8px;"
            f"padding:10px;margin:4px 0;font-size:0.82rem;color:#34d399'>"
            f"<b>Active Patient</b><br/>"
            f"<span style='color:#a7f3d0;word-break:break-all'>{pid}</span></div>",
            unsafe_allow_html=True
        )
        if st.sidebar.button("✕ Clear Selection", key="btn_clear"):
            _clear_patient()
            st.rerun()

    # ── Quick system status in sidebar ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")

    # Kafka
    try:
        from kafka_producer import test_connectivity
        kafka_ok = test_connectivity()
        st.sidebar.markdown(
            f"**Kafka** &nbsp; {'🟢 Connected' if kafka_ok else '🔴 Offline'}"
        )
    except Exception:
        st.sidebar.markdown("**Kafka** &nbsp; ⚪ N/A")

    # ML Models
    avail = models_available()
    found = sum(1 for v in avail.values() if v)
    total = len(TARGETS)
    color = "🟢" if found == total else ("🟡" if found > 0 else "🔴")
    st.sidebar.markdown(f"**ML Models** &nbsp; {color} {found}/{total} loaded")

    # Snowflake
    sf_ok, sf_err = _test_snowflake_connection()
    if sf_ok:
        st.sidebar.markdown("**Snowflake** &nbsp; 🟢 Connected")
    else:
        st.sidebar.markdown("**Snowflake** &nbsp; 🔴 Error")


# ---------------------------------------------------------------------------
# ── TAB 1: PATIENT OVERVIEW ───────────────────────────────────────────────
# ---------------------------------------------------------------------------
def render_patient_overview(patient_id: str):
    info = get_patient_info(patient_id, cache_version=_get_db_version())
    if info is None:
        st.error(f"Patient `{patient_id}` not found in database.")
        return

    # Demographics header
    st.markdown('<div class="section-header">👤 Demographics</div>',
                unsafe_allow_html=True)

    col_name, col_gender, col_race, col_marital = st.columns(4)
    col_name.metric("Full Name", info.get("FULL_NAME", "—"))
    col_gender.metric("Gender",   info.get("GENDER", "—"))
    col_race.metric("Race",       info.get("RACE", "—"))
    col_marital.metric("Marital", info.get("MARITAL", "—"))

    col_bd, col_eth, col_status = st.columns(3)
    bd = info.get("BIRTHDATE")
    col_bd.metric("Birthdate", str(bd) if bd else "—")
    col_eth.metric("Ethnicity", info.get("ETHNICITY", "—"))
    dd = info.get("DEATHDATE")
    death_str = "Alive" if (dd is None or str(dd) == "9999-12-31") else str(dd)
    col_status.metric("Status", death_str)

    st.markdown("---")

    # 360 summary counts
    st.markdown('<div class="section-header">📊 Patient 360 Summary</div>',
                unsafe_allow_html=True)

    enc   = get_encounters(patient_id, limit=1000, cache_version=_get_db_version())
    cond  = get_conditions(patient_id, cache_version=_get_db_version())
    meds  = get_medications(patient_id, cache_version=_get_db_version())
    hist_obs, rt_obs = get_latest_observations(patient_id, cache_version=_get_db_version())

    total_enc   = len(enc)
    hist_enc    = len(enc[enc["SOURCE"] == "historical"]) if not enc.empty else 0
    rt_enc      = len(enc[enc["SOURCE"] == "realtime"])   if not enc.empty else 0
    total_cond  = len(cond)
    hist_cond   = len(cond[cond["SOURCE"] == "historical"]) if not cond.empty else 0
    rt_cond     = len(cond[cond["SOURCE"] == "realtime"])   if not cond.empty else 0
    total_meds  = len(meds)
    hist_meds   = len(meds[meds["SOURCE"] == "historical"]) if not meds.empty else 0
    rt_meds     = len(meds[meds["SOURCE"] == "realtime"])   if not meds.empty else 0
    total_obs   = len(hist_obs) + len(rt_obs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Encounters",  total_enc,  f"📂 {hist_enc} hist · ⚡ {rt_enc} live")
    c2.metric("Total Conditions",  total_cond, f"📂 {hist_cond} hist · ⚡ {rt_cond} live")
    c3.metric("Total Medications", total_meds, f"📂 {hist_meds} hist · ⚡ {rt_meds} live")
    c4.metric("Observations",      total_obs,  f"📂 {len(hist_obs)} hist · ⚡ {len(rt_obs)} live")

    st.markdown("---")

    # Clinical details in sub-tabs
    st.markdown('<div class="section-header">🗂 Clinical Records</div>',
                unsafe_allow_html=True)

    subtab_enc, subtab_cond, subtab_meds, subtab_obs, subtab_allerg = st.tabs([
        "📅 Encounters", "🩺 Conditions", "💊 Medications",
        "🔬 Observations", "⚠️ Allergies",
    ])

    with subtab_enc:
        if enc.empty:
            st.info("No encounters found.")
        else:
            _styled_df(enc)

    with subtab_cond:
        c = get_conditions(patient_id, cache_version=_get_db_version())
        if c.empty:
            st.info("No conditions found.")
        else:
            _styled_df(c)

    with subtab_meds:
        m = get_medications(patient_id, cache_version=_get_db_version())
        if m.empty:
            st.info("No medications found.")
        else:
            _styled_df(m)

    with subtab_obs:
        if not hist_obs.empty:
            st.markdown("**Historical Observations (Gold)**")
            _styled_df(hist_obs)
        if not rt_obs.empty:
            st.markdown("**⚡ Real-Time Observations**")
            _styled_df(rt_obs)
        if hist_obs.empty and rt_obs.empty:
            st.info("No observations found.")

    with subtab_allerg:
        a = get_allergies(patient_id, cache_version=_get_db_version())
        if a.empty:
            st.info("No allergies recorded.")
        else:
            _styled_df(a)


def _styled_df(df: pd.DataFrame):
    """Display a DataFrame with SOURCE column color-coded."""
    if "SOURCE" in df.columns:
        def _color_rows(row):
            if row.get("SOURCE") == "realtime":
                return ["background-color: #1f1a0e; color: #f59e0b"] * len(row)
            return [""] * len(row)
        st.dataframe(
            df.style.apply(_color_rows, axis=1),
            use_container_width=True,
        )
    else:
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# ── TAB 2: RISK ASSESSMENT ────────────────────────────────────────────────
# ---------------------------------------------------------------------------
@st.fragment
def render_risk_assessment(patient_id: str):
    st.markdown(
        "<div style='color:#94a3b8;font-size:0.82rem;margin-bottom:12px'>"
        "⚠️ <em>These scores are model-predicted probabilities, not clinical diagnoses. "
        "For informational purposes only.</em></div>",
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🔄 Refresh Risk Scores", type="primary"):
            with st.spinner("Computing risk scores from GOLD + REALTIME…"):
                try:
                    risk = compute_risk(patient_id)
                    # If no before baseline yet, set it now
                    if st.session_state.get("risk_before") is None:
                        st.session_state["risk_before"] = risk
                        st.session_state["risk_after"]  = None
                    else:
                        st.session_state["risk_after"] = risk
                except Exception as e:
                    st.error(f"Risk computation failed: {e}")

    # ── Current scores display ──
    risk_before = st.session_state.get("risk_before")
    risk_after  = st.session_state.get("risk_after")
    current     = risk_after or risk_before

    if current is None:
        st.info("👆 Click **Refresh Risk Scores** to compute the current disease risk.")
        return

    st.markdown('<div class="section-header">🎯 Current Risk Scores (MODEL-PREDICTED)</div>',
                unsafe_allow_html=True)

    scores = current.get("scores", {})
    cols = st.columns(len(TARGETS))
    for i, target in enumerate(TARGETS):
        info = scores.get(target, {})
        prob = info.get("probability")
        pct  = info.get("pct")
        tier = _risk_tier(prob)
        emoji = _risk_emoji(prob)

        with cols[i]:
            if prob is not None:
                fill_color = RISK_COLORS[tier][0]
                bg_color   = RISK_COLORS[tier][1]
                st.metric(
                    label=target,
                    value=f"{pct:.1f}%",
                    delta=f"{emoji} {tier.title()} Risk",
                    delta_color="off",
                )
                # Thin bar chart for visual clarity
                pct_safe = max(0.0, min(100.0, pct or 0.0))
                st.markdown(
                    f'<div class="risk-bar-wrap">'
                    f'<div class="risk-bar-fill" style="width:{pct_safe:.0f}%;'
                    f'background:{fill_color}"></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.metric(label=target, value="—", delta="Model unavailable",
                          delta_color="off")

    if current.get("models_missing"):
        st.warning(f"Model files missing for: {', '.join(current['models_missing'])}")

    # ── Feature values expander ──
    with st.expander("📋 Feature Values Used for This Prediction"):
        feats = current.get("features", {})
        feat_df = pd.DataFrame(
            [(k, round(v, 4) if isinstance(v, float) else v)
             for k, v in feats.items()],
            columns=["Feature", "Value"],
        )
        st.dataframe(feat_df, use_container_width=True, height=350)

    # ── Before / After comparison ──
    if risk_before and risk_after:
        st.markdown("---")
        st.markdown(
            '<div class="section-header">⚖️ Before / After Risk Comparison</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='color:#94a3b8;font-size:0.8rem;margin-bottom:10px'>"
            "Model score changed after the new event was incorporated into the patient state.</div>",
            unsafe_allow_html=True,
        )

        before_scores = risk_before.get("scores", {})
        after_scores  = risk_after.get("scores", {})

        # Header row
        st.markdown(
            '<div class="ba-row ba-header">'
            '<div>Disease</div><div>Before</div><div>After</div><div>Change</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        for target in TARGETS:
            b_info = before_scores.get(target, {})
            a_info = after_scores.get(target, {})
            b_prob = b_info.get("probability")
            a_prob = a_info.get("probability")
            b_pct  = b_info.get("pct")
            a_pct  = a_info.get("pct")

            if b_prob is not None and a_prob is not None:
                delta     = a_prob - b_prob
                delta_pct = delta * 100
                icon = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "—")
                css_class = ("ba-up" if delta > 0.005
                             else ("ba-down" if delta < -0.005 else "ba-same"))
                change_str = (f"+{delta_pct:.1f}%" if delta > 0.005
                              else (f"{delta_pct:.1f}%" if delta < -0.005 else "No change"))
            else:
                icon = "—"
                css_class = "ba-same"
                change_str = "—"

            b_str = f"{b_pct:.1f}%" if b_pct is not None else "—"
            a_str = f"{a_pct:.1f}%" if a_pct is not None else "—"
            emoji = EVENT_TYPE_ICONS.get("ENCOUNTER", "")

            st.markdown(
                f'<div class="ba-row">'
                f'<div><b>{target}</b></div>'
                f'<div style="color:#94a3b8">{b_str}</div>'
                f'<div>{a_str}</div>'
                f'<div class="{css_class}">{icon} {change_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='color:#475569;font-size:0.75rem;margin-top:10px'>"
            "▲ increase &nbsp; ▼ decrease &nbsp; — unchanged</div>",
            unsafe_allow_html=True,
        )

    elif risk_before and not risk_after:
        st.info(
            "💡 **Baseline captured.** Submit a medical event in the **Live Event** tab, "
            "then click Refresh Risk Scores again to see the before/after comparison."
        )


# ---------------------------------------------------------------------------
# ── TAB 3: CLINICAL TIMELINE ──────────────────────────────────────────────
# ---------------------------------------------------------------------------
def render_clinical_timeline(patient_id: str):
    st.markdown(
        "<div style='color:#94a3b8;font-size:0.82rem;margin-bottom:12px'>"
        "Chronological view of all clinical events · "
        "<span style='color:#f59e0b'>⚡ Orange = Real-time events</span> · "
        "<span style='color:#60a5fa'>🔵 Blue = Historical (Gold)</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading clinical timeline…"):
        timeline = get_timeline(patient_id, limit=200, cache_version=_get_db_version())

    if timeline.empty:
        st.info("No clinical events found for this patient.")
        return

    total     = len(timeline)
    hist_cnt  = len(timeline[timeline["SOURCE"] == "historical"])
    rt_cnt    = len(timeline[timeline["SOURCE"] == "realtime"])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Events", total)
    col_b.metric("Historical",   hist_cnt)
    col_c.metric("⚡ Real-Time", rt_cnt)

    st.markdown("---")

    # Filter controls
    filt_col1, filt_col2 = st.columns(2)
    with filt_col1:
        source_filter = st.multiselect(
            "Filter by source",
            ["historical", "realtime"],
            default=["historical", "realtime"],
            key="tl_source",
        )
    with filt_col2:
        type_opts = sorted(timeline["EVENT_TYPE"].unique().tolist())
        type_filter = st.multiselect(
            "Filter by event type",
            type_opts,
            default=type_opts,
            key="tl_type",
        )

    filtered = timeline[
        timeline["SOURCE"].isin(source_filter) &
        timeline["EVENT_TYPE"].isin(type_filter)
    ]

    if filtered.empty:
        st.info("No events match the selected filters.")
        return

    # Render timeline cards
    prev_date = None
    for _, row in filtered.iterrows():
        ev_date = row["EVENT_DATE"]
        source  = row.get("SOURCE", "historical")
        ev_type = row.get("EVENT_TYPE", "EVENT")
        desc    = row.get("DESCRIPTION") or "—"
        detail  = row.get("DETAIL") or ""
        icon    = EVENT_TYPE_ICONS.get(ev_type, "📌")

        # Date separator
        date_str = ev_date.strftime("%Y-%m-%d") if hasattr(ev_date, "strftime") else str(ev_date)[:10]
        if date_str != prev_date:
            st.markdown(
                f"<div style='color:#475569;font-size:0.75rem;"
                f"margin:16px 0 4px;padding-top:8px;"
                f"border-top:1px solid #1e2533'>{date_str}</div>",
                unsafe_allow_html=True,
            )
            prev_date = date_str

        is_rt    = source == "realtime"
        card_cls = "timeline-card realtime" if is_rt else "timeline-card"
        badge_cls = "badge-realtime" if is_rt else "badge-historical"
        badge_txt = "⚡ REALTIME" if is_rt else "HISTORICAL"
        detail_html = (
            f"<span style='color:#94a3b8;font-size:0.8rem'>{detail}</span>"
            if detail else ""
        )

        st.markdown(
            f'<div class="{card_cls}">'
            f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">'
            f'    <span class="timeline-badge {badge_cls}">{badge_txt}</span>'
            f'    <span style="font-size:0.88rem;font-weight:600">{icon} {ev_type}</span>'
            f'  </div>'
            f'  <div style="font-size:0.9rem">{desc}</div>'
            f'  {detail_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# ── TAB 4: LIVE EVENT ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
@st.fragment
def render_live_event(patient_id: str):
    # ── Event form ──
    st.markdown('<div class="section-header">➕ Submit New Medical Event</div>',
                unsafe_allow_html=True)
    st.caption(f"Patient: **{patient_id}**")

    event_type = st.selectbox("Event Type", SUPPORTED_EVENT_TYPES, key="le_type")
    payload    = _build_payload_form(event_type)

    # ── Live validation preview ──
    st.markdown('<div class="section-header">🔍 Event Validation</div>',
                unsafe_allow_html=True)

    trial_event = MedicalEvent(
        patient_id=patient_id,
        event_type=event_type,
        payload=dict(payload),
    )
    _render_validation_panel(trial_event)

    st.markdown("---")

    # ── Submit button ──
    submit = st.button("🚀 Submit Event", type="primary", key="le_submit")

    # ── Display outcome stored from previous submit (survives st.rerun()) ──
    _pending = st.session_state.pop("_event_outcome", None)
    if _pending:
        if _pending.get("success"):
            if _pending.get("risk_updated"):
                st.success("✅ Event persisted and risk updated. Check the **Risk Assessment** tab.")
            else:
                st.success("✅ Event persisted. Click **Refresh Risk Scores** in Risk Assessment.")
        else:
            st.warning(_pending.get("msg", "Event outcome unknown."))
        with st.expander("📄 Event Payload", expanded=False):
            st.json(_pending.get("payload", {}))

    if submit:
        event = MedicalEvent(
            patient_id=patient_id,
            event_type=event_type,
            payload=payload,
        )

        try:
            validate_event(event, fill_defaults=True)
        except ValidationError as ve:
            st.error(f"❌ Validation failed: {ve}")
            return

        # ── Processing status tracker ──
        st.markdown('<div class="section-header">⚙️ Event Processing Status</div>',
                    unsafe_allow_html=True)

        # Step 1 — Event created
        _status_row("Event Created", True, event.event_id[:12] + "…")

        # Step 2 — Publish (Kafka primary; direct-insert fallback only when Kafka unavailable)
        ok, method = _try_produce(event)
        kafka_ok  = ok and method == "kafka"
        direct_ok = ok and method == "direct"
        _status_row(
            "Published",
            ok,
            f"Kafka topic `{KAFKA_TOPIC}`" if kafka_ok else
            ("Direct Snowflake insert (Kafka unavailable)" if direct_ok else f"Failed: {method}"),
        )

        if not ok:
            st.error(f"❌ Event could not be published: {method}")
            return

        # Step 3 — Wait for REALTIME persistence
        #   • Direct insert: already in Snowflake → first poll should hit immediately.
        #   • Kafka path:   Spark consumes and writes; poll until confirmed.
        import time as _time
        persistence_confirmed = False
        poll_status = st.empty()

        deadline = _time.monotonic() + _PERSISTENCE_TIMEOUT_S
        attempt  = 0
        while _time.monotonic() < deadline:
            attempt += 1
            if _event_persisted_in_snowflake(event.event_id, event.event_type):
                persistence_confirmed = True
                break
            elapsed = attempt * _PERSISTENCE_POLL_S
            poll_status.info(
                f"⏳ Waiting for REALTIME persistence "
                f"(attempt {attempt}, ~{elapsed:.0f}s elapsed of {_PERSISTENCE_TIMEOUT_S}s max)…"
            )
            _time.sleep(_PERSISTENCE_POLL_S)

        poll_status.empty()   # clear the polling indicator

        _status_row(
            "Snowflake REALTIME Confirmed",
            persistence_confirmed,
            (f"event_id `{event.event_id[:12]}…` found in "
             f"{_EVENT_TYPE_TO_RT_TABLE.get(event.event_type.upper(), '?')}")
            if persistence_confirmed
            else f"Not confirmed within {_PERSISTENCE_TIMEOUT_S}s",
        )

        if not persistence_confirmed:
            # Do NOT compute risk on potentially stale data — be truthful
            _status_row("Spark Processing",   None,  "⏳ Event may still be in-flight.")
            _status_row("Risk Scores Updated", False, "Skipped — persistence not confirmed.")
            st.warning(
                "⚠️ Event was published to Kafka but REALTIME persistence was not confirmed "
                f"within {_PERSISTENCE_TIMEOUT_S}s. "
                "Spark may still be processing it. "
                "Click **Refresh Risk Scores** in the Risk Assessment tab once processing completes."
            )
            st.session_state["last_event"] = {
                "event_id":   event.event_id,
                "event_type": event.event_type,
                "method":     method,
                "ts":         event.event_timestamp,
                "ok":         False,
            }
            return

        # Step 4 — Spark confirmed
        _status_row(
            "Spark / Ingestion",
            True,
            "Confirmed in REALTIME" if kafka_ok else "Direct insert confirmed",
        )

        # Step 5 — Recompute risk using fresh data from Snowflake
        # feature_engine.py reads directly from Snowflake (no cache) so it will
        # see the newly persisted event immediately.
        risk_updated = False
        try:
            updated_risk = compute_risk(patient_id)
            st.session_state["risk_after"] = updated_risk
            risk_updated = True
        except Exception as exc:
            st.warning(f"Risk recomputation failed: {exc}")

        _status_row(
            "Risk Scores Updated",
            risk_updated,
            "See Risk Assessment tab" if risk_updated else "Computation error",
        )

        # Persist last-event record for System Health tab
        st.session_state["last_event"] = {
            "event_id":   event.event_id,
            "event_type": event.event_type,
            "method":     method,
            "ts":         event.event_timestamp,
            "ok":         True,
        }

        # Store outcome message so it survives the upcoming st.rerun()
        st.session_state["_event_outcome"] = {
            "success":      True,
            "risk_updated": risk_updated,
            "payload":      event.to_dict(),
            "msg":          "",
        }

        # Bust UI caches so patient overview / timeline fetch fresh data on next rerun
        _bump_db_version()
        # st.rerun() causes a full page rerun; because we stored outcome in session_state
        # above, the message block at the top of this function will display it correctly.
        st.rerun()


def _build_payload_form(event_type: str) -> dict:
    """Build the dynamic form fields and return the payload dict."""
    payload = {}
    today_str = date.today().isoformat()
    now_str   = datetime.now().isoformat(timespec="seconds")

    if event_type == "ENCOUNTER":
        payload["ENCOUNTER_DATE"]    = st.text_input("Encounter Date (ISO-8601)", value=now_str,   key="lp_enc_date")
        payload["CODE"]              = st.text_input("Code",                                        key="lp_enc_code")
        payload["DESCRIPTION"]       = st.text_input("Description",                                 key="lp_enc_desc")
        payload["REASONCODE"]        = st.text_input("Reason Code (optional)",  value="N/A",       key="lp_enc_rc")
        payload["REASONDESCRIPTION"] = st.text_input("Reason Description (optional)", value="Routine / Unspecified", key="lp_enc_rd")

    elif event_type == "OBSERVATION":
        obs_opts = ["Body Weight","Body Height","Body Mass Index",
                    "Systolic Blood Pressure","Diastolic Blood Pressure",
                    "Glucose","Total Cholesterol","Other"]
        obs_choice = st.selectbox("Observation Type", obs_opts, key="lp_obs_sel")
        payload["DESCRIPTION"]      = (st.text_input("Description", key="lp_obs_descfree")
                                       if obs_choice == "Other" else obs_choice)
        payload["OBSERVATION_DATE"] = st.text_input("Observation Date (YYYY-MM-DD)", value=today_str, key="lp_obs_date")
        val_type = st.radio("Value type", ["Numeric","Text"], horizontal=True, key="lp_obs_valtype")
        if val_type == "Numeric":
            payload["VALUE_NUMERIC"] = st.number_input("Numeric Value", value=0.0, key="lp_obs_num")
            payload["VALUE_TEXT"]    = None
        else:
            payload["VALUE_TEXT"]    = st.text_input("Text Value", key="lp_obs_txt")
            payload["VALUE_NUMERIC"] = None

    elif event_type == "CONDITION":
        payload["CODE"]        = st.text_input("Code",                        key="lp_cond_code")
        payload["DESCRIPTION"] = st.text_input("Description",                 key="lp_cond_desc")
        payload["START_DATE"]  = st.text_input("Start Date (ISO-8601)", value=today_str, key="lp_cond_sd")
        payload["END_DATE"]    = st.text_input("End Date (ISO-8601, 9999-12-31 if ongoing)", value="9999-12-31", key="lp_cond_ed")

    elif event_type == "MEDICATION":
        payload["CODE"]               = st.text_input("Code",                    key="lp_med_code")
        payload["DESCRIPTION"]        = st.text_input("Description",             key="lp_med_desc")
        payload["REASONCODE"]         = st.text_input("Reason Code (optional)", value="N/A", key="lp_med_rc")
        payload["REASONDESCRIPTION"]  = st.text_input("Reason Description",     key="lp_med_rd")
        payload["START_DATE"]         = st.text_input("Start Date", value=today_str, key="lp_med_sd")
        payload["END_DATE"]           = st.text_input("End Date (9999-12-31 if ongoing)", value="9999-12-31", key="lp_med_ed")

    elif event_type == "PROCEDURE":
        payload["CODE"]               = st.text_input("Code",                    key="lp_proc_code")
        payload["DESCRIPTION"]        = st.text_input("Description",             key="lp_proc_desc")
        payload["REASONCODE"]         = st.text_input("Reason Code (optional)", value="N/A", key="lp_proc_rc")
        payload["REASONDESCRIPTION"]  = st.text_input("Reason Description",     key="lp_proc_rd")
        payload["PROCEDURE_DATE"]     = st.text_input("Procedure Date", value=now_str, key="lp_proc_date")

    elif event_type == "IMMUNIZATION":
        payload["CODE"]              = st.text_input("Code",               key="lp_imm_code")
        payload["DESCRIPTION"]       = st.text_input("Description",        key="lp_imm_desc")
        payload["IMMUNIZATION_DATE"] = st.text_input("Immunization Date", value=today_str, key="lp_imm_date")

    elif event_type == "ALLERGY":
        payload["CODE"]        = st.text_input("Code",                   key="lp_alg_code")
        payload["DESCRIPTION"] = st.text_input("Description",            key="lp_alg_desc")
        payload["START_DATE"]  = st.text_input("Start Date", value=today_str, key="lp_alg_sd")
        payload["END_DATE"]    = st.text_input("End Date (9999-12-31 if ongoing)", value="9999-12-31", key="lp_alg_ed")

    elif event_type == "CAREPLAN":
        payload["CODE"]               = st.text_input("Code",                   key="lp_cp_code")
        payload["DESCRIPTION"]        = st.text_input("Description",            key="lp_cp_desc")
        payload["REASONCODE"]         = st.text_input("Reason Code (optional)", value="N/A", key="lp_cp_rc")
        payload["REASONDESCRIPTION"]  = st.text_input("Reason Description",     key="lp_cp_rd")
        payload["START_DATE"]         = st.text_input("Start Date", value=today_str, key="lp_cp_sd")
        payload["END_DATE"]           = st.text_input("End Date (9999-12-31 if ongoing)", value="9999-12-31", key="lp_cp_ed")

    payload["ENCOUNTER_ID"] = st.text_input(
        "Encounter ID (optional — link to existing encounter)",
        value="", key="lp_enc_id"
    ) or None

    return payload


def _render_validation_panel(event: MedicalEvent):
    """Render a live per-field validation dashboard."""
    checks = []

    # event_id
    checks.append(("Event ID",      bool(event.event_id),           event.event_id[:12] + "…"))
    # patient_id
    pid_ok = bool(event.patient_id and event.patient_id.strip())
    checks.append(("Patient ID",    pid_ok,                          event.patient_id or "—"))
    # event_type
    from event_schema import SUPPORTED_EVENT_TYPES
    type_ok = event.event_type in SUPPORTED_EVENT_TYPES
    checks.append(("Event Type",    type_ok,                         event.event_type))
    # timestamp
    ts_ok = bool(event.event_timestamp)
    checks.append(("Timestamp",     ts_ok,                           event.event_timestamp[:19] if ts_ok else "—"))

    # Required payload fields
    required = REQUIRED_PAYLOAD_FIELDS.get(event.event_type, [])
    all_required_ok = True
    for field_name in required:
        val = event.payload.get(field_name)
        field_ok = val is not None and str(val).strip() != ""
        if not field_ok:
            all_required_ok = False
        checks.append((f"payload.{field_name}", field_ok,
                        str(val)[:30] if val else "❌ missing"))

    # OBSERVATION special rule
    if event.event_type == "OBSERVATION":
        has_num  = event.payload.get("VALUE_NUMERIC") is not None
        has_text = event.payload.get("VALUE_TEXT") is not None
        val_ok   = has_num or has_text
        checks.append(("At least one value",  val_ok,
                        "VALUE_NUMERIC or VALUE_TEXT present" if val_ok else "❌ both null"))

    # Render table
    for label, ok, detail in checks:
        icon = "✅" if ok else "❌"
        color = "#94a3b8" if ok else "#f87171"
        st.markdown(
            f'<div class="val-row">'
            f'  <span class="val-icon">{icon}</span>'
            f'  <span style="min-width:180px">{label}</span>'
            f'  <span style="color:{color};font-size:0.8rem">{detail}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Overall result
    all_ok = all(ok for _, ok, _ in checks)
    if all_ok:
        st.success("✅ All validation checks passed — ready to submit.")
    else:
        st.warning("⚠️ Fix the fields marked ❌ before submitting.")


def _status_row(label: str, ok: bool | None, detail: str = ""):
    """Render a single processing-status row. ok=None means 'pending'."""
    if ok is True:
        icon  = "✅"
        color = "#34d399"
    elif ok is False:
        icon  = "❌"
        color = "#f87171"
    else:
        icon  = "⏳"
        color = "#fbbf24"

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:7px 0;'
        f'border-bottom:1px solid #1e2533">'
        f'  <span style="font-size:1.1rem">{icon}</span>'
        f'  <span style="min-width:240px;font-weight:600">{label}</span>'
        f'  <span style="color:{color};font-size:0.82rem">{detail}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# ── TAB 5: SYSTEM HEALTH ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------

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

def render_system_health():
    st.markdown('<div class="section-header">🖥 Infrastructure Status</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # Kafka
    try:
        from kafka_producer import test_connectivity
        kafka_ok = test_connectivity()
        col1.metric("Kafka", "🟢 Connected" if kafka_ok else "🔴 Offline",
                    KAFKA_BOOTSTRAP_SERVERS)
    except Exception as e:
        col1.metric("Kafka", "⚪ N/A", str(e)[:40])

    # Snowflake
    from config import SNOWFLAKE_ACCOUNT
    sf_ok, sf_err = _test_snowflake_connection()
    if sf_ok:
        col2.metric("Snowflake", "🟢 Connected", SNOWFLAKE_ACCOUNT)
    else:
        col2.metric("Snowflake", "🔴 Error", sf_err[:40])

    # Spark Streaming
    col3.metric("Spark Streaming", "⚪ Status N/A",
                "Check spark_streaming.py terminal")

    # ML Models
    avail = models_available()
    found = sum(1 for v in avail.values() if v)
    total = len(TARGETS)
    col4.metric("ML Models",
                f"{'🟢' if found == total else '🟡'} {found}/{total} Loaded",
                "All ready" if found == total else "Some missing")

    st.markdown("---")
    st.markdown('<div class="section-header">🤖 ML Model Details</div>',
                unsafe_allow_html=True)

    model_cols = st.columns(len(TARGETS))
    for i, (target, exists) in enumerate(avail.items()):
        with model_cols[i]:
            if exists:
                from config import model_path
                p = model_path(target)
                size_kb = p.stat().st_size // 1024
                st.metric(target, "✅ Ready", f"{size_kb} KB")
            else:
                st.metric(target, "❌ Missing", "Upload model file")

    # Last processed event
    st.markdown("---")
    st.markdown('<div class="section-header">📨 Last Processed Event</div>',
                unsafe_allow_html=True)

    last = st.session_state.get("last_event")
    if last:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Event ID",   last["event_id"][:12] + "…")
        c2.metric("Type",       last["event_type"])
        c3.metric("Method",     last["method"].title())
        c4.metric("Timestamp",  str(last["ts"])[:19])

        if last.get("ok"):
            st.success("Last event was successfully processed.")
        else:
            st.error("Last event processing failed.")
    else:
        st.info("No events processed in this session yet.")

    # Configuration reference
    st.markdown("---")
    st.markdown('<div class="section-header">⚙️ Configuration</div>',
                unsafe_allow_html=True)

    from config import (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_DATABASE,
                        SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE,
                        SCHEMA_GOLD, SCHEMA_SILVER, SCHEMA_ML, SCHEMA_REALTIME)

    cfg_data = {
        "Snowflake Account":   SNOWFLAKE_ACCOUNT,
        "Snowflake User":      SNOWFLAKE_USER,
        "Snowflake Database":  SNOWFLAKE_DATABASE,
        "Snowflake Warehouse": SNOWFLAKE_WAREHOUSE,
        "Snowflake Role":      SNOWFLAKE_ROLE,
        "Schema — Gold":       SCHEMA_GOLD,
        "Schema — Silver":     SCHEMA_SILVER,
        "Schema — ML":         SCHEMA_ML,
        "Schema — Realtime":   SCHEMA_REALTIME,
        "Kafka Servers":       KAFKA_BOOTSTRAP_SERVERS,
        "Kafka Topic":         KAFKA_TOPIC,
    }
    cfg_df = pd.DataFrame(list(cfg_data.items()), columns=["Parameter", "Value"])
    st.dataframe(cfg_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# ── MAIN ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def main():
    # Initialise schema
    schema_status = _init_schema()
    if schema_status is not True:
        st.error(f"Failed to initialize REALTIME schema: {schema_status}")
        st.stop()

    # Sidebar
    render_sidebar()

    # ── Header ──
    st.markdown(
        "<h1 style='margin-bottom:2px'>🏥 Healthcare Real-Time Risk Monitor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#64748b;margin-top:0'>Real-Time Patient State & Clinical Risk Monitoring · "
        "Kafka · Spark Structured Streaming · Snowflake · XGBoost</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    patient_id = st.session_state.get("selected_patient_id")

    # ── No patient selected ──
    if patient_id is None:
        st.info("👈 Search for a patient in the sidebar to get started.")

        # Model status preview
        st.markdown("### Model Status")
        avail = models_available()
        cols  = st.columns(len(TARGETS))
        for i, (target, exists) in enumerate(avail.items()):
            with cols[i]:
                st.metric(
                    target,
                    "✓ Ready" if exists else "✗ Missing",
                    delta="Full features" if exists else "Upload model file",
                    delta_color="normal" if exists else "inverse",
                )
        return

    # ── Patient selected — five tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 Patient Overview",
        "🎯 Risk Assessment",
        "📅 Clinical Timeline",
        "⚡ Live Event",
        "🖥 System Health",
    ])

    with tab1:
        render_patient_overview(patient_id)

    with tab2:
        render_risk_assessment(patient_id)

    with tab3:
        render_clinical_timeline(patient_id)

    with tab4:
        render_live_event(patient_id)

    with tab5:
        render_system_health()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.getenv("SNOWFLAKE_PASSWORD"):
        st.error(
            "❌ SNOWFLAKE_PASSWORD is not set. "
            "Run: `export SNOWFLAKE_PASSWORD='your_password'` before starting Streamlit."
        )
        st.stop()

    main()
