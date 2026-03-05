"""
dashboard/app.py
----------------
Streamlit predictive maintenance dashboard.

Features:
  - Sensor input sliders with realistic industrial ranges
  - Real-time API call to POST /predict_failure
  - Animated risk gauge (Plotly)
  - Color-coded risk alert banner
  - Feature contribution breakdown
  - Works both locally (API on localhost:8000) and in Docker (API on api:8000)
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
API_URL     = os.getenv("API_URL", "http://localhost:8000")
PREDICT_URL = f"{API_URL}/predict_failure"
HEALTH_URL  = f"{API_URL}/health"

RISK_COLORS = {
    "Low":    "#22c55e",   # Green
    "Medium": "#f59e0b",   # Amber
    "High":   "#ef4444",   # Red
}

RISK_GAUGE_RANGE = {
    "Low":    [0, 35],
    "Medium": [35, 65],
    "High":   [65, 100],
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "PredictiveMaintenance.AI",
    page_icon  = "🏭",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS — industrial dark theme ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e2e8f0;
    }

    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.0rem;
        font-weight: 600;
        color: #38bdf8;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }

    .sub-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 4px;
        margin-bottom: 28px;
    }

    .risk-banner {
        padding: 18px 24px;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }

    .risk-low    { background: #14532d; color: #4ade80; border: 1px solid #22c55e; }
    .risk-medium { background: #451a03; color: #fbbf24; border: 1px solid #f59e0b; }
    .risk-high   { background: #450a0a; color: #f87171; border: 1px solid #ef4444; }

    .metric-card {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        color: #e2e8f0;
    }

    .factor-item {
        background: #1e2433;
        border-left: 3px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #94a3b8;
    }

    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-ok     { background: #22c55e; }
    .status-error  { background: #ef4444; }

    div[data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2d3748;
    }

    .stSlider > div > div > div { background-color: #38bdf8 !important; }

    h3 { color: #94a3b8; font-size: 0.85rem; font-weight: 600;
         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Helper: API health check ──────────────────────────────────────────────────
def check_api_health() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        return response.json().get("model_loaded", False)
    except Exception:
        return False


# ── Helper: call prediction endpoint ─────────────────────────────────────────
def call_predict_api(payload: dict) -> Optional[dict]:
    try:
        response = requests.post(PREDICT_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the FastAPI server running on port 8000?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


# ── Helper: Plotly gauge chart ────────────────────────────────────────────────
def build_gauge(probability: float, risk_level: str) -> go.Figure:
    color = RISK_COLORS[risk_level]
    pct   = round(probability * 100, 1)

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = pct,
        number = {"suffix": "%", "font": {"size": 42, "color": color, "family": "IBM Plex Mono"}},
        gauge = {
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#2d3748",
                "tickfont": {"color": "#64748b", "size": 11},
            },
            "bar":   {"color": color, "thickness": 0.3},
            "bgcolor": "#1e2433",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "#14532d"},
                {"range": [35, 65], "color": "#451a03"},
                {"range": [65, 100],"color": "#450a0a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": pct,
            },
        },
        title = {
            "text": "7-Day Failure Probability",
            "font": {"size": 14, "color": "#64748b", "family": "IBM Plex Sans"},
        },
        domain = {"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        paper_bgcolor = "#0f1117",
        plot_bgcolor  = "#0f1117",
        margin        = dict(t=60, b=10, l=30, r=30),
        height        = 280,
    )

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
api_ok = check_api_health()
status_class = "status-ok" if api_ok else "status-error"
status_text  = "API CONNECTED" if api_ok else "API OFFLINE"

st.markdown(f"""
<div class="main-title">🏭 PredictiveMaintenance.AI</div>
<div class="sub-title">
    Industrial Equipment Failure Prediction  ·
    <span class="status-dot {status_class}"></span>{status_text}
</div>
""", unsafe_allow_html=True)

# ── Sidebar: sensor inputs ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Machine Identifier")
    machine_id = st.text_input("Machine ID", value="MIL-0042", placeholder="e.g. MIL-0042")

    st.markdown("### Product Type")
    product_type = st.selectbox(
        "Quality Variant",
        options=["L — Low (50%)", "M — Medium (30%)", "H — High (20%)"],
        index=1,
    )
    product_type_code = product_type.split(" ")[0]   # Extract L / M / H

    st.markdown("### Temperature")
    air_temp_c     = st.slider("Air Temperature (°C)",     min_value=-10.0, max_value=60.0,  value=25.1, step=0.1)
    process_temp_c = st.slider("Process Temperature (°C)", min_value=-10.0, max_value=100.0, value=36.4, step=0.1)

    st.markdown("### Mechanical Parameters")
    rotational_speed_rpm = st.slider("Rotational Speed (RPM)", min_value=1000.0, max_value=3000.0, value=1551.0, step=10.0)
    torque_nm            = st.slider("Torque (Nm)",             min_value=0.0,    max_value=100.0,  value=42.8,  step=0.5)

    st.markdown("### Tool Condition")
    tool_wear_min = st.slider("Tool Wear (min)", min_value=0.0, max_value=300.0, value=108.0, step=1.0)

    # Tool wear visual warning
    if tool_wear_min >= 200:
        st.warning("⚠️ Tool wear critical — replacement recommended")
    elif tool_wear_min >= 150:
        st.info("ℹ️ Tool wear elevated — monitor closely")

    st.markdown("---")
    run_prediction = st.button("🔍  RUN PREDICTION", use_container_width=True, type="primary")


# ── Main content ───────────────────────────────────────────────────────────────
if run_prediction:
    payload = {
        "machine_id":           machine_id if machine_id else None,
        "product_type":         product_type_code,
        "air_temp_c":           air_temp_c,
        "process_temp_c":       process_temp_c,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm":            torque_nm,
        "tool_wear_min":        tool_wear_min,
    }

    with st.spinner("Calling prediction API..."):
        result = call_predict_api(payload)

    if result:
        risk   = result["risk_level"]
        prob   = result["failure_probability"]
        risk_class = f"risk-{risk.lower()}"

        # ── Risk banner ───────────────────────────────────────────────────────
        st.markdown(
            f'<div class="risk-banner {risk_class}">⚠ RISK LEVEL: {risk.upper()}</div>',
            unsafe_allow_html=True
        )

        # ── Two column layout ─────────────────────────────────────────────────
        col1, col2 = st.columns([1.1, 1])

        with col1:
            # Gauge
            gauge_fig = build_gauge(prob, risk)
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

            # Key metrics row
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Probability</div>
                    <div class="metric-value" style="color:{RISK_COLORS[risk]}">{prob:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Threshold</div>
                    <div class="metric-value">{result['threshold_used']:.2f}</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model</div>
                    <div class="metric-value" style="font-size:1.1rem">{result['model_version']}</div>
                </div>""", unsafe_allow_html=True)

        with col2:
            # Recommendation
            st.markdown("### Recommendation")
            rec_color = RISK_COLORS[risk]
            st.markdown(f"""
            <div style="background:#1e2433; border-left:4px solid {rec_color};
                        padding:16px; border-radius:0 8px 8px 0; margin-bottom:20px;
                        color:#e2e8f0; font-size:0.95rem; line-height:1.6;">
                {result['recommendation']}
            </div>""", unsafe_allow_html=True)

            # Top risk factors
            st.markdown("### Top Contributing Factors")
            for factor in result["top_risk_factors"]:
                st.markdown(f'<div class="factor-item">▸ {factor}</div>', unsafe_allow_html=True)

            # Raw sensor summary
            st.markdown("### Sensor Snapshot")
            sensor_data = {
                "product_type":         product_type_code,
                "air_temp_c":           f"{air_temp_c} °C",
                "process_temp_c":       f"{process_temp_c} °C",
                "temp_differential":    f"{process_temp_c - air_temp_c:.1f} °C",
                "rotational_speed_rpm": f"{rotational_speed_rpm:,.0f} RPM",
                "torque_nm":            f"{torque_nm:.1f} Nm",
                "tool_wear_min":        f"{tool_wear_min:.0f} min",
            }
            for k, v in sensor_data.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 0;border-bottom:1px solid #2d3748;">'
                    f'<span style="color:#64748b;font-family:IBM Plex Mono,monospace;font-size:0.8rem">{k}</span>'
                    f'<span style="color:#e2e8f0;font-family:IBM Plex Mono,monospace;font-size:0.8rem;font-weight:600">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

else:
    # Default state: show instructions
    st.markdown("""
    <div style="background:#1e2433; border:1px dashed #2d3748; border-radius:12px;
                padding:40px; text-align:center; margin-top:40px;">
        <div style="font-size:3rem; margin-bottom:16px;">🔧</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
                    color:#38bdf8; margin-bottom:8px;">Ready for Analysis</div>
        <div style="color:#64748b; font-size:0.9rem;">
            Configure sensor readings in the left panel,<br>
            then click <strong style="color:#e2e8f0">RUN PREDICTION</strong> to get a real-time failure risk assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show sample risk levels reference
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="risk-banner risk-low">🟢 LOW RISK<br>
        <span style="font-size:0.75rem;font-weight:400">Probability &lt; 35%<br>Normal operations</span></div>""",
        unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="risk-banner risk-medium">🟡 MEDIUM RISK<br>
        <span style="font-size:0.75rem;font-weight:400">Probability 35–65%<br>Monitor closely</span></div>""",
        unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="risk-banner risk-high">🔴 HIGH RISK<br>
        <span style="font-size:0.75rem;font-weight:400">Probability &gt; 65%<br>Immediate action</span></div>""",
        unsafe_allow_html=True)