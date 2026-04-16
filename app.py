"""
Predictive Maintenance Dashboard — Real-Time Fault Prediction
Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance — Industrial Robotic Arms",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .fault-normal {
        background: linear-gradient(135deg, #0d4f2e, #1a7a4a) !important;
        border: 1px solid #2ecc71 !important;
    }
    .fault-warning {
        background: linear-gradient(135deg, #4a3800, #6d5200) !important;
        border: 1px solid #f39c12 !important;
    }
    .fault-danger {
        background: linear-gradient(135deg, #4a0e0e, #6d1414) !important;
        border: 1px solid #e74c3c !important;
    }
    .fault-info {
        background: linear-gradient(135deg, #0e2a4a, #143d6d) !important;
        border: 1px solid #3498db !important;
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d2ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 0.3rem;
    }

    .action-continue {
        color: #2ecc71; font-weight: 700; font-size: 1.4rem;
    }
    .action-schedule {
        color: #f39c12; font-weight: 700; font-size: 1.4rem;
    }
    .action-emergency {
        color: #e74c3c; font-weight: 700; font-size: 1.4rem;
    }

    .sidebar .stSlider > div > div > div {
        background-color: #3a7bd5;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('models/trained_models.pkl', 'rb') as f:
        return pickle.load(f)

try:
    data = load_models()
except FileNotFoundError:
    st.error("⚠️ Models not found! Run `python save_models.py` first.")
    st.stop()

scaler = data['scaler']
scaler_reg = data['scaler_reg']
le = data['label_encoder']
class_names = data['class_names']
feature_cols = data['feature_cols']
feature_stats = data['feature_stats']
models = data['models']
regression = data['regression']
Q = data['q_table']

state_names = ['Normal', 'Low Fault', 'Medium Fault', 'High Fault']
action_names = ['✅ Continue Operating', '🔧 Schedule Maintenance', '🚨 Emergency Stop']
action_css = ['action-continue', 'action-schedule', 'action-emergency']

severity_to_state = {0.0: 0, 0.007: 1, 0.014: 2, 0.021: 3}


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Predictive Maintenance Dashboard</h1>
    <p>Real-Time Bearing Fault Detection for Industrial Robotic Arms — CWRU Dataset</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — INPUT
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Sensor Input Values")
    st.markdown("Adjust the vibration signal features below or load a sample.")

    # Model selector
    selected_model = st.selectbox(
        "🧠 Select Classifier",
        list(models.keys()),
        index=0,
        help="Choose which trained ML model to use for prediction"
    )

    st.markdown("---")

    # Sample loader
    sample_type = st.selectbox(
        "📋 Load Sample Data",
        ["Custom Input", "Normal Bearing", "Ball Fault (0.007)",
         "Ball Fault (0.021)", "Inner Race Fault (0.014)",
         "Outer Race Fault (0.021)"]
    )

    # Define sample values
    samples = {
        "Normal Bearing": [0.21, -0.22, 0.012, 0.065, 0.066, -0.15, -0.07, 3.05, 5.43],
        "Ball Fault (0.007)": [0.50, -0.45, 0.019, 0.140, 0.141, 0.05, 0.10, 3.40, 7.50],
        "Ball Fault (0.021)": [0.65, -0.60, 0.013, 0.185, 0.186, 0.03, -0.25, 3.10, 14.50],
        "Inner Race Fault (0.014)": [1.45, -1.35, 0.022, 0.280, 0.281, -0.10, 4.20, 5.15, 12.80],
        "Outer Race Fault (0.021)": [2.50, -2.40, 0.015, 0.560, 0.561, 0.02, 1.20, 4.45, 37.00],
    }

    st.markdown("---")
    st.markdown("### 📊 Feature Values")

    input_values = {}
    feature_descriptions = {
        'max': 'Maximum amplitude',
        'min': 'Minimum amplitude',
        'mean': 'Mean value',
        'sd': 'Standard deviation',
        'rms': 'Root mean square',
        'skewness': 'Signal skewness',
        'kurtosis': 'Signal kurtosis',
        'crest': 'Crest factor',
        'form': 'Form factor'
    }

    for i, col in enumerate(feature_cols):
        stats = feature_stats[col]

        if sample_type != "Custom Input":
            default_val = samples[sample_type][i]
        else:
            default_val = stats['mean']

        input_values[col] = st.slider(
            f"**{col.upper()}** — {feature_descriptions[col]}",
            min_value=float(stats['min'] - abs(stats['min']) * 0.5),
            max_value=float(stats['max'] + abs(stats['max']) * 0.5),
            value=float(default_val),
            step=0.001,
            format="%.4f"
        )

    predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────
# MAIN — PREDICTION
# ─────────────────────────────────────────────────────────────
# Always run prediction (either on button or on input change)
X_input = np.array([[input_values[c] for c in feature_cols]])
X_input_sc = scaler.transform(X_input)

# Classification
model = models[selected_model]
prediction = model.predict(X_input_sc)[0]
predicted_class = class_names[prediction]

# Get probabilities if available
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(X_input_sc)[0]
else:
    probabilities = np.zeros(len(class_names))
    probabilities[prediction] = 1.0

# Severity regression
if predicted_class != 'Normal':
    X_reg_sc = scaler_reg.transform(X_input)
    severity_pred = float(regression.predict(X_reg_sc)[0])
    severity_pred = max(0.007, min(0.021, severity_pred))  # clamp
else:
    severity_pred = 0.0

# Q-Learning recommendation
if severity_pred <= 0.001:
    q_state = 0
elif severity_pred <= 0.01:
    q_state = 1
elif severity_pred <= 0.017:
    q_state = 2
else:
    q_state = 3

best_action = int(np.argmax(Q[q_state]))

# Confidence
confidence = float(probabilities[prediction]) * 100


# ─────────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1.2, 1, 1])

# Card class based on prediction
if predicted_class == 'Normal':
    card_class = 'fault-normal'
    emoji = '✅'
    severity_text = 'No Fault Detected'
elif predicted_class == 'Ball':
    card_class = 'fault-info'
    emoji = '🔵'
    severity_text = f'{severity_pred:.3f}" diameter'
elif predicted_class == 'InnerRace':
    card_class = 'fault-warning'
    emoji = '🟠'
    severity_text = f'{severity_pred:.3f}" diameter'
else:
    card_class = 'fault-danger'
    emoji = '🔴'
    severity_text = f'{severity_pred:.3f}" diameter'

with col1:
    st.markdown(f"""
    <div class="prediction-card {card_class}">
        <div style="font-size: 3.5rem;">{emoji}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">
            {predicted_class.replace('InnerRace', 'Inner Race').replace('OuterRace', 'Outer Race')} Fault
        </div>
        <div style="font-size: 1rem; color: #a0aec0;">
            Detected by {selected_model}
        </div>
        <div style="margin-top: 1rem;">
            <span style="font-size: 1.4rem; font-weight: 600; color: #00d2ff;">
                {confidence:.1f}% Confidence
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="prediction-card">
        <div style="font-size: 2.5rem;">📏</div>
        <div style="font-size: 1rem; color: #a0aec0; margin: 0.5rem 0;">Fault Severity</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: white;">
            {severity_text}
        </div>
        <div style="font-size: 0.9rem; color: #a0aec0; margin-top: 0.5rem;">
            State: {state_names[q_state]}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="prediction-card">
        <div style="font-size: 2.5rem;">🛠️</div>
        <div style="font-size: 1rem; color: #a0aec0; margin: 0.5rem 0;">Recommended Action</div>
        <div class="{action_css[best_action]}">
            {action_names[best_action]}
        </div>
        <div style="font-size: 0.85rem; color: #a0aec0; margin-top: 0.5rem;">
            Q-Learning Policy Decision
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ROBOTIC ARM + BEARING ANIMATION (rendered via components.html)
# ─────────────────────────────────────────────────────────────
import streamlit.components.v1 as components

st.markdown("---")
st.markdown("### 🦾 Robotic Arm — Bearing Fault Visualization")

# Determine fault colors and animations
if predicted_class == 'Normal':
    outer_color = "#2ecc71"; inner_color = "#2ecc71"; ball_color = "#2ecc71"
    outer_anim = ""; inner_anim = ""; ball_anim = ""
    status_text = "ALL SYSTEMS NORMAL"; status_color = "#2ecc71"
    bearing_label = "Bearing Healthy"
elif predicted_class == 'OuterRace':
    outer_color = "#e74c3c"; inner_color = "#4a90a4"; ball_color = "#4a90a4"
    outer_anim = "animation: pulse-fault 1s ease-in-out infinite;"
    inner_anim = ""; ball_anim = ""
    status_text = "OUTER RACE FAULT DETECTED"; status_color = "#e74c3c"
    bearing_label = "Outer Race Damaged"
elif predicted_class == 'InnerRace':
    outer_color = "#4a90a4"; inner_color = "#e74c3c"; ball_color = "#4a90a4"
    outer_anim = ""; ball_anim = ""
    inner_anim = "animation: pulse-fault 1s ease-in-out infinite;"
    status_text = "INNER RACE FAULT DETECTED"; status_color = "#f39c12"
    bearing_label = "Inner Race Damaged"
else:  # Ball
    outer_color = "#4a90a4"; inner_color = "#4a90a4"; ball_color = "#e74c3c"
    outer_anim = ""; inner_anim = ""
    ball_anim = "animation: pulse-fault 0.6s ease-in-out infinite;"
    status_text = "BALL FAULT DETECTED"; status_color = "#3498db"
    bearing_label = "Rolling Element Damaged"

# Severity bar
sev_pct = min(100, (severity_pred / 0.025) * 100) if severity_pred > 0 else 0
sev_bar_color = "#2ecc71" if sev_pct < 40 else ("#f39c12" if sev_pct < 75 else "#e74c3c")

# Pre-compute ball rotation style
ball_group_style = "animation: rotate-balls 3s linear infinite; transform-origin: 0px 0px;" if predicted_class == 'Normal' else "transform-origin: 0px 0px;"

# Component status cards
outer_status = '⚠️ FAULT' if predicted_class == 'OuterRace' else '✅ OK'
inner_status = '⚠️ FAULT' if predicted_class == 'InnerRace' else '✅ OK'
ball_status = '⚠️ FAULT' if predicted_class == 'Ball' else '✅ OK'

arm_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: transparent; font-family: 'Inter', 'Segoe UI', sans-serif; }}
    @keyframes pulse-fault {{
        0%, 100% {{ opacity: 1; filter: brightness(1); }}
        50% {{ opacity: 0.4; filter: brightness(2); }}
    }}
    @keyframes rotate-balls {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    @keyframes arm-sway {{
        0%, 100% {{ transform: rotate(-2deg); }}
        50% {{ transform: rotate(2deg); }}
    }}
    .wrapper {{
        display: flex;
        gap: 1.5rem;
        align-items: stretch;
        padding: 0.5rem;
    }}
    .svg-col {{
        flex: 1.3;
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .detail-col {{
        flex: 1;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
    }}
    .status-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin-top: 1rem;
    }}
    .status-card {{
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }}
    .sev-bar-bg {{
        background: #2c3e50;
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }}
    .sev-bar-fill {{
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s;
    }}
</style>
</head>
<body>
<div class="wrapper">
    <!-- SVG ARM -->
    <div class="svg-col">
        <svg width="500" height="380" viewBox="0 0 520 380" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <radialGradient id="mg" cx="50%" cy="40%" r="60%">
                    <stop offset="0%" stop-color="#6c7a89"/>
                    <stop offset="100%" stop-color="#2c3e50"/>
                </radialGradient>
                <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#5d6d7e"/>
                    <stop offset="100%" stop-color="#2c3e50"/>
                </linearGradient>
                <filter id="gl"><feGaussianBlur stdDeviation="3" result="b"/>
                    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
            </defs>

            <!-- BASE -->
            <rect x="20" y="320" width="140" height="40" rx="8" fill="#1a1a2e" stroke="#34495e" stroke-width="2"/>
            <rect x="50" y="300" width="80" height="25" rx="4" fill="url(#ag)" stroke="#4a5568" stroke-width="1.5"/>
            <text x="90" y="350" text-anchor="middle" fill="#a0aec0" font-size="11">BASE</text>

            <!-- SHOULDER -->
            <circle cx="90" cy="290" r="16" fill="url(#mg)" stroke="#4a5568" stroke-width="2"/>
            <circle cx="90" cy="290" r="6" fill="#1a1a2e"/>

            <!-- ARM GROUP -->
            <g style="transform-origin: 90px 290px; animation: arm-sway 4s ease-in-out infinite;">
                <rect x="75" y="170" width="30" height="120" rx="6" fill="url(#ag)" stroke="#4a5568" stroke-width="1.5"/>
                <circle cx="90" cy="160" r="18" fill="url(#mg)" stroke="#4a5568" stroke-width="2"/>
                <rect x="85" y="80" width="120" height="26" rx="6" fill="url(#ag)" stroke="#4a5568" stroke-width="1.5" transform="rotate(-25, 90, 160)"/>
                <circle cx="195" cy="98" r="10" fill="url(#mg)" stroke="#4a5568" stroke-width="1.5"/>
                <rect x="195" y="85" width="35" height="8" rx="3" fill="#5d6d7e" stroke="#4a5568" stroke-width="1" transform="rotate(-25, 195, 98)"/>
                <rect x="195" y="102" width="35" height="8" rx="3" fill="#5d6d7e" stroke="#4a5568" stroke-width="1" transform="rotate(-25, 195, 98)"/>
            </g>

            <!-- CONNECTORS -->
            <line x1="108" y1="160" x2="260" y2="130" stroke="#4a5568" stroke-width="1" stroke-dasharray="5,5" opacity="0.6"/>
            <line x1="108" y1="160" x2="260" y2="260" stroke="#4a5568" stroke-width="1" stroke-dasharray="5,5" opacity="0.6"/>

            <!-- BEARING -->
            <g transform="translate(370, 195)">
                <circle cx="0" cy="0" r="85" fill="none" stroke="{outer_color}" stroke-width="14" style="{outer_anim}" filter="url(#gl)"/>
                <circle cx="0" cy="0" r="85" fill="none" stroke="{outer_color}" stroke-width="2" opacity="0.3"/>
                <circle cx="0" cy="0" r="35" fill="none" stroke="{inner_color}" stroke-width="12" style="{inner_anim}" filter="url(#gl)"/>
                <circle cx="0" cy="0" r="20" fill="#1a1a2e" stroke="#4a5568" stroke-width="1"/>
                <circle cx="0" cy="0" r="12" fill="#0f0c29" stroke="#2c3e50" stroke-width="1"/>

                <g style="{ball_group_style}">
                    <circle cx="0" cy="-60" r="11" fill="{ball_color}" stroke="#2c3e50" stroke-width="1.5" style="{ball_anim}" filter="url(#gl)"/>
                    <circle cx="52" cy="-30" r="11" fill="{ball_color}" stroke="#2c3e50" stroke-width="1.5" style="{ball_anim}" filter="url(#gl)"/>
                    <circle cx="52" cy="30" r="11" fill="{ball_color}" stroke="#2c3e50" stroke-width="1.5" style="{ball_anim}" filter="url(#gl)"/>
                    <circle cx="0" cy="60" r="11" fill="{ball_color}" stroke="#2c3e50" stroke-width="1.5" style="{ball_anim}" filter="url(#gl)"/>
                    <circle cx="-52" cy="30" r="11" fill="{ball_color}" stroke="#2c3e50" stroke-width="1.5" style="{ball_anim}" filter="url(#gl)"/>
                    <circle cx="-52" cy="-30" r="11" fill="{ball_color}" stroke="#2c3e50" stroke-width="1.5" style="{ball_anim}" filter="url(#gl)"/>
                </g>

                <text x="0" y="-100" text-anchor="middle" fill="{outer_color}" font-size="10" font-weight="600">OUTER RACE</text>
                <text x="0" y="110" text-anchor="middle" fill="{inner_color}" font-size="10" font-weight="600">INNER RACE</text>
                <text x="85" y="-55" text-anchor="start" fill="{ball_color}" font-size="9">BALLS</text>
                <line x1="63" y1="-30" x2="80" y2="-52" stroke="{ball_color}" stroke-width="1"/>
            </g>

            <text x="370" y="360" text-anchor="middle" fill="{status_color}" font-size="13" font-weight="700">{bearing_label}</text>
        </svg>
    </div>

    <!-- DETAIL PANEL -->
    <div class="detail-col">
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div style="font-size:1rem; color:#a0aec0;">System Status</div>
            <div style="font-size:1.1rem; font-weight:700; color:{status_color}; margin-top:0.3rem;">
                {status_text}
            </div>
        </div>

        <div style="margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                <span style="color:#a0aec0; font-size:0.85rem;">Severity Level</span>
                <span style="color:white; font-weight:600; font-size:0.85rem;">{severity_pred:.4f}"</span>
            </div>
            <div class="sev-bar-bg">
                <div class="sev-bar-fill" style="background:{sev_bar_color}; width:{sev_pct}%;"></div>
            </div>
        </div>

        <div class="status-grid">
            <div class="status-card" style="border-left:3px solid {outer_color};">
                <div style="color:#a0aec0; font-size:0.75rem;">Outer Race</div>
                <div style="color:{outer_color}; font-weight:700; font-size:0.95rem;">{outer_status}</div>
            </div>
            <div class="status-card" style="border-left:3px solid {inner_color};">
                <div style="color:#a0aec0; font-size:0.75rem;">Inner Race</div>
                <div style="color:{inner_color}; font-weight:700; font-size:0.95rem;">{inner_status}</div>
            </div>
            <div class="status-card" style="border-left:3px solid {ball_color};">
                <div style="color:#a0aec0; font-size:0.75rem;">Rolling Elements</div>
                <div style="color:{ball_color}; font-weight:700; font-size:0.95rem;">{ball_status}</div>
            </div>
            <div class="status-card" style="border-left:3px solid #2ecc71;">
                <div style="color:#a0aec0; font-size:0.75rem;">Cage / Housing</div>
                <div style="color:#2ecc71; font-weight:700; font-size:0.95rem;">✅ OK</div>
            </div>
        </div>

        <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.1);">
            <div style="color:#a0aec0; font-size:0.8rem; text-align:center;">
                🔍 Bearing cross-section shows real-time fault location<br>
                <span style="color:#2ecc71;">Green = Healthy</span> •
                <span style="color:#e74c3c;">Red = Fault Detected</span>
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""

components.html(arm_html, height=420)


# ─────────────────────────────────────────────────────────────
# PROBABILITY & CHARTS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📊 Class Probabilities")

    prob_df = pd.DataFrame({
        'Fault Type': [c.replace('InnerRace', 'Inner Race').replace('OuterRace', 'Outer Race') for c in class_names],
        'Probability': probabilities
    }).sort_values('Probability', ascending=True)

    colors = []
    for ft in prob_df['Fault Type']:
        if 'Normal' in ft: colors.append('#2ecc71')
        elif 'Ball' in ft: colors.append('#3498db')
        elif 'Inner' in ft: colors.append('#f39c12')
        else: colors.append('#e74c3c')

    fig = go.Figure(go.Bar(
        x=prob_df['Probability'],
        y=prob_df['Fault Type'],
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1%}' for p in prob_df['Probability']],
        textposition='outside',
        textfont=dict(size=14, color='white')
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=13),
        xaxis=dict(range=[0, 1.15], gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=300,
        margin=dict(l=0, r=30, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### 🕸️ Feature Radar")

    # Normalize inputs to 0-1 for radar
    norm_vals = []
    for col in feature_cols:
        s = feature_stats[col]
        rng = s['max'] - s['min']
        if rng == 0: rng = 1
        norm_vals.append((input_values[col] - s['min']) / rng)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=[c.upper() for c in feature_cols] + [feature_cols[0].upper()],
        fill='toself',
        fillcolor='rgba(0, 210, 255, 0.15)',
        line=dict(color='#00d2ff', width=2),
        name='Input'
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1.2], gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        height=300,
        margin=dict(l=40, r=40, t=10, b=10),
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ALL MODELS COMPARISON
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔄 All Models Prediction Comparison")

compare_data = []
for name, m in models.items():
    pred = m.predict(X_input_sc)[0]
    pred_name = class_names[pred].replace('InnerRace', 'Inner Race').replace('OuterRace', 'Outer Race')
    if hasattr(m, 'predict_proba'):
        conf = f"{m.predict_proba(X_input_sc)[0][pred] * 100:.1f}%"
    else:
        conf = "N/A"
    marker = " 👈" if name == selected_model else ""
    compare_data.append({
        'Model': name + marker,
        'Prediction': pred_name,
        'Confidence': conf
    })

compare_df = pd.DataFrame(compare_data)
st.dataframe(compare_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# Q-TABLE
# ─────────────────────────────────────────────────────────────
with st.expander("🧠 Q-Learning Decision Matrix"):
    q_df = pd.DataFrame(
        Q,
        index=state_names,
        columns=['Continue', 'Schedule Maint.', 'Emergency Stop']
    )
    st.dataframe(q_df.style.format("{:.1f}").background_gradient(cmap='RdYlGn', axis=1),
                 use_container_width=True)

    st.markdown(f"""
    **Current State:** `{state_names[q_state]}` → **Action:** `{action_names[best_action]}`

    The Q-Learning agent was trained over 10,000 episodes to learn the optimal
    maintenance policy. Higher Q-values (green) indicate preferred actions for each state.
    """)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4a5568; font-size: 0.85rem; padding: 1rem;">
    Predictive Maintenance for Industrial Robotic Arms — CWRU Bearing Dataset<br>
    Built with Streamlit • Scikit-learn • Plotly
</div>
""", unsafe_allow_html=True)
