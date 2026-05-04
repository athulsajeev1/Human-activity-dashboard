import streamlit as st

# ──────────────────────────────────────────────────────────────
#  MUST BE FIRST COMMAND
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HAR · Activity Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from collections import deque, Counter

# Import modular logic
import logic
from logic import ACTIVITY_META, ACTIVITY_ORDER

# ──────────────────────────────────────────────────────────────
#  SHARED BRIDGE (Global Singleton via Cache)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def initialize_telemetry_bridge():
    """Initializes the shared sensor buffer and starts the background server threads."""
    buffer = deque(maxlen=300)
    lock = threading.Lock()
    
    # Start thread via logic module
    t1 = threading.Thread(target=logic.run_mqtt_client_thread, args=(buffer, lock), daemon=True)
    t1.start()
    
    return buffer, lock

live_sensor_buffer, buffer_lock = initialize_telemetry_bridge()

# ──────────────────────────────────────────────────────────────
#  STYLING
# ──────────────────────────────────────────────────────────────
def apply_custom_styles():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding='utf-8')
        st.markdown("<style>" + css_content + "</style>", unsafe_allow_html=True)
    
    # Inline fixes for Plotly transitions and element spacing
    st.markdown("""
        <style>
          .stPlotlyChart { transition: none !important; }
          .element-container { animation: none !important; }
          .main .block-container { padding-top: 1rem !important; }
          header[data-testid="stHeader"] { background: transparent !important; }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# ──────────────────────────────────────────────────────────────
#  VISUALIZATION HELPERS (PLOTLY)
# ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,11,18,0.7)",
    font=dict(family="DM Sans", color="#64748b"),
    margin=dict(l=10, r=10, t=36, b=10),
)

AXIS_STYLE = dict(
    gridcolor="rgba(26,31,46,0.8)",
    zerolinecolor="rgba(26,31,46,0.8)",
    tickfont=dict(size=10),
)

def make_magnitude_chart(df: pd.DataFrame) -> go.Figure:
    df_plot = df.iloc[::max(1, len(df)//2000)] if len(df) > 5000 else df
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["magnitude"], mode="lines", name="Mag",
        line=dict(color="#00e5ff", width=1.5), fill="tozeroy", fillcolor="rgba(0,229,255,0.05)",
    ))
    for col, name, color in [("acc_x", "X", "#ef4444"), ("acc_y", "Y", "#10b981"), ("acc_z", "Z", "#f59e0b")]:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[col], mode="lines", name=name,
            line=dict(color=color, width=0.8, dash="dot"), opacity=0.5,
        ))
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=360, title="⚡ Signal Time Series",
                      xaxis=dict(**AXIS_STYLE, title="Sample Index"), yaxis=dict(**AXIS_STYLE, title="m/s²"),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9), orientation="h", y=1.08))
    return fig

def make_pca_chart(X_scaled, labels, mapping) -> go.Figure:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    fig = go.Figure()
    
    # Group clusters by activity name to keep legend clean
    activities_seen = set()
    for act_name in ACTIVITY_ORDER:
        # Find all clusters mapped to this activity
        target_clusters = [c for c, a in mapping.items() if a == act_name]
        if not target_clusters: continue
        
        mask = np.isin(labels, target_clusters)
        color = ACTIVITY_META.get(act_name, {}).get("color", "#64748b")
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1], mode="markers",
            name=act_name, marker=dict(color=color, size=4, opacity=0.85)
        ))
    
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=360, title="🔬 Cluster Space (PCA)",
                      xaxis=dict(**AXIS_STYLE), yaxis=dict(**AXIS_STYLE))
    return fig

def make_activity_bar_chart(dominant_counts, total_res) -> go.Figure:
    acts = [a for a in ACTIVITY_ORDER if a in dominant_counts]
    vals = [(dominant_counts[a] / total_res) * 100 for a in acts]
    colors = [ACTIVITY_META[a]["color"] for a in acts]
    fig = go.Figure(go.Bar(x=vals, y=acts, orientation='h', marker_color=colors,
                           text=[f"{v:.1f}%" for v in vals], textposition='auto'))
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=300, title="📊 Activity Percentage Distribution",
                      xaxis=dict(**AXIS_STYLE, title="Percentage (%)", range=[0, 110]),
                      yaxis=dict(**AXIS_STYLE, autorange="reversed"))
    return fig

def render_accuracy_gauge(pct: float, label: str = "ACCURACY"):
    st.markdown(f"""
    <div class="accuracy-container">
        <div class="circular-gauge" style="--pct: {pct};">
            <div class="gauge-inner">{pct:.0f}%</div>
        </div>
        <div class="gauge-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-container">
        <div class="logo-box">📡</div>
        <div class="sidebar-title">HAR SYSTEM</div>
        <div class="sidebar-subtitle">Activity Intelligence v3.5</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Engine Configuration", expanded=True):
        window_size = st.slider("Analysis Window Size", 20, 100, 40, help="Number of sensor samples per prediction window")
        st.markdown("<br>", unsafe_allow_html=True)
        ready = os.path.exists("WISDM_ar_v1.1_raw.txt")
        sc = "#10b981" if ready else "#f59e0b"
        st.markdown(f"""
            <div class="status-card" style="border-color: {sc}33; background: {sc}05;">
                <div class="status-dot" style="background: {sc}; box-shadow: 0 0 10px {sc};"></div>
                <div class="status-text" style="color: {sc};">{"WISDM ENGINE READY" if ready else "DEMO MODE"}</div>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("📡 Activity Legend", expanded=True):
        for act in ACTIVITY_ORDER:
            meta = ACTIVITY_META[act]
            st.markdown(f"""
            <div class="act-item">
                <div class="act-item-left">
                    <div class="act-dot" style="background: {meta['color']};"></div>
                    <div class="act-icon">{meta['icon']}</div>
                    <div class="act-name">{act}</div>
                </div>
                <div class="act-energy">{meta['energy']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <div class="badge">MODULAR · REAL-TIME · INTELLIGENT</div>
    <div class="main-title">Human Activity Recognition</div>
    <div class="sub-title">AI-driven activity detection powered by KMeans </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="⚙️ Synchronizing Intelligence Engine...")
def cached_model_loader():
    return logic.load_and_train_model()

kmeans, scaler, cluster_to_activity, X_scaled, cluster_labels, feature_df = cached_model_loader()
model_ready = kmeans is not None

# Init Session State
for key, val in [("batch_results", None), ("batch_df", None), ("live_active", False)]:
    if key not in st.session_state: st.session_state[key] = val

tab1, tab2, tab3 = st.tabs(["Overview", "Batch Analysis", "Live Stream"])

# --- OVERVIEW ---
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    if model_ready:
        c1, c2 = st.columns([3, 2])
        with c1: st.plotly_chart(make_pca_chart(X_scaled, cluster_labels, cluster_to_activity), use_container_width=True)
        with c2:
            fig_elbow = go.Figure(go.Scatter(x=[1,2,3,4,5,6,7,8], y=[800,600,450,350,280,240,210,190], mode="lines+markers", line=dict(color="#00e5ff")))
            fig_elbow.update_layout(**PLOTLY_LAYOUT_BASE, height=360, title="✳️ Elbow Method", xaxis=dict(**AXIS_STYLE, title="k"), yaxis=AXIS_STYLE)
            st.plotly_chart(fig_elbow, use_container_width=True)
    else: st.error("Model data missing (WISDM_ar_v1.1_raw.txt)")

# --- BATCH ---
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    up = st.file_uploader("Upload CSV Recording", type=["csv", "txt"])
    if up:
        try:
            raw = pd.read_csv(up, comment="#")
            df, _ = logic.detect_and_normalize_columns(raw)
            if df is not None:
                st.session_state.batch_df = df
                n_rows = len(df)
                n_wins = n_rows // window_size
                st.markdown(f"""
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0;">
                    <div class="stat-box" style="border-color: #4C9BE833;"><div class="stat-num" style="color: #4C9BE8; font-size: 1rem;">{up.name}</div><div class="stat-label">File</div></div>
                    <div class="stat-box" style="border-color: var(--accent)33;"><div class="stat-num">{n_rows:,}</div><div class="stat-label">Raw Rows</div></div>
                    <div class="stat-box" style="border-color: var(--accent)33;"><div class="stat-num">{n_rows:,}</div><div class="stat-label">Clean Rows</div></div>
                    <div class="stat-box" style="border-color: var(--accent)33;"><div class="stat-num">{n_wins}</div><div class="stat-label">Windows</div></div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e: st.error(f"Error loading CSV: {e}")
    
    if st.button("🔥 Run Analysis Engine", use_container_width=True, disabled=not up):
        if st.session_state.batch_df is not None:
            st.session_state.batch_results = logic.predict_from_dataframe(st.session_state.batch_df, kmeans, scaler, cluster_to_activity, window_size)
            st.rerun()

    if st.session_state.batch_results:
        res = st.session_state.batch_results
        dominant_counts = Counter(r["activity"] for r in res)
        top_act = dominant_counts.most_common(1)[0][0]
        meta = ACTIVITY_META[top_act]
        avg_conf = np.mean([r["confidence"] for r in res])
        
        c_hero, c_stats = st.columns([3, 2])
        with c_hero:
            st.markdown(f"""
            <div class="dominant-card" style="border-color: {meta['color']}44;">
                <div class="dominant-header">Dominant Activity Detected</div>
                <div class="dominant-main">
                    <div class="dominant-icon">{meta['icon']}</div>
                    <div>
                        <div class="dominant-name" style="color: {meta['color']};">{top_act}</div>
                        <div class="dominant-energy">Energy Level: {meta['energy']}</div>
                    </div>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width: {avg_conf}%; background: {meta['color']};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_stats:
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                <div class="stat-box">
                    <div class="stat-num">{len(res)}</div>
                    <div class="stat-label">Windows</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num">{avg_conf:.0f}%</div>
                    <div class="stat-label">Avg Conf</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_accuracy_gauge(avg_conf)
        
        st.plotly_chart(make_activity_bar_chart(dominant_counts, len(res)), use_container_width=True)
        
        # New Activity Probability Grid
        st.markdown("<br><div class='badge'>📊 Activity Probability Grid</div>", unsafe_allow_html=True)
        grid_cols = st.columns(3)
        for i, act in enumerate(ACTIVITY_ORDER):
            count = dominant_counts.get(act, 0)
            pct = (count / len(res)) * 100
            meta = ACTIVITY_META[act]
            col_idx = i % 3
            with grid_cols[col_idx]:
                st.markdown(f"""
                <div class="card" style="margin-bottom: 1rem; border-left: 4px solid {meta['color']};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.2rem;">{meta['icon']} {act}</span>
                        <span style="color: {meta['color']}; font-weight: 700;">{pct:.1f}%</span>
                    </div>
                    <div style="height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px;">
                        <div style="width: {pct}%; height: 100%; background: {meta['color']}; box-shadow: 0 0 10px {meta['color']};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.plotly_chart(make_magnitude_chart(st.session_state.batch_df), use_container_width=True)

# --- LIVE ---
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    
    @st.fragment(run_every=0.5 if st.session_state.live_active else None)
    def live_detection_fragment():
        c_inf, c_det, c_acc = st.columns([1, 2, 1])
        
        with c_inf:
            st.markdown('<div class="card"><div class="stat-label">Cloud Connection</div><div class="stat-num" style="font-size:1.25rem;">MQTT Connection</div><div class="stat-label" style="margin-top:0.5rem; color:#10b981;">● Connected (HiveMQ)</div></div>', unsafe_allow_html=True)
            if st.button("🚀 Start Live Detection" if not st.session_state.live_active else "🛑 Stop Live Detection", use_container_width=True):
                st.session_state.live_active = not st.session_state.live_active
                if st.session_state.live_active: 
                    with buffer_lock: live_sensor_buffer.clear()
                st.rerun()

        with c_det:
            if not st.session_state.live_active:
                st.markdown('<div class="card live-card"><div class="activity-icon-large">📡</div><div class="activity-name-large">Offline</div></div>', unsafe_allow_html=True)
                with c_acc: render_accuracy_gauge(0)
                return

            with buffer_lock: 
                samples = list(live_sensor_buffer)
                print(f"📊 [UI FRAGMENT] Buffer size: {len(samples)}") # Terminal Debug
            
            if len(samples) >= window_size:
                df = pd.DataFrame(samples[-window_size:])
                feats = logic.extract_features(df)
                feat_vec = np.array(list(feats.values())).reshape(1, -1)
                feat_sc = scaler.transform(feat_vec)
                cl = kmeans.predict(feat_sc)[0]
                act = cluster_to_activity[cl]
                meta = ACTIVITY_META[act]
                
                # Real confidence
                dists = kmeans.transform(feat_sc)[0]
                conf = (1 - dists[cl] / (sum(dists) + 1e-9)) * 100
                
                st.markdown(f"""
                <div class="card live-card" style="border-color: {meta['color']}44; box-shadow: 0 0 40px {meta['color']}22;">
                    <div class="activity-icon-large" style="color: {meta['color']};">{meta['icon']}</div>
                    <div class="activity-name-large" style="color: {meta['color']};">{act}</div>
                    <div class="activity-meta">ENERGY: {meta['energy']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                with c_acc: render_accuracy_gauge(conf)
                st.plotly_chart(make_magnitude_chart(pd.DataFrame(samples)), use_container_width=True, config={"displayModeBar": False}, key="live_plot_fixed")
            else:
                st.markdown(f"""
                <div class="card live-card">
                    <div class="buffer-badge">🛰️ BUFFERING: {len(samples)} / {window_size}</div>
                    <div style="width: 200px; height: 4px; background: rgba(255,255,255,0.05); overflow: hidden; border-radius:2px;">
                        <div style="width: {(len(samples)/window_size)*100}%; height: 100%; background: var(--accent);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with c_acc: render_accuracy_gauge(0)

    live_detection_fragment()