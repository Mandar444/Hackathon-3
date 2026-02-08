import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# --- SETTINGS ---
st.set_page_config(
    page_title="HACKATHON v4.5 AI",
    layout="wide",
    page_icon="💧"
)

# --- THEME INJECTION ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;500&family=Orbitron:wght@400;700&family=Inter:wght@300;400;700&display=swap');
    .main { background-color: #0b0e14; color: #d1d5db; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #3b82f6; text-transform: uppercase; letter-spacing: 2px; }
    .hud-container { background: #1f2937; border: 2px solid #3b82f6; border-radius: 15px; padding: 40px; text-align: center; margin: 20px 0; }
    .hud-value { font-family: 'JetBrains Mono', monospace; font-size: 84px; color: #60a5fa; font-weight: 800; line-height: 1; }
    .hud-label { font-family: 'Orbitron', sans-serif; color: #9ca3af; font-size: 16px; margin-bottom: 10px;}
    .graph-explanation { background: rgba(31, 41, 55, 0.5); padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6; font-size: 13px; color: #9ca3af; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# --- CONFIG & MAPPINGS ---
BUILDING_MAP = {'Hostel': 0, 'Academic': 1, 'Lab': 2}
DAY_MAP = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
PHASE_MAP = {'Normal': 0, 'Exam': 1, 'Vacation': 2}

def get_db_connection():
    return sqlite3.connect('campus_water.db')

@st.cache_data(ttl=60)
def load_historical_data():
    if not os.path.exists('campus_water.db'): return None
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM water_records", conn)
    conn.close()
    return df

@st.cache_resource
def load_production_model():
    model_path = 'models/water_model.pkl'
    if not os.path.exists(model_path):
        return None, None
    data = joblib.load(model_path)
    return data['model'], data['metadata']

# --- INITIALIZE SYSTEM ---
df = load_historical_data()
model, metadata = load_production_model()

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.markdown("<h2 style='color: #3b82f6;'>🛡️ HACKATHON v4.5</h2>", unsafe_allow_html=True)
    st.caption("PRODUCTION DEPLOYMENT")
    st.markdown("---")
    choice = st.radio("OPERATIONAL MODES", ["🌐 NETWORK OVERVIEW", "🔮 PREDICTION ENGINE", "📂 SYSTEM LOGS"])
    st.markdown("---")
    
    if model:
        st.success("✅ AI Engine: Online")
        st.info(f"📅 Deployed: {metadata['train_date']}")
        st.info(f"📊 Verified Accuracy: {metadata['accuracy']*100:.1f}%")
    else:
        st.error("❌ AI Model Offline")
        st.warning("Run train_model.py to deploy.")

# --- UI LOGIC ---
if df is None:
    st.error("🚨 DATABASE NOT INITIALIZED. Run data_generation.py.")
else:
    if choice == "🌐 NETWORK OVERVIEW":
        st.title("🌐 CAMPUS INFRASTRUCTURE TELEMETRY")
        
        # Summary
        c1, c2, c3 = st.columns(3)
        c1.metric("AVG HOURLY LOAD", f"{df['consumption_liters'].mean():.0f} L")
        c2.metric("CURRENT SECTORS", "3 ACTIVE")
        c3.metric("TELEMETRY NODES", f"{len(df)} RECORDED")

        st.markdown("---")
        
        # --- ROW 1: HEATMAP ---
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            st.subheader("🔥 CONSUMPTION INTENSITY (DAY VS HOUR)")
            pivot_df = df.pivot_table(index='day_of_week', columns='time_of_day', values='consumption_liters', aggfunc='mean')
            pivot_df = pivot_df.reindex(list(DAY_MAP.keys()))
            fig_heat = px.imshow(pivot_df, labels=dict(x="Hour", y="Day", color="Liters"),
                               color_continuous_scale='Viridis', template="plotly_dark")
            fig_heat.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_heat, use_container_width=True)
        with col_h2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="graph-explanation">
                    <b>🔍 STRATEGIC INSIGHT:</b> This heatmap identifies <b>Extreme Stress Windows</b>. 
                    Yellow zones reveal exactly when the infrastructure is under peak demand. 
                    Use this to schedule tank refills 2 hours <i>before</i> a hot zone hits.
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- ROW 2: BOXPLOT & SCATTER ---
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.subheader("📊 SECTOR VARIANCE")
            fig_box = px.box(df, x='building_type', y='consumption_liters', color='building_type',
                            template="plotly_dark", color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'])
            fig_box.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("""
                <div class="graph-explanation">
                    <b>📉 WHY THIS MATTERS:</b> The Box Plot shows the <b>Reliability</b> of data. 
                    Wide boxes mean unpredictable usage. Narrower boxes represent stable, controllable consumption.
                </div>
            """, unsafe_allow_html=True)

        with col_b2:
            st.subheader("👥 PEOPLE VS WATER CORRELATION")
            fig_scatter = px.scatter(df.sample(min(1000, len(df))), x='occupancy_percentage', y='consumption_liters', 
                                    color='building_type', size='consumption_liters',
                                    template="plotly_dark", opacity=0.6)
            fig_scatter.update_layout(height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("""
                <div class="graph-explanation">
                    <b>🧬 PROOF OF CAUSE:</b> This scatter proves that <b>Occupancy</b> is the direct driver. 
                    If dots appear at 5000L with 10% occupancy, it is <b>Undisputable Proof of Leakage</b>.
                </div>
            """, unsafe_allow_html=True)

    elif choice == "🔮 PREDICTION ENGINE":
        st.title("🔮 AI FORECASTING CORE")
        
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                b_type = st.selectbox("🎯 SECTOR", list(BUILDING_MAP.keys()))
                day = st.selectbox("📅 DAY", list(DAY_MAP.keys()))
            with c2:
                phase = st.radio("🎓 PHASE", list(PHASE_MAP.keys()), horizontal=True)
                occ = st.slider("👥 OCCUPANCY %", 0, 100, 75)
            with c3:
                hour = st.select_slider("🕒 TIME WINDOW", options=list(range(24)), value=12,
                                       format_func=lambda x: f"{x}:00 to {x+1}:00")
                run = st.button("🚀 EXECUTE NEURAL INFERENCE")

        if run:
            if model is None:
                st.error("❌ Deployment Error: Primary brain (models/water_model.pkl) not found.")
            else:
                input_data = pd.DataFrame([{
                    'building_code': BUILDING_MAP[b_type], 'day_code': DAY_MAP[day], 
                    'phase_code': PHASE_MAP[phase], 'occupancy_percentage': occ, 'time_of_day': hour
                }])
                prediction = model.predict(input_data)[0]
                
                # --- HUD ---
                st.markdown(f"""
                    <div class="hud-container">
                        <div class="hud-label">ESTIMATED WATER LOAD</div>
                        <div class="hud-value">{prediction:.2f}L</div>
                        <div class="hud-label">🛡️ HACKATHON v4.5 ENGINE INTEGRITY (R2 {metadata['accuracy']:.2f})</div>
                    </div>
                """, unsafe_allow_html=True)

                # --- NEW: COMPARATIVE ANALYSIS & FEATURE IMPORTANCE ---
                e_col1, e_col2 = st.columns(2)
                
                with e_col1:
                    st.subheader("🏛️ HISTORICAL VS AI")
                    # Find nearest history match
                    similar = df[(df['building_type'] == b_type) & 
                                (df['academic_phase'] == phase) &
                                (abs(df['time_of_day'] - hour) <= 1)].copy()
                    
                    if not similar.empty:
                        similar['occ_diff'] = abs(similar['occupancy_percentage'] - occ)
                        best_match = similar.sort_values('occ_diff').iloc[0]
                        
                        fig_compare = go.Figure(data=[
                            go.Bar(name='AI Prediction', x=['Consumption'], y=[prediction], marker_color='#3b82f6'),
                            go.Bar(name='Past Reality', x=['Consumption'], y=[best_match['consumption_liters']], marker_color='#10b981')
                        ])
                        fig_compare.update_layout(template="plotly_dark", height=300, barmode='group',
                                                title="AI vs Closest Historical Record")
                        st.plotly_chart(fig_compare, use_container_width=True)
                    else:
                        st.info("No identical historical conditions found. AI is synthesizing result.")

                with e_col2:
                    st.subheader("🧩 DECISION DRIVERS")
                    # Get feature importance from the pre-trained model metadata if it's there, 
                    # else calculate it or use a default importance visual.
                    # Note: RandomForestRegressor.feature_importances_
                    importances = model.feature_importances_
                    feat_names = ['Sector', 'Day', 'Phase', 'People', 'Time']
                    fig_imp = px.bar(x=feat_names, y=importances, labels={'x':'Factor', 'y':'Weight'},
                                    template="plotly_dark", color_discrete_sequence=['#3b82f6'])
                    fig_imp.update_layout(height=300, title="Why this number? (Feature Weighting)")
                    st.plotly_chart(fig_imp, use_container_width=True)

                st.info(f"💡 AI Insight: The predicted load of {prediction:.0f}L is triggered by {b_type} activity patterns during the {phase} cycle.")

    elif choice == "📂 SYSTEM LOGS":
        st.title("📂 SQL DATA LEDGER")
        st.markdown("Direct read from `campus_water.db` repository.")
        st.dataframe(df.tail(200), use_container_width=True)
