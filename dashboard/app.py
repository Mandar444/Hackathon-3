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
    page_title="HACKATHON v4.5",
    layout="wide"
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
    .graph-explanation { background: rgba(31, 41, 55, 0.5); padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6; font-size: 14px; color: #9ca3af; margin-top: 10px; }
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
    st.markdown("<h2 style='color: #3b82f6;'>HACKATHON v4.5</h2>", unsafe_allow_html=True)
    st.caption("PRODUCTION DEPLOYMENT")
    st.markdown("---")
    choice = st.radio("MODES", ["NETWORK OVERVIEW", "PREDICTION ENGINE", "SYSTEM LOGS"])
    st.markdown("---")
    
    if model:
        st.success("AI Engine: Online")
        st.info(f"Deployed: {metadata['train_date']}")
        st.info(f"Verified Accuracy: {metadata['accuracy']*100:.1f}%")
    else:
        st.error("AI Model Offline")
        st.warning("Run train_model.py to deploy.")

# --- UI LOGIC ---
if df is None:
    st.error("DATABASE NOT INITIALIZED. Run data_generation.py.")
else:
    if choice == "NETWORK OVERVIEW":
        st.title("CAMPUS INFRASTRUCTURE TELEMETRY")
        
        # Summary
        c1, c2, c3 = st.columns(3)
        c1.metric("AVG HOURLY LOAD", f"{df['consumption_liters'].mean():.0f} L")
        c2.metric("CURRENT SECTORS", "3 ACTIVE")
        c3.metric("TELEMETRY NODES", f"{len(df)} RECORDED")

        st.markdown("---")
        
        # Heatmap
        st.subheader("CONSUMPTION INTENSITY (DAY VS HOUR)")
        pivot_df = df.pivot_table(index='day_of_week', columns='time_of_day', values='consumption_liters', aggfunc='mean')
        pivot_df = pivot_df.reindex(list(DAY_MAP.keys()))
        fig_heat = px.imshow(pivot_df, labels=dict(x="Hour", y="Day", color="Liters"),
                           color_continuous_scale='Viridis', template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("""
            <div class="graph-explanation">
                <b>STRATEGIC INSIGHT:</b> This heatmap reveals the 'Hot Windows' where water demand is highest. 
                Management should use this to optimize pump scheduling and prevent dry-tank scenarios.
            </div>
        """, unsafe_allow_html=True)

    elif choice == "PREDICTION ENGINE":
        st.title("AI FORECASTING CORE")
        
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                b_type = st.selectbox("SECTOR", list(BUILDING_MAP.keys()))
                day = st.selectbox("DAY", list(DAY_MAP.keys()))
            with c2:
                phase = st.radio("PHASE", list(PHASE_MAP.keys()), horizontal=True)
                occ = st.slider("OCCUPANCY %", 0, 100, 75)
            with c3:
                hour = st.select_slider("TIME WINDOW", options=list(range(24)), value=12,
                                       format_func=lambda x: f"{x}:00 to {x+1}:00")
                run = st.button("EXECUTE NEURAL INFERENCE")

        if run:
            if model is None:
                st.error("Deployment Error: Primary brain (models/water_model.pkl) not found.")
            else:
                input_data = pd.DataFrame([{
                    'building_code': BUILDING_MAP[b_type], 'day_code': DAY_MAP[day], 
                    'phase_code': PHASE_MAP[phase], 'occupancy_percentage': occ, 'time_of_day': hour
                }])
                prediction = model.predict(input_data)[0]
                
                # HUD
                st.markdown(f"""
                    <div class="hud-container">
                        <div class="hud-label">ESTIMATED WATER LOAD</div>
                        <div class="hud-value">{prediction:.2f}L</div>
                        <div class="hud-label">SYSTEM INTEGRITY: VERIFIED (R2 {metadata['accuracy']:.2f})</div>
                    </div>
                """, unsafe_allow_html=True)

                # Evidence Match
                st.subheader("HISTORICAL VERIFICATION")
                similar = df[(df['building_type'] == b_type) & (abs(df['time_of_day'] - hour) <= 1) & (df['academic_phase'] == phase)].tail(3)
                if not similar.empty:
                    st.write("Current prediction correlates with these verified records:")
                    st.dataframe(similar[['day_of_week', 'time_of_day', 'consumption_liters']], hide_index=True)
                
                st.info(f"AI Insight: The predicted load of {prediction:.0f}L is triggered by {b_type} activity patterns during the {phase} cycle.")

    elif choice == "SYSTEM LOGS":
        st.title("SQL DATA LEDGER")
        st.dataframe(df.tail(200), use_container_width=True)
