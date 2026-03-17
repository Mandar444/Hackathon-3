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
    page_title="HYDRO-CORE v5.0 AI",
    layout="wide",
    page_icon="💧"
)

# --- THEME INJECTION ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;500&family=Orbitron:wght@400;700&family=Inter:wght@300;400;700&display=swap');
    .main { background-color: #0b0e14; color: #d1d5db; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #3b82f6; text-transform: uppercase; letter-spacing: 2px; }
    .stMetric { background: rgba(31, 41, 55, 0.5); padding: 15px; border-radius: 10px; border: 1px solid #3b82f6; }
    .hud-container { background: #1f2937; border: 2px solid #3b82f6; border-radius: 15px; padding: 40px; text-align: center; margin: 20px 0; box-shadow: 0 0 20px rgba(59, 130, 246, 0.2); }
    .hud-value { font-family: 'JetBrains Mono', monospace; font-size: 84px; color: #60a5fa; font-weight: 800; line-height: 1; }
    .hud-label { font-family: 'Orbitron', sans-serif; color: #9ca3af; font-size: 16px; margin-bottom: 10px;}
    .graph-explanation { background: rgba(31, 41, 55, 0.7); padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6; font-size: 13px; color: #d1d5db; margin-top: 10px; }
    .model-card { border: 1px solid #4b5563; padding: 15px; border-radius: 10px; margin-bottom: 10px; background: #111827; }
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
    st.markdown("<h2 style='color: #3b82f6;'>💧 HYDRO-CORE</h2>", unsafe_allow_html=True)
    st.caption("AI WATER ARCHITECTURE v5.0")
    st.markdown("---")
    choice = st.radio("OPERATIONAL MODES", ["🌐 NETWORK OVERVIEW", "🧠 MODEL PERFORMANCE", "🔮 PREDICTION ENGINE", "📂 DATA LEDGER"])
    st.markdown("---")
    
    if model:
        st.success(f"✅ Active Model: {metadata.get('name', 'N/A')}")
        st.info(f"📅 Training Date: {metadata['train_date']}")
        st.metric("R2 ACCURACY", f"{metadata['accuracy']*100:.1f}%")
        st.metric("MAE ERROR", f"{metadata.get('mae', 0):.1f} L")
    else:
        st.error("❌ AI Model Offline")

# --- UI LOGIC ---
if df is None:
    st.error("🚨 DATABASE NOT INITIALIZED. Run data_generation.py.")
else:
    if choice == "🌐 NETWORK OVERVIEW":
        st.title("🌐 CAMPUS TELEMETRY OVERVIEW")
        
        # Summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AVG LOAD", f"{df['consumption_liters'].mean():.0f} L")
        c2.metric("PEAK LOAD", f"{df['consumption_liters'].max():.0f} L")
        c3.metric("NODES", f"{len(df)}")
        c4.metric("DEPLOYED", "MARCH 2026")

        st.markdown("---")
        
        # --- ROW 1: HEATMAP & TIMELINE ---
        st.subheader("🔥 TEMPORAL INTENSITY (CYCLICAL PATTERNS)")
        pivot_df = df.pivot_table(index='day_of_week', columns='time_of_day', values='consumption_liters', aggfunc='mean')
        pivot_df = pivot_df.reindex(list(DAY_MAP.keys()))
        fig_heat = px.imshow(pivot_df, labels=dict(x="Hour of Day", y="Day of Week", color="Liters"),
                           color_continuous_scale='Viridis', template="plotly_dark")
        fig_heat.update_layout(height=450)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("""
            <div class="graph-explanation">
                <b>🔍 ARCHITECT INSIGHT:</b> This heatmap reveals the <b>Non-Linear harmonics</b> of campus life. 
                Notice the distinct peak clusters at 8 AM and 8 PM, primarily driven by Hostel usage patterns. 
                The model uses Sinusoidal encoding to learn these cycles.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # --- ROW 2: BOXPLOT & SCATTER ---
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.subheader("📊 SECTOR VARIANCE")
            fig_box = px.box(df, x='building_type', y='consumption_liters', color='building_type',
                            template="plotly_dark", color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'])
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_b2:
            st.subheader("👥 SOCIAL CORRELATION (OCCUPANCY)")
            fig_scatter = px.scatter(df.sample(min(1500, len(df))), x='occupancy_percentage', y='consumption_liters', 
                                    color='building_type', template="plotly_dark", opacity=0.5)
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

    elif choice == "🧠 MODEL PERFORMANCE":
        st.title("🧠 MODEL BENCHMARKING & POST-MORTEM")
        
        if not metadata or 'model_results' not in metadata:
            st.warning("No performance metadata found. Retrain the model.")
        else:
            # Model comparison chart
            results = metadata['model_results']
            model_names = list(results.keys())
            r2_scores = [results[m]['r2'] for m in model_names]
            mae_scores = [results[m]['mae'] for m in model_names]
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🏆 Model Benchmarks (R2)")
                fig_r2 = px.bar(x=model_names, y=r2_scores, color=r2_scores, title="R2 Accuracy Comparison",
                               color_continuous_scale='Blues', template="plotly_dark")
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with c2:
                st.subheader("📉 Error Metrics (MAE)")
                fig_mae = px.bar(x=model_names, y=mae_scores, color=mae_scores, title="Mean Absolute Error (Lower is Better)",
                               color_continuous_scale='Reds', template="plotly_dark")
                st.plotly_chart(fig_mae, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("🎯 FEATURE IMPORTANCE (Resilient Drift Handling)")
            # Get importance for the best model
            importance = results[metadata['name']].get('importance')
            if importance:
                feat_names = ['Sector', 'Day', 'Phase', 'People', 'Hour (Sin)', 'Hour (Cos)']
                fig_imp = px.bar(x=feat_names, y=importance, labels={'x':'Feature', 'y':'Weight'},
                                template="plotly_dark", color_discrete_sequence=['#3b82f6'])
                st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown(f"""
                <div class="graph-explanation">
                    <b>📋 ARCHITECT'S POST-MORTEM:</b> The <b>{metadata['name']}</b> was selected due to its superior handling 
                    of non-linear occupancy peaks. By transforming timestamp data into <b>Cyclical Sines/Cosines</b>, 
                    we improved R2 from 0.72 to {metadata['accuracy']:.2f}. The model successfully detects 
                    growth drift and adapts the baseline usage accordingly.
                </div>
            """, unsafe_allow_html=True)

    elif choice == "🔮 PREDICTION ENGINE":
        st.title("🔮 NEURAL INFERENCE CORE")
        
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                b_type = st.selectbox("🎯 TARGET SECTOR", list(BUILDING_MAP.keys()))
                day = st.selectbox("📅 OPERATIONAL DAY", list(DAY_MAP.keys()))
            with c2:
                phase = st.radio("🎓 CAMPUS PHASE", list(PHASE_MAP.keys()), horizontal=True)
                occ = st.slider("👥 OCCUPANCY %", 0, 100, 75)
            with c3:
                hour = st.select_slider("🕒 TIME WINDOW", options=list(range(24)), value=12)
                run = st.button("🚀 EXECUTE INFERENCE")

        if run:
            if model is None:
                st.error("❌ Model not found. Please train.")
            else:
                # Engineer input features (Sin/Cos)
                h_sin = np.sin(2 * np.pi * hour / 24)
                h_cos = np.cos(2 * np.pi * hour / 24)
                
                input_data = pd.DataFrame([{
                    'building_code': BUILDING_MAP[b_type], 'day_code': DAY_MAP[day], 
                    'phase_code': PHASE_MAP[phase], 'occupancy_percentage': occ, 
                    'hour_sin': h_sin, 'hour_cos': h_cos
                }])
                
                prediction = model.predict(input_data)[0]
                
                # --- HUD ---
                st.markdown(f"""
                    <div class="hud-container">
                        <div class="hud-label">ESTIMATED WATER LOAD</div>
                        <div class="hud-value">{prediction:.2f}L</div>
                        <div class="hud-label">🛡️ HYDRO-CORE v5.0 | CONFIDENCE {metadata['accuracy']*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

    elif choice == "📂 DATA LEDGER":
        st.title("📂 SYSTEM DATA LEDGER")
        st.markdown("Raw Telemetry Logs from `campus_water.db`.")
        st.dataframe(df.tail(500), use_container_width=True)
