# 💧 Hackathon v4.5: Enterprise Water Prediction System

## 📌 Overview

Hydro-Core is an AI-driven decision support system designed to manage campus water networks through high-precision forecasting. By analyzing factors such as Building Type, Human Occupancy, Temporal Cycles, and Academic Phases, the system provides actionable insights to prevent wastage and optimize supply.

## 🚀 Key Features

- **SQL-First Architecture**: Uses an appendable SQLite database for persistent telemetry logging.
- **AI Model v4.0 (Random Forest)**: High-accuracy predictor (90%+ R²) considering 5 real-world factors.
- **Evidence-Based Dashboard**: Interactive Streamlit UI with historical matches and AI reasoning.
- **Automated Anomaly Scrubbing**: Simple threshold-based cleaning to ignore "garbage" sensor readings.

## 📂 Project Structure (Cleaned Repository)

- `dashboard/app.py`: The primary Hydro-Core AI Hub.
- `campus_water_prediction.ipynb`: Research notebook for feature importance and data scrubbing.
- `data_generation.py`: The core Data Synthesis and SQL logic script.
- `campus_water.db`: SQLite database storing all historical records.
- `requirements.txt`: System dependencies.

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Initialize Database**:
   ```bash
   python data_generation.py
   ```
3. **Run Hydro-Hub**:
   ```bash
   streamlit run dashboard/app.py
   ```

## 📊 Prediction Engine

The system uses a **Random Forest Regressor** trained on historical campus cycles. It incorporates:

- **Sector**: Hostel, Academic, or Lab.
- **Calendar**: Day of the week logic.
- **Phase**: Normal, Exam, or Vacation cycles.
- **Human Factor**: Real-time occupancy percentage.
- **Temporal**: 24-hour load distribution.
