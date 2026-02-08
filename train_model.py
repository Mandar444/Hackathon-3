import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

"""
HYDRO-CORE MODEL ENGINE (v4.5)
------------------------------
This script handles the full ML pipeline:
1. SQL Data Extraction
2. Adaptive Anomaly Scrubbing
3. Random Forest Training
4. Model Serialization (Joblib)
"""

# Mappings used across the system
BUILDING_MAP = {'Hostel': 0, 'Academic': 1, 'Lab': 2}
DAY_MAP = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
PHASE_MAP = {'Normal': 0, 'Exam': 1, 'Vacation': 2}

def run_training_pipeline(db_path='campus_water.db', model_output='models/water_model.pkl'):
    print("--- Initializing Production Training Pipeline ---")
    
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found. Run data_generation.py first.")
        return

    # 1. SQL EXTRACTION
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM water_records", conn)
    conn.close()
    print(f"Done: Extracted {len(df)} records from SQL.")

    # 2. SIMPLE COLUMN CLEANING/MAPPING
    df['building_code'] = df['building_type'].map(BUILDING_MAP)
    df['day_code'] = df['day_of_week'].map(DAY_MAP)
    df['phase_code'] = df['academic_phase'].map(PHASE_MAP)

    # 3. SCRUBBING (Hard-Limit Filter)
    # User requested outlier cap at 5000L
    df_clean = df[df['consumption_liters'] <= 5000].copy()
    print(f"Cleanup: Scrubbed {len(df) - len(df_clean)} outliers (>5000L).")

    # 4. TRAINING
    features = ['building_code', 'day_code', 'phase_code', 'occupancy_percentage', 'time_of_day']
    X = df_clean[features]
    y = df_clean['consumption_liters']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. VERIFICATION
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"Results: Accuracy (R2): {score:.4f} | Error: {mae:.2f}L")

    # 6. SERIALIZATION (Saving the Brain)
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump({
        'model': model,
        'metadata': {
            'accuracy': score,
            'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': features
        }
    }, model_output)
    
    print(f"Save: Production Model Serialized to: {model_output}")

if __name__ == "__main__":
    run_training_pipeline()
