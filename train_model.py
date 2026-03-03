import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

"""
HYDRO-CORE MODEL ENGINE (HACKATHON v4.5)
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
    print("--- Initializing HACKATHON v4.5 Training Pipeline ---")
    
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

    # 4. TRAINING (Optimized for Non-Linear Patterns)
    features = ['building_code', 'day_code', 'phase_code', 'occupancy_percentage', 'time_of_day']
    X = df_clean[features]
    y = df_clean['consumption_liters']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Increased estimators and depth to better capture the non-linear "Sinusoidal" curves
    model = RandomForestRegressor(
        n_estimators=300, 
        max_depth=None, 
        min_samples_split=2,
        random_state=42,
        n_jobs=-1 # Use all CPU cores for speed
    )
    model.fit(X_train, y_train)
    
    # 5. VERIFICATION
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    # Recall Optimization: We define "Critical High Usage" as usage above the 70th percentile
    # This makes the Recall metric much more meaningful for campus management.
    threshold = np.percentile(y, 70) 
    y_test_class = (y_test > threshold).astype(int)
    preds_class = (preds > threshold).astype(int)
    
    from sklearn.metrics import f1_score, recall_score
    f1 = f1_score(y_test_class, preds_class, zero_division=0)
    recall = recall_score(y_test_class, preds_class, zero_division=0)
    
    print(f"Results: R2 Accuracy: {r2:.4f} | F1 Score: {f1:.4f} | Recall: {recall:.4f}")
    print(f"Mean Error: {mae:.2f}L per hour")

    # 6. SERIALIZATION (Saving the Model)
    # The .pkl file is basically a "Saved Game" for the AI. 
    # It stores the patterns the model learned so we don't have to re-train it every time.
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump({
        'model': model,
        'metadata': {
            'accuracy': r2,
            'f1_score': f1,
            'recall': recall,
            'threshold': threshold,
            'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': features
        }
    }, model_output)
    
    print(f"Save: System State Saved to: {model_output} (Ready for Dashboard)")



if __name__ == "__main__":
    run_training_pipeline()
