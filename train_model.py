import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score

"""
HYDRO-CORE ANALYTICS ENGINE v5.0
--------------------------------
This script handles the full ML pipeline with multi-model evaluation:
1. SQL Data Extraction (with Timestamp parsing)
2. Feature Engineering (Sinuoidal Time encoding)
3. Multi-Model Benchmarking (Random Forest vs Gradient Boosting vs Linear)
4. Advanced Metrics (MAE, RMSE, R2, Explained Variance)
5. Model Serialization
"""

BUILDING_MAP = {'Hostel': 0, 'Academic': 1, 'Lab': 2}
DAY_MAP = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
PHASE_MAP = {'Normal': 0, 'Exam': 1, 'Vacation': 2}

def engineer_features(df):
    """
    ARCHITECTURE NOTE: Traditional ML struggles with raw 'hour' integers.
    We convert hours into Sine/Cosine waves to represent the cyclical nature of time.
    """
    df['building_code'] = df['building_type'].map(BUILDING_MAP)
    df['day_code'] = df['day_of_week'].map(DAY_MAP)
    df['phase_code'] = df['academic_phase'].map(PHASE_MAP)
    
    # Cyclical Encoding for Time of Day
    df['hour_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24)
    
    features = ['building_code', 'day_code', 'phase_code', 'occupancy_percentage', 'hour_sin', 'hour_cos']
    return df, features

def run_training_pipeline(db_path='campus_water.db', model_output='models/water_model.pkl'):
    print("--- Initializing Multi-Model Pipeline v5.0 ---")
    
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    # 1. DATA EXTRACTION
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM water_records", conn)
    conn.close()
    
    # 2. FEATURE ENGINEERING
    df, features = engineer_features(df)
    X = df[features]
    y = df['consumption_liters']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. MULTI-MODEL COMPETITION
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    results = {}
    best_model = None
    best_r2 = -np.inf
    
    print(f"{'Model':<20} | {'R2':<8} | {'MAE':<8} | {'RMSE':<8}")
    print("-" * 50)
    
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        exp_var = explained_variance_score(y_test, preds)
        
        results[name] = {
            'r2': r2, 'mae': mae, 'rmse': rmse, 'exp_var': exp_var,
            'importance': list(m.feature_importances_) if hasattr(m, 'feature_importances_') else None
        }
        
        print(f"{name:<20} | {r2:.4f} | {mae:.2f} | {rmse:.2f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = m
            best_model_name = name

    print("-" * 50)
    print(f"WINNER: {best_model_name} (R2: {best_r2:.4f})")

    # 4. CRITICAL RECALL CALCULATION
    threshold = np.percentile(y, 75)
    y_test_class = (y_test > threshold).astype(int)
    preds_class = (best_model.predict(X_test) > threshold).astype(int)
    
    from sklearn.metrics import recall_score, f1_score
    recall = recall_score(y_test_class, preds_class)
    f1 = f1_score(y_test_class, preds_class)

    # 5. SERIALIZATION
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump({
        'model': best_model,
        'metadata': {
            'name': best_model_name,
            'accuracy': best_r2,
            'f1_score': f1,
            'recall': recall,
            'mae': results[best_model_name]['mae'],
            'rmse': results[best_model_name]['rmse'],
            'threshold': threshold,
            'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': features,
            'model_results': results
        }
    }, model_output)
    
    print(f"[SUCCESS] Best model ({best_model_name}) saved to {model_output}")

if __name__ == "__main__":
    run_training_pipeline()

