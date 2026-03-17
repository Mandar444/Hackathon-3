import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta

"""
DATA GENERATION ENGINE v5.0 (Resilient & Non-Linear)
--------------------------------------------------
Simulates campus water consumption with complex interactions, 
stochastic noise, and drift patterns.
"""

def generate_records(num_records, start_date=None, existing_df=None):
    np.random.seed(None) # Truly random
    
    building_types = ['Hostel', 'Academic', 'Lab']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    phases = ['Normal', 'Exam', 'Vacation']
    
    if start_date is None:
        start_date = datetime(2024, 1, 1, 0, 0)
    
    data = []
    current_time = start_date

    for i in range(num_records):
        b_type = np.random.choice(building_types)
        day = current_time.strftime('%A')
        # Phase logic: 70% Normal, 20% Exam, 10% Vacation
        # We can simulate seasons based on months
        month = current_time.month
        if month in [5, 6, 12]: # Vacation months
            phase = 'Vacation'
        elif month in [4, 11]: # Exam months
            phase = 'Exam'
        else:
            phase = 'Normal'
            
        hour = current_time.hour
        occ = np.random.randint(10, 100)
        
        # 1. BASE CAPACITY
        base_capacity = 4200 if b_type == 'Hostel' else 2200
        
        # 2. NON-LINEAR TIME OF DAY (Multiple harmonic waves)
        # Primary peak at 8 AM, secondary peak at 8 PM
        time_factor = 0.6 * np.exp(-((hour - 8)**2) / 8) + 0.4 * np.exp(-((hour - 20)**2) / 12)
        # Add baseline usage
        time_factor += 0.15 * np.random.random()
        
        # 3. NON-LINEAR OCCUPANCY (Sigmoid relationship: saturation effect)
        # Low occupancy = zero usage, medium = rapid growth, high = saturation
        occ_norm = occ / 100
        occ_factor = 1 / (1 + np.exp(-10 * (occ_norm - 0.5)))
        
        # 4. WEEKEND VS WEEKDAY (Interaction effect)
        day_mult = 1.0
        if day in ['Saturday', 'Sunday']:
            day_mult = 1.3 if b_type == 'Hostel' else 0.15
            
        # 5. PHASE INFLUENCE
        phase_mult = 1.0
        if phase == 'Vacation': 
            phase_mult = 0.1
        elif phase == 'Exam': 
            phase_mult = 1.4
        
        # 6. DRIFT SIMULATION (Long term efficiency or scaling)
        # Simulate slight increase in usage over records (e.g., student population growth)
        drift = 1.0 + (i / 10000) * 0.1 # 10% growth over 10k records
        
        # 7. COMPLEX INTERACTION ENGINE
        usage = base_capacity * occ_factor * time_factor * day_mult * phase_mult * drift
        
        # Add Interaction: If occupancy is high AND it's a peak time, add a 'surge'
        if occ > 85 and (7 <= hour <= 10 or 19 <= hour <= 22):
            usage *= 1.3
            
        # Add Stochastic Noise
        noise = np.random.normal(0, usage * 0.15) # 15% noise
        usage += noise
        
        # Add random spikes (Leaks/Maintenance)
        if np.random.random() < 0.015: # 1.5% chance
            usage += np.random.randint(800, 3000)

        usage = abs(round(usage, 2))
        
        data.append({
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'building_type': b_type,
            'day_of_week': day,
            'academic_phase': phase,
            'occupancy_percentage': occ,
            'time_of_day': hour,
            'consumption_liters': usage
        })
        
        # Increment time by 1 hour
        current_time += timedelta(hours=1)
        
    return pd.DataFrame(data)

def update_database(num_records=2000):
    db_path = 'campus_water.db'
    conn = sqlite3.connect(db_path)
    
    try:
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='water_records'")
        exists = cursor.fetchone()
        
        if exists:
            # Get latest timestamp
            last_df = pd.read_sql("SELECT timestamp FROM water_records ORDER BY timestamp DESC LIMIT 1", conn)
            start_date = datetime.strptime(last_df['timestamp'].iloc[0], '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)
            print(f"--- Appending {num_records} New Randomized Records ---")
            new_df = generate_records(num_records, start_date=start_date)
            new_df.to_sql('water_records', conn, if_exists='append', index=False)
        else:
            print(f"--- Initializing Database with 10,000 Records ---")
            new_df = generate_records(10000)
            new_df.to_sql('water_records', conn, if_exists='replace', index=False)
            
        conn.close()
        print(f"[SUCCESS] Database updated at {db_path}")
    except Exception as e:
        conn.close()
        print(f"[ERROR] Database update failed: {e}")

if __name__ == "__main__":
    # If the user wants to update with 2000, we call with 2000
    update_database(2000)


