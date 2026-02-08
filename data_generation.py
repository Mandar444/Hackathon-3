import pandas as pd
import numpy as np
import sqlite3
import os

"""
DATA GENERATION ENGINE (For Faculty Review)
------------------------------------------
This script simulates 2,500 hourly water consumption records for a university campus.
The logic is based on 5 primary factors to ensure the 'Deep Learning' model has 
realistic patterns to learn from.

MATHEMATICAL LOGIC:
1. Base Capacity: Hostels have a higher baseline than Academic blocks.
2. Time Factor: Morning (6-9 AM) and Evening (6-9 PM) spikes simulate showers/cooking.
3. Day Factor: Weekends reduce Academic load but maintain/increase Hostel load.
4. Phase Factor: 'Exam' season slightly increases load; 'Vacation' drops it by 60%.
5. Occupancy: A linear multiplier representing the percentage of people in the building.
"""

def setup_sql_database(num_records=2500):
    np.random.seed(42)
    data = []
    
    building_types = ['Hostel', 'Academic', 'Lab']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    phases = ['Normal', 'Exam', 'Vacation']
    
    print(f"--- Generating {num_records} Realistic Metadata Records ---")

    for _ in range(num_records):
        b_type = np.random.choice(building_types)
        day = np.random.choice(days)
        phase = np.random.choice(phases, p=[0.7, 0.2, 0.1])
        hour = np.random.randint(0, 24)
        occ = np.random.randint(10, 100)
        
        # 1. BASE CAPACITY (Max liters per hour at 100% occupancy)
        base_capacity = 4000 if b_type == 'Hostel' else 2100
        
        # 2. TIME OF DAY INFLUENCE (The 'Spike' Logic)
        time_mult = 1.0
        if 6 <= hour <= 9: 
            time_mult = 1.6  # Morning Peak (Showers/Breakfast)
        elif 18 <= hour <= 21: 
            time_mult = 1.4  # Evening Peak
        elif 0 <= hour <= 5: 
            time_mult = 0.2  # Night Minimum (Baseline leaks only)
        
        # 3. WEEKEND VS WEEKDAY LOGIC
        day_mult = 1.0
        if day in ['Saturday', 'Sunday']:
            # Hostels stay busy, Academic blocks go quiet
            day_mult = 1.1 if b_type == 'Hostel' else 0.3
            
        # 4. ACADEMIC CYCLE LOGIC
        phase_mult = 1.0
        if phase == 'Vacation': 
            phase_mult = 0.35 # 65% drop in usage
        elif phase == 'Exam': 
            phase_mult = 1.15 # 15% increase in focus areas
        
        # 5. FINAL CALCULATION ENGINE
        usage = base_capacity * (occ/100) * time_mult * day_mult * phase_mult
        
        # Add 10% Gaussian Noise (Simulates real-world sensor fluctuations)
        usage += np.random.normal(0, usage * 0.1)
        
        # Safety Hard-Cap for Realistic Limits
        usage = min(usage, 7000) if b_type == 'Hostel' else min(usage, 4000)
        
        data.append({
            'building_type': b_type,
            'day_of_week': day,
            'academic_phase': phase,
            'occupancy_percentage': occ,
            'time_of_day': hour,
            'consumption_liters': abs(round(usage, 2))
        })
    
    df_gen = pd.DataFrame(data)
    
    # CONNECT TO SQL (Ensures data is appendable and persistent)
    conn = sqlite3.connect('campus_water.db') 
    df_gen.to_sql('water_records', conn, if_exists='append', index=False)
    conn.close()
    
    print(f"[SUCCESS] Database created with {len(df_gen)} records.")
    print(f"[INFO] Location: campus_water.db")

if __name__ == "__main__":
    setup_sql_database()
