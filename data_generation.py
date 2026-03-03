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

def setup_sql_database(num_records=10000):
    np.random.seed(42)
    data = []
    
    building_types = ['Hostel', 'Academic', 'Lab']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    phases = ['Normal', 'Exam', 'Vacation']
    
    print(f"--- Generating {num_records} Complex Non-Linear Records ---")

    for _ in range(num_records):
        b_type = np.random.choice(building_types)
        day = np.random.choice(days)
        phase = np.random.choice(phases, p=[0.7, 0.2, 0.1])
        hour = np.random.randint(0, 24)
        occ = np.random.randint(10, 100)
        
        # 1. BASE CAPACITY
        base_capacity = 4000 if b_type == 'Hostel' else 2100
        
        # 2. NON-LINEAR TIME OF DAY (Sinusoidal waves instead of step functions)
        # Main peak at 8 AM and 8 PM
        time_factor = (np.sin((hour - 8) * (2 * np.pi / 24)) + 1) / 2
        # Secondary peak/variance
        time_factor += 0.5 * (np.cos((hour - 20) * (2 * np.pi / 24)) + 1) / 2
        
        # 3. NON-LINEAR OCCUPANCY (Polynomial relationship)
        # Usage isn't just occ/100; maybe it's quadratic (small crowds use less, large crowds use exponentially more)
        occ_factor = (occ / 100) ** 1.5 
        
        # 4. WEEKEND VS WEEKDAY (Interaction effect)
        day_mult = 1.0
        if day in ['Saturday', 'Sunday']:
            day_mult = 1.2 if b_type == 'Hostel' else 0.2
            
        # 5. PHASE INFLUENCE
        phase_mult = 1.0
        if phase == 'Vacation': 
            phase_mult = 0.2
        elif phase == 'Exam': 
            phase_mult = 1.3
        
        # 6. COMPLEX INTERACTION ENGINE
        # Usage is no longer a simple product of independent sliders
        usage = base_capacity * occ_factor * time_factor * day_mult * phase_mult
        
        # Add Interaction: If occupancy is high AND it's a peak time, add a 'surge'
        if occ > 80 and (7 <= hour <= 10 or 19 <= hour <= 22):
            usage *= 1.25
            
        # Add 25% Heavy Noise (Real-time simulation variance)
        noise = np.random.normal(0, usage * 0.25)
        usage += noise
        
        # Add random spikes (Leaks/Maintenance)
        if np.random.random() < 0.02: # 2% chance of a leak
            usage += np.random.randint(500, 2000)

        # Final Cleaning
        usage = abs(round(usage, 2))
        
        data.append({
            'building_type': b_type,
            'day_of_week': day,
            'academic_phase': phase,
            'occupancy_percentage': occ,
            'time_of_day': hour,
            'consumption_liters': usage
        })
    
    df_gen = pd.DataFrame(data)
    
    # CONNECT TO SQL (Fresh database)
    conn = sqlite3.connect('campus_water.db') 
    df_gen.to_sql('water_records', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"[SUCCESS] Database created with {len(df_gen)} records.")
    print(f"[INFO] Location: campus_water.db")

if __name__ == "__main__":
    setup_sql_database()

