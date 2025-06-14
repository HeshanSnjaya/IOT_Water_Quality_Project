import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate more realistic sample data
np.random.seed(42)
start_date = datetime(2024, 11, 1)
dates = [start_date + timedelta(days=i) for i in range(30)]

data = []
for i, date in enumerate(dates):
    # Simulate seasonal and daily variations
    temp_base = 29 + 2 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 0.5)
    ph_base = 8.5 + 0.3 * np.sin(2 * np.pi * i / 15) + np.random.normal(0, 0.1)
    tds_base = 7000 + 1000 * np.sin(2 * np.pi * i / 20) + np.random.normal(0, 200)
    
    row = {
        'Date': date,
        'Pond No': np.random.randint(0, 24),
        'pH': np.clip(ph_base, 7.0, 9.0),
        'TDS': np.clip(tds_base, 4000, 12000),
        'Temperature': np.clip(temp_base, 25, 35),
        'Turbidity': np.clip(15 + 10 * np.random.random(), 5, 50),
        'DO': np.clip(5.5 + np.random.normal(0, 0.8), 3, 8),
        'Salinity': np.clip(14 + 3 * np.random.normal(), 8, 22),
        'month': date.month
    }
    data.append(row)

# Save to CSV
realistic_df = pd.DataFrame(data)
realistic_df.to_csv('sample_prawn_farm_data.csv', index=False)