import warnings
warnings.filterwarnings('ignore')

import sys
import torch
import os
import pickle

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import get_data
from main import  model, tools

# Levels and area of interest
levels = ["500 mb", "700 mb"]
extent = [-85.7, -78, 30.6, 34.8]
#[-85.7, -80, 30.6, 34.8]
#[-96, -88.5, 36, 40.61]
#[-85.7, -77.9, 30.6, 34.9]
dates = ["2024-03-15", "2024-09-18", "2024-12-25"] + tools.generate_dates( 2 )
hours = 4
factor = 2

file_path = 'datas/datas_pre.pkl'
# Load existing data if available
if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        existing_datas = pickle.load(f)
else:
    existing_datas = []

# Get the dates we already have
existing_dates = {data.date for data in existing_datas}
new_datas = []

# For each date, skip if it's already processed, else download and create data entries
for date in dates:
    if date in existing_dates:
        print(f"Skipping date {date}, already exists.")
        continue
    
    for level in levels:
        # Retrieve HRRR data
        data = get_data.hrrr(
            date=date, 
            level=level,
            hours=hours, 
            extent=extent, 
            factor=factor
        )
        new_datas.append(data)

# If we got new data, add to existing and save to file
if new_datas:
    existing_datas.extend(new_datas)
    with open(file_path, 'wb') as f:
        pickle.dump(existing_datas, f)
    print("New data saved.")
else:
    print("No new data to save.")