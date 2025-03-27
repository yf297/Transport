import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pickle

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import get_data
from main import model, tools

# Levels and area of interest
# Levels and area of interest
bands = ["CMI_C08","CMI_C09"]
extent = [-85.7, -78, 30.6, 34.8]
#[-85.7, -77.9, 30.6, 34.9]
#[-73.5, -69.4, 41, 42.6]
#[-83.4, -78, 32.2, 34.8]
dates = ["2024-09-18"]
#tools.generate_dates(1)
hours = 4
factor = 3


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
    
    for band in bands:
        # Retrieve GOES data
        data = get_data.goes(
            date=date,
            hours=hours, 
            band=band,
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