import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pickle

# Add the parent directory to Python's import path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import get_data
from main import model, tools

# Parameters
BANDS = ["CMI_C08"]
EXTENT = [-85.7, -78, 30.6, 34.8]
#[-85.7, -81, 30.6, 34.8]
DATES = ["2024-06-21"] 
#+ ["2024-12-25"] + tools.generate_dates(2, month=3) + tools.generate_dates(2, month=6)
HOURS = 3
MINUTES = [15, 60]
FILE_PATH = 'datas/datas_pre.pkl'

# Ensure the output directory exists
os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)

# Create a parameter list
params_list = [
    {
        "date": date,
        "band": band,
        "hours": HOURS,
        "extent": EXTENT,
        "minutes": minute
    }
    for date in DATES
    for band in BANDS
    for minute in MINUTES
]

datas = []
i = 0
for params in params_list:
    data = get_data.goes(**params)
    data.id = i
    i += 1
    datas.append(data)

# Save all data to file
with open(FILE_PATH, 'wb') as f:
    pickle.dump(datas, f)

print("Data saved.")