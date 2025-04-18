import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import get_data

# Parameters
BANDS = [8]
DATES =  ["2024-09-18"]
FILE_PATH = 'datas/datas_pre.pkl'

# Ensure the output directory exists
os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)

# Create a parameter list
params_list = [
    {
        "Date": date,
        "Hours": 4,
        "Band": band,
        "Extent":[-88, -76, 30.6, 40.8]
    }
    
    for date in DATES
    for band in BANDS
]

datas = []
for params in params_list:
    data = get_data.goes(**params)
    datas.append(data)

# Save all data to file
with open(FILE_PATH, 'wb') as f:
    pickle.dump(datas, f)

print("Data saved.")