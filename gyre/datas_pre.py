import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pickle

# Add the parent directory to Python's import path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import generate_data

# Parameters
temporal_lenghscales = [1.0]
zonal_lenghscales = [0.2]
meridional_lenghscales = [0.2]
FILE_PATH = 'datas/datas_pre.pkl'

# Ensure the output directory exists
os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)

# Create a parameter list
params_list = [
    {
        "l1": l1,
        "l2": l2,
        "l3": l3,

    }
    for l1 in temporal_lenghscales
    for l2 in zonal_lenghscales
    for l3 in meridional_lenghscales
]

datas = []
i = 0
for params in params_list:
    data = generate_data.gyre(**params)
    data.id = i
    i += 1
    datas.append(data)

# Save all data to file
with open(FILE_PATH, 'wb') as f:
    pickle.dump(datas, f)

print("Data saved.")