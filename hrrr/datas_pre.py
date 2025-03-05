import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from main import get_data, tools, model


sat = "hrrr"  # or "goes"
levels = ["500 mb", "700 mb"]  
extent = extent = [-80, -65, 38, 48]
dates = tools.generate_dates(5)

file_path = f'datas/datas_pre.pkl'

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        existing_datas = pickle.load(f)
else:
    existing_datas = []

# Extract existing dates to avoid duplicates
existing_dates = {data.date for data in existing_datas}

new_datas = []
for date in dates:
    if date in existing_dates:
        print(f"Skipping date {date}, already exists.")
        continue 
    
    for level in levels:
        T, XY, Z, XY_UV = get_data.hrrr(date=date, level=level, hours=4, extent=extent)
        data_entry = model.data(T, XY, Z, XY_UV)
        data_entry.extent = extent
        data_entry.date = date
        data_entry.level = level
        new_datas.append(data_entry)

if new_datas:
    existing_datas.extend(new_datas)
    with open(file_path, 'wb') as f:
        pickle.dump(existing_datas, f)
    print("New data saved.")
else:
    print("No new data to save.")
