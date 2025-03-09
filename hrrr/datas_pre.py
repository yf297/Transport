import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from main import get_data, model, tools

levels = ["500 mb", "700 mb"]
extent =[-96, -75, 26, 37]
dates = [ "2024-06-23", "2024-07-25", "2024-08-06", "2024-08-18","2024-09-13", "2024-09-30", "2024-10-11"] + tools.generate_dates(3)
file_path = f'datas/datas_pre.pkl'

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        existing_datas = pickle.load(f)
else:
    existing_datas = []

existing_dates = {data.date for data in existing_datas}

new_datas = []
for date in dates:
    if date in existing_dates:
        print(f"Skipping date {date}, already exists.")
        continue 
    
    for level in levels:
        T, XY, Z, XY_UV = get_data.hrrr(date=date, level=level, hours=6, extent=extent, factor = 9)
        data_entry = model.data(T, XY, Z, XY_UV)
        data_entry.extent = extent
        data_entry.date = date
        data_entry.level = level
        new_datas.append(data_entry)

print(new_datas[0].m)
if new_datas:
    existing_datas.extend(new_datas)
    with open(file_path, 'wb') as f:
        pickle.dump(existing_datas, f)
    print("New data saved.")
else:
    print("No new data to save.")
