import warnings
warnings.filterwarnings('ignore')

from main import get_data, tools, model
import pickle

sat = "hrrr" # or "goes"
levels = ["500 mb", "700 mb"] # or bands
extent = [-81,-70,41.5,45]
dates = tools.generate_dates(6) + [
    "2024-09-05",
    "2024-09-13"]

datas = []
k = 0
for i in range(len(dates)):
    for j in range(len(levels)):
        T, XY, Z, XY_UV = get_data.hrrr(date=dates[i], 
                                        level = levels[j],
                                        hours = 4,
                                        extent = extent)
        datas.append(model.data(T,XY,Z,XY_UV))
        datas[k].extent = extent
        datas[k].date = dates[i]
        datas[k].level = levels[j]
        k +=1
    
with open(f'datas/{sat}/datas_pre.pkl', 'wb') as f:
    pickle.dump(datas, f)