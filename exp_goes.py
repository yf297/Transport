import sys
import pathlib
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path.cwd()/"src"))
import random
import torch 
import numpy as np
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import pandas as pd
import loaders.goes
import fields.vector_field 


cvf = []
dvf = []

def run_experiment(start = "00:00", end = "04:06", by = "00:05"):
    dsf0 = loaders.goes.discrete_scalar_field("2024-07-13", 
                                             band = 8, 
                                             start = start, 
                                             end = end, 
                                             by = by, 
                                             extent=(-88, -75.1, 30, 36.3))
    mapping = {"00:05": 6, "00:15": 2, "00:30": 1}
    nn = mapping[by]
    print(dsf0.coord_field.times[dsf0.coord_field.times.size(0)//2]/3600)
    cvf0 = fields.vector_field.ContinuousVectorField()
    cvf0.train(dsf0, epochs=50, nn = nn, k = 4, size = 2000)
    cvf.append(cvf0)
    vector = cvf0.func(dsf0.coord_field.times, dsf0.coord_field.locations) 
    dvf0 = fields.vector_field.DiscreteVectorField(dsf0.coord_field, vector)  
    dvf.append(dvf0)
    del cvf0

       
experiments = [
    ("00:00", "04:06", "00:05"),
    ("00:00", "04:06", "00:15"),
    ("00:00", "04:06", "00:30"),
]

for start, end, by in experiments:
    run_experiment(start=start, end=end, by=by)
    
    
experiments = [
    ("00:00", "04:00", "00:05"),
    ("00:00", "04:00", "00:15"),
    ("00:00", "04:00", "00:30"),
]

def describe_label(start: str, end: str, by: str) -> str:
    def to_minutes(t: str) -> int:
        h, m = map(int, t.split(':'))
        return h*60 + m
    half_window = (to_minutes(end) - to_minutes(start)) // 2
    hrs, mins = divmod(half_window, 60)
    offset = ''.join(f"{v}{unit}" for v, unit in ((hrs, 'h'), (mins, 'min')) if v)
    by_min = to_minutes(by)
    interval = f"{by_min}min" if by_min < 60 else f"{by_min//60}h"
    return f"{offset} each side, {interval} interval"
labels = [describe_label(s, e, b) for s, e, b in experiments]


rmse_matrix = [
    [cvf_i.RMSE(dvf_j,dvf_j.coord_field.times.size(0)//2) for dvf_j in dvf] 
    for cvf_i in cvf
]

df = pd.DataFrame(rmse_matrix, index=labels, columns=labels)

latex_code = df.to_latex(
    index=True,
    label="tab:rmse",
    float_format="%.3f"
)

# write it out
with open("rmse_table.tex", "w") as f:
    f.write(latex_code)