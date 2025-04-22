import sys
import pathlib
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path.cwd()/"src"))
import random, calendar
from datetime import datetime
import torch 
import numpy as np
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import pandas as pd
from loaders.hrrr import discrete_scalar_field, discrete_vector_field
from fields.vector_field import ContinuousVectorField

def run_experiment(date: str, level: int):
    dsf = discrete_scalar_field(date, hours=4, level=f"{level} mb", extent=(-92, -74, 26, 36.3))
    cvf = ContinuousVectorField()
    cvf.train(dsf, epochs=50, sample_size=4000)
    dvf = discrete_vector_field(date, hours=4, level=f"{level} mb", extent=(-92, -74, 26, 36.3))
    return {
        "date": date,
        "level": level,
        "$\sigma^2$": float(cvf.sigma2),
        "$l_0$": float(cvf.l0),
        "$l_1$": float(cvf.l1),
        "$l_2$": float(cvf.l2),
        "RMS": float(dvf.RMS()),
        "RMSE":   float(cvf.RMSE(dvf))
    }

ms  = ["2024-03-28", "2024-12-25"]
dates = [
    (d := datetime.fromisoformat(m)).replace(
        day=random.randint(1, calendar.monthrange(d.year, d.month)[1])
    ).strftime("%Y-%m-%d")
    for m in ms
]
levels = [500, 700]

results = []
for date in dates:
    for lvl in levels:
        results.append(run_experiment(date, lvl))

df = pd.DataFrame(results)
latex = df.to_latex(
    index=False,
    float_format="%.3f",
    caption="Estimated coveriance parameters and RMSE by date and pressure level",
    label="tab:params_rmse"
)
out = pathlib.Path(__file__).parent / "table.tex"
with open(out, "w") as f:
    f.write(latex)
print(f"Wrote LaTeX table to {out}")