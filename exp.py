import sys
import pathlib
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path.cwd()/"src"))
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
    dsf = discrete_scalar_field(date, hours=4, level=f"{level} mb", extent=(-92, -74, 26, 40))
    cvf = ContinuousVectorField()
    cvf.train(dsf, epochs=100, sample_size=2000)
    dvf = discrete_vector_field(date, hours=4, level=f"{level} mb", extent=(-92, -74, 26, 40))
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

dates  = ["2024-03-22", "2024-06-21", "2024-09-18", "2024-12-25"]
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