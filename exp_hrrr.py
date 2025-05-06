import warnings
import sys
import os
import pathlib
import torch 
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import HTML

warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path.cwd()/"src"))


import loaders.hrrr
import fields.vector_field

dates = [
    #"2024-09-18",
    "2024-09-19",
    "2024-09-20",
    "2024-09-21",
    #"2024-09-22"
]

levels = [500]
results = {}  

for date in dates:
    for level in levels:
        # Load data
        dsf = loaders.hrrr.discrete_scalar_field(date=date, level=level, hours=4, extent=(-85.5, -75.1, 30.5, 36.5))
        dvf = loaders.hrrr.discrete_vector_field(date=date, level=level, hours=4, extent=(-85.5, -75.1, 30.5, 36.5))

        # Create nested output folder: date/levelmb
        folder_name = os.path.join("hrrr", date, f"{level}mb")
        os.makedirs(folder_name, exist_ok=True)

        # Plot and save scalar field at start and end frames
        for frame in [0,dsf.coord_field.times.shape[0]-1]:
            fig = dsf.plot(frame=frame)
            fig.savefig(os.path.join(folder_name, f"dsf_frame{frame}.png"))
            plt.close(fig)

        # Plot and save discrete vector field at center frame
        fig = dvf.plot(factor = 12, frame=2)
        fig.savefig(os.path.join(folder_name, "dvf_frame2.png"))
        plt.close(fig)

        # Train continuous vector field
        cvf = fields.vector_field.ContinuousVectorField()
        cvf.train(dsf, epochs=100,factor = 7)

        # Plot and save continuous vector field at center frame
        fig = cvf.plot(dsf.coord_field, factor=12, frame=2)
        fig.savefig(os.path.join(folder_name, "cvf_frame2.png"))
        plt.close(fig)

        sigma2, l0, l1, l2 = cvf.sigma2, cvf.l0/3600, cvf.l1, cvf.l2

        # compute RMS & RMSE
        n = dvf.coord_field.times.size(0)
        RMS = sum(dvf.RMS(frame=i) for i in range(n)) / n
        RMSE = sum(cvf.RMSE(dvf, frame=i) for i in range(n)) / n

        results[(date, level)] = (sigma2, l0, l1, l2, RMS, RMSE)

# write LaTeX table
out = []
out.append(r"\begin{table}[ht!]")
out.append(r"\centering")
out.append(r"\begin{tabular}{llrrrrrr}")
out.append(r"\hline")
out.append(r"{Date} & {Level} & $\hat{\sigma}^2$ & $\hat{l}_0$ (h) & $\hat{l}_1$ (m) & $\hat{l}_2$ (m) & RMS (m/s) & RMSE (m/s) \\")
out.append(r"\hline")

for date in dates:
    # first (500 mb) row
    s2, l0, l1, l2, rms, rmse = results[(date, 500)]
    out.append(r"\multirow{2}{*}{%s} & 500 mb & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\" %
               (date, s2, l0, l1, l2, rms, rmse))
    # second (700 mb) row (no date)
    s2, l0, l1, l2, rms, rmse = results[(date, 700)]
    out.append(r" & 700 mb & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\" %
               (s2, l0, l1, l2, rms, rmse))
    out.append(r"\hline")

out.append(r"\end{tabular}")
out.append(r"\caption{Estimated Covariance Parameters with RMS and RMSE}")
out.append(r"\label{tab:cov_est}")
out.append(r"\end{table}")

os.makedirs("tables", exist_ok=True)
with open("tables/cov_est_table.tex", "w") as f:
    f.write("\n".join(out))

print("Wrote LaTeX table to tables/cov_est_table.tex")
