import warnings
import sys
import os
import pathlib
import torch 
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import HTML
import time

warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path.cwd()/"src"))

import loaders.goes
import fields.vector_field

dates = [
    "2024-09-20"
]

bands = [8, 9, 10]
results = {}

for date in dates:
    for band in bands:
        dsf = loaders.goes.discrete_scalar_field(
            date=date,
            band=band,
            start="00:00",
            end="04:00",
            extent=(-86.1, -75.1, 30, 40.1)
        )

        seconds = dsf.coord_field.T[dsf.coord_field.T.size(0)//2]
        t = time.strftime("%H:%M", time.gmtime(int(seconds)))
        dvf = loaders.goes.discrete_vector_field(
            date=date,
            time=t,
            band=band,
            extent=(-86.1, -75.1, 30, 40.1)
        )

        folder_name = os.path.join("goes", date, f"{band}")
        os.makedirs(folder_name, exist_ok=True)

        for frame in [0, dsf.coord_field.T.size(0)-1]:
            fig = dsf.plot(frame=frame)
            fig.savefig(os.path.join(folder_name, f"dsf_hour_{frame}.png"))
            plt.close(fig)

        fig = dvf.plot(gif=False, scale=20)
        fig.savefig(os.path.join(folder_name, "dvf_hour_2.png"))
        plt.close(fig)

        cvf = fields.vector_field.ContinuousVectorField()
        cvf.train(dsf, epochs=100, size=6000)

        fig = cvf.plot(dvf.coord_field, scale=20)
        fig.savefig(os.path.join(folder_name, "cvf_hour_2.png"))
        plt.close(fig)

        sigma2, l0, l1, l2, tau2 = (
            cvf.sigma2,
            cvf.l0/3600,
            cvf.l1,
            cvf.l2,
            cvf.tau2,
        )

        results[(date, band)] = (sigma2, l0, l1, l2, tau2)


# write LaTeX table for bands 8, 9 & 10
out = []
out.append(r"\begin{table}[ht!]")
out.append(r"\centering")
out.append(r"\begin{tabular}{llrrrrr}")
out.append(r"\hline")
out.append(r"Date & Band & $\hat{\sigma}^2$ & $\hat{l}_0$ & $\hat{l}_1$ & $\hat{l}_2$ & $\hat{\tau}^2$ \\")
out.append(r"\hline")

for date in dates:
    # band 8, start multirow of 3
    s2, l0, l1, l2, tau2 = results[(date, 8)]
    out.append(
        r"\multirow{3}{*}{%s} & 8  & %.2f & %.2f & %.2f & %.2f & %.2f \\"
        % (date, s2, l0, l1, l2, tau2)
    )
    # band 9
    s2, l0, l1, l2, tau2 = results[(date, 9)]
    out.append(
        r"                       & 9  & %.2f & %.2f & %.2f & %.2f & %.2f \\"
        % (s2, l0, l1, l2, tau2)
    )
    # band 10
    s2, l0, l1, l2, tau2 = results[(date, 10)]
    out.append(
        r"                       & 10 & %.2f & %.2f & %.2f & %.2f & %.2f \\"
        % (s2, l0, l1, l2, tau2)
    )
    out.append(r"\hline")

out.append(r"\end{tabular}")
out.append(r"\caption{Estimated Covariance Parameters for GOES Bands 8, 9, and 10}")
out.append(r"\label{tab:cov_goes}")
out.append(r"\end{table}")

os.makedirs("goes", exist_ok=True)
with open("goes/table.tex", "w") as f:
    f.write("\n".join(out))

print("Wrote LaTeX table to goes/table.tex")
