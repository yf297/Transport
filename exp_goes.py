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
    "2024-09-20",
]

bands = [8, 9, 10]
results = {}  

for date in dates:
    for band in bands:
        # Load data
        dsf = loaders.goes.discrete_scalar_field(date=date, 
                                        band = band, 
                                        start = "00:00", 
                                        end = "04:05", 
                                        by = "00:30", 
                                        extent=(-86.1, -75.1, 30, 37.1))
        t = time.strftime("%H:%M", time.gmtime(int(dsf.coord_field.T[4])))
        dvf = loaders.goes.discrete_vector_field(date = date, 
                                         time = t,
                                         band = band,
                                         extent=(-86.1, -75.1, 30, 37.1))


        # Create nested output folder: date/levelmb
        folder_name = os.path.join("goes", date, f"{band}")
        os.makedirs(folder_name, exist_ok=True)

        # Plot and save scalar field at start and end frames
        for frame in [0,dsf.coord_field.T.shape[0]-1]:
            fig = dsf.plot(frame=frame)
            fig.savefig(os.path.join(folder_name, f"dsf_frame{frame}.png"))
            plt.close(fig)

        # Plot and save discrete vector field at center frame
        fig = dvf.plot(gif = False)
        fig.savefig(os.path.join(folder_name, "dvf_frame2.png"))
        plt.close(fig)

        # Train continuous vector field
        cvf = fields.vector_field.ContinuousVectorField()
        cvf.train(dsf, epochs=100, nn=1, k=4, size=4000)

        # Plot and save continuous vector field at center frame
        fig = cvf.plot(dsf.coord_field, factor=12, frame=4)
        fig.savefig(os.path.join(folder_name, "cvf_frame2.png"))
        plt.close(fig)

        sigma2, l0, l1, l2 = cvf.sigma2, cvf.l0/3600, cvf.l1, cvf.l2

        # compute RMS & RMSE
        n = dvf.coord_field.T.size(0)
        RMS = sum(dvf.RMS(frame=i) for i in range(n)) / n
        RMSE = 0.0

        results[(date, band)] = (sigma2, l0, l1, l2, RMS, RMSE)

