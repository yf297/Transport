import warnings
warnings.filterwarnings('ignore')

import pickle
from main import tools
import os
import matplotlib.pyplot as plt
import torch
from tabulate import tabulate 

sat = "hrrr" # or "goes"
with open(f'datas/{sat}/datas_fit.pkl', 'rb') as f:
    datas = pickle.load(f)

frames = [0,2,4]
table = []
headers = [
    "Date",
    "Level",
    "sigma (k)",
    "l_1 (h)",
    "l_2 (m)",
    "l_3 (m)",
    "tau (k)",
    "RMS (m/s)"
    "RMSE (m/s)"
]

for i in range(len(datas)):
    row = [
        datas[i].date,
        datas[i].level,
        round(datas[i].gp.kernel.outputscale.item() * datas[i].output_std.item(),  4),
        round(datas[i].gp.kernel.base_kernel.lengthscale[0][0].item(), 4),
        round(datas[i].gp.kernel.base_kernel.lengthscale[0][1].item() * datas[i].input_std.item(), 4),
        round(datas[i].gp.kernel.base_kernel.lengthscale[0][2].item() * datas[i].input_std.item(), 4),
        round(datas[i].gp.likelihood.noise.item() * datas[i].output_std.item(), 4),
        round(tools.rmse(datas[i], mag = 0).item(), 4),
        round(tools.rmse(datas[i], scale = 1/86400).item(), 4)
    ]
    table.append(row)
        
    date_dir = os.path.join(f"results/{sat}/plots", str(datas[i].date))
    level_dir = os.path.join(date_dir, str(datas[i].level))
    os.makedirs(level_dir, exist_ok=True)
    
    for frame_num in frames:
        indices = torch.linspace(0,datas[i].m -1, datas[i].m).to(torch.int)
        fig = datas[i].plot_observations(indices, frame=frame_num)
        plot_path = os.path.join(level_dir, f"scalar_plot_frame_{frame_num}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        
        indices = datas[i].indices
        fig = datas[i].plot_vel(indices, frame=frame_num)
        plot_path = os.path.join(level_dir, f"estimated_plot_frame_{frame_num}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        
        indices = datas[i].indices
        fig = datas[i].plot_vel(indices, frame=frame_num)
        plot_path = os.path.join(level_dir, f"true_plot_frame_{frame_num}.png")
        fig.savefig(plot_path)
        plt.close(fig)


latex_table  = tabulate(table, headers, tablefmt="latex")
with open(f"results/{sat}/table.txt", "w") as f:
    f.write(latex_table)

