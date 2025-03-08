import warnings
warnings.filterwarnings('ignore')
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
from tabulate import tabulate
from main import tools
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

pre_file_path = f'datas/datas_pre.pkl'
fit_file_path = f'datas/datas_fit.pkl'

with open(fit_file_path, 'rb') as f:
    datas_fit = pickle.load(f)

filtered_fit = [
    data for data in datas_fit if tools.rmse(data, scale=1/86400).item() <= 6
]


date_levels = defaultdict(set)
for data in filtered_fit:
    date_levels[data.date].add(data.level)
valid_dates = {date for date, levels in date_levels.items() if {"500 mb", "700 mb"}.issubset(levels)}
filtered_fit = [data for data in filtered_fit if data.date in valid_dates]


valid_dates = {data.date for data in filtered_fit}

with open(pre_file_path, 'rb') as f:
    datas_pre = pickle.load(f)

filtered_pre = [data for data in datas_pre if data.date in valid_dates]

with open(fit_file_path, 'wb') as f:
    pickle.dump(filtered_fit, f)

with open(pre_file_path, 'wb') as f:
    pickle.dump(filtered_pre, f)


frames = [0]

grouped_data = defaultdict(list)
for data in filtered_fit:
    row = [
        data.level,
        round(data.gp.kernel.outputscale.item() * data.output_std.item(), 2),
        round(data.gp.kernel.base_kernel.lengthscale[0][0].item(), 2),
        round(data.gp.kernel.base_kernel.lengthscale[0][1].item() * data.input_std.item(), 2),
        round(data.gp.kernel.base_kernel.lengthscale[0][2].item() * data.input_std.item(), 2),
        round(data.gp.likelihood.noise.item() * data.output_std.item(), 2),
        round(tools.rmse(data, mag=0).item(), 2),
        round(tools.rmse(data, scale=1/86400).item(), 2),
        round(tools.rmse_U(data, mag = 0).item(), 2),
        round(tools.rmse_U(data, scale=1/86400).item(), 2),
        round(tools.rmse_V(data, mag = 0).item(), 2),
        round(tools.rmse_V(data, scale=1/86400).item(), 2)
    ]
    grouped_data[data.date].append(row)

    date_dir = os.path.join(f"results/plots", str(data.date))
    level_dir = os.path.join(date_dir, str(data.level))
    os.makedirs(level_dir, exist_ok=True)

    for frame_num in frames:
        indices = torch.randperm(datas_pre[0].m)
        fig = data.plot_observations(indices, frame=frame_num)
        plot_path = os.path.join(level_dir, f"scalar_plot_frame_{frame_num}.png")
        fig.savefig(plot_path)
        plt.close(fig)

        indices = data.indices[:600]
        fig = data.plot_vel(indices, frame=frame_num, color="red")
        plot_path = os.path.join(level_dir, f"estimated_plot_frame_{frame_num}.png")
        fig.savefig(plot_path)
        plt.close(fig)

        indices = data.indices[:600]
        fig = data.plot_vel_data(indices, frame=frame_num, color="blue")
        plot_path = os.path.join(level_dir, f"true_plot_frame_{frame_num}.png")
        fig.savefig(plot_path)
        plt.close(fig)

# Generate LaTeX table with grouped dates
latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{llrrrrrrr}\n\\hline\n"
latex_table += "{Date} & {Level} & $\\sigma^2$ & $l_1$ & $l_2$ & $l_3$ & $\\tau^2$ & {RMS } & {RMSE } \\\\\n\\hline\n"

for date, rows in sorted(grouped_data.items()):
    latex_table += f"\\multirow{{{len(rows)}}}{{*}}{{{date}}} "
    for i, row in enumerate(rows):
        if i == 0:
            latex_table += f"& {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} & {row[7]} \\\\\n"
        else:
            latex_table += f" & {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} & {row[7]}  \\\\\n"
    latex_table += "\\hline\n"

latex_table += "\\end{tabular}\n\\caption{}\n\\label{tab:cov_est}\n\\end{table}"

# Save LaTeX table to a file
with open(f"results/table.txt", "w") as f:
    f.write(latex_table)
    

zonal_lengthscales_500 = []
rmse_u_rms_ratios_500 = []
zonal_lengthscales_700 = []
rmse_u_rms_ratios_700 = []

for date, rows in grouped_data.items():
    for row in rows:
        level = row[0]
        zonal_lengthscale = row[3]
        rms_u = row[8]
        rmse_u = row[9]
        if rms_u > 0:
            ratio_u = rmse_u / rms_u
            scaled_lengthscale = zonal_lengthscale / rms_u 
            if level == "500 mb":
                zonal_lengthscales_500.append(scaled_lengthscale)
                rmse_u_rms_ratios_500.append(ratio_u)
            elif level == "700 mb":
                zonal_lengthscales_700.append(scaled_lengthscale)
                rmse_u_rms_ratios_700.append(ratio_u)

plt.figure()
plt.scatter(zonal_lengthscales_500, rmse_u_rms_ratios_500, color='blue', label='500 mb')
plt.scatter(zonal_lengthscales_700, rmse_u_rms_ratios_700, color='red', label='700 mb')
plt.xlabel("Zonal Lengthscale / RMS_U")
plt.ylabel("RMSE_U / RMS_U")
plt.title("RMSE_U/RMS_U vs (Zonal Lengthscale / RMS_U)")
plt.grid(True)
plt.legend()
plt.savefig("results/rmse_vs_zonal.png")
plt.close()


meridional_lengthscales_500 = []
rmse_v_rms_ratios_500 = []
meridional_lengthscales_700 = []
rmse_v_rms_ratios_700 = []

for date, rows in grouped_data.items():
    for row in rows:
        level = row[0]
        meridional_lengthscale = row[4]
        rms_v = row[10]
        rmse_v = row[11]
        if rms_v > 0:
            ratio_v = rmse_v / rms_v
            scaled_lengthscale = meridional_lengthscale / rms_v 
            if level == "500 mb":
                meridional_lengthscales_500.append(scaled_lengthscale)
                rmse_v_rms_ratios_500.append(ratio_v)
            elif level == "700 mb":
                meridional_lengthscales_700.append(scaled_lengthscale)
                rmse_v_rms_ratios_700.append(ratio_v)

plt.figure()
plt.scatter(meridional_lengthscales_500, rmse_v_rms_ratios_500, color='blue', label='500 mb')
plt.scatter(meridional_lengthscales_700, rmse_v_rms_ratios_700, color='red', label='700 mb')
plt.xlabel("Meridional Lengthscale / RMS_V")
plt.ylabel("RMSE_V / RMS_V")
plt.title("RMSE_V/RMS_V vs (Meridional Lengthscale / RMS_V)")
plt.grid(True)
plt.legend()
plt.savefig("results/rmse_vs_meridional.png")
plt.close()