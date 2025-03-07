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


frames = [0,3,6]

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

    # Generate and save plots
    '''for frame_num in frames:
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
        plt.close(fig)'''

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
    

temporal_lengthscales = []
rmse_rrmse_ratios = []

for date, rows in grouped_data.items():
    for row in rows:
        temporal_lengthscale = row[2]  # Assuming `l_1` corresponds to the temporal lengthscale
        rmse = row[7] 
        rms = row[6]  

        if rms > 0:  # Avoid division by zero
            ratio = rmse / rms
            temporal_lengthscales.append(temporal_lengthscale)
            rmse_rrmse_ratios.append(ratio)

# Convert lists to numpy arrays
temporal_lengthscales = np.array(temporal_lengthscales)
rmse_rrmse_ratios = np.array(rmse_rrmse_ratios)

# Sort values for a smooth plot
sorted_indices = np.argsort(temporal_lengthscales)
temporal_lengthscales = temporal_lengthscales[sorted_indices]
rmse_rrmse_ratios = rmse_rrmse_ratios[sorted_indices]

# Plot RMSE/RRMSE ratio vs Temporal Lengthscale
plt.figure(figsize=(8, 5))
plt.plot(temporal_lengthscales, rmse_rrmse_ratios, marker='o', linestyle='-', color='b', label="RMSE/RMS")
plt.xlabel("Temporal Lengthscale")
plt.ylabel("RMSE / RMS")
plt.title("RMSE/RMS vs Temporal Lengthscale")
plt.grid(True)
plt.legend()
plt.savefig("results/rmse_vs_temporal.png")


temporal_lengthscales = []
rmse_rrmse_ratios = []

for date, rows in grouped_data.items():
    for row in rows:
        temporal_lengthscale = row[3] 
        rmse = row[9] 
        rms = row[8]  

        if rms > 0:  # Avoid division by zero
            ratio = rmse / rms
            temporal_lengthscales.append(temporal_lengthscale)
            rmse_rrmse_ratios.append(ratio)

# Convert lists to numpy arrays
temporal_lengthscales = np.array(temporal_lengthscales)
rmse_rrmse_ratios = np.array(rmse_rrmse_ratios)

# Sort values for a smooth plot
sorted_indices = np.argsort(temporal_lengthscales)
temporal_lengthscales = temporal_lengthscales[sorted_indices]
rmse_rrmse_ratios = rmse_rrmse_ratios[sorted_indices]

# Plot RMSE/RRMSE ratio vs Temporal Lengthscale
plt.figure(figsize=(8, 5))
plt.plot(temporal_lengthscales, rmse_rrmse_ratios, marker='o', linestyle='-', color='b', label="RMSE_U/RMS_U")
plt.xlabel("Zonal Lengthscale")
plt.ylabel("RMSE_U/RMS_U")
plt.title("RMSE_U/RMS_U vs Zonal Lengthscale")
plt.grid(True)
plt.legend()
plt.savefig("results/rmse_vs_zonal.png")


temporal_lengthscales = []
rmse_rrmse_ratios = []

for date, rows in grouped_data.items():
    for row in rows:
        temporal_lengthscale = row[4]  # Assuming `l_1` corresponds to the temporal lengthscale
        rmse = row[11] 
        rms = row[10]  

        if rms > 0:  # Avoid division by zero
            ratio = rmse / rms
            temporal_lengthscales.append(temporal_lengthscale)
            rmse_rrmse_ratios.append(ratio)

# Convert lists to numpy arrays
temporal_lengthscales = np.array(temporal_lengthscales)
rmse_rrmse_ratios = np.array(rmse_rrmse_ratios)

# Sort values for a smooth plot
sorted_indices = np.argsort(temporal_lengthscales)
temporal_lengthscales = temporal_lengthscales[sorted_indices]
rmse_rrmse_ratios = rmse_rrmse_ratios[sorted_indices]

# Plot RMSE/RRMSE ratio vs Temporal Lengthscale
plt.figure(figsize=(8, 5))
plt.plot(temporal_lengthscales, rmse_rrmse_ratios, marker='o', linestyle='-', color='b', label="RMSE_V/RMS_V")
plt.xlabel("Meridional Lengthscale")
plt.ylabel("RMSE_V/RMS_V")
plt.title("RMSE_V/RMS_V vs Meridional Lengthscale")
plt.grid(True)
plt.legend()
plt.savefig("results/rmse_vs_meridional.png")