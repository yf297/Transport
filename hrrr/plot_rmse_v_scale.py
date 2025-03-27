import warnings
warnings.filterwarnings('ignore')

import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import tools

fit_file_path = 'datas/datas_fit.pkl'
with open(fit_file_path, 'rb') as f:
    datas_fit = pickle.load(f)

grouped_data = defaultdict(list)

# Collect all rows from datas_fit
for data in datas_fit:
    row = [
        data.level,
        data.sigma2,
        data.l1,  # temporal lengthscale
        data.l2,  # zonal lengthscale
        data.l3,  # meridional lengthscale
        data.tau2,
        round(tools.rmse(data, mag=0).item(), 2),          # RMS  = row[6]
        round(tools.rmse(data, scale=1/86400).item(), 2),  # RMSE = row[7]
        round(tools.rmse_U(data, mag=0).item(), 2),        # RMS_U  = row[8]
        round(tools.rmse_U(data, scale=1/86400).item(), 2),# RMSE_U = row[9]
        round(tools.rmse_V(data, mag=0).item(), 2),        # RMS_V  = row[10]
        round(tools.rmse_V(data, scale=1/86400).item(), 2) # RMSE_V = row[11]
    ]
    grouped_data[data.date].append(row)

# Prepare lists for the x- and y-values
# Zonal
zonal_lengthscales_500, rmse_u_rms_ratios_500 = [], []
zonal_lengthscales_700, rmse_u_rms_ratios_700 = [], []

# Meridional
meridional_lengthscales_500, rmse_v_rms_ratios_500 = [], []
meridional_lengthscales_700, rmse_v_rms_ratios_700 = [], []

# Temporal
temporal_lengthscales_500, rmse_rms_ratios_500 = [], []
temporal_lengthscales_700, rmse_rms_ratios_700 = [], []

# Single pass: gather zonal, meridional, temporal data
for date, rows in grouped_data.items():
    for row in rows:
        level = row[0]

        # --- Temporal (row[2] = l1, row[6] = RMS, row[7] = RMSE) ---
        temporal_l = row[2]
        rms_total = row[6]
        rmse_total = row[7]
        if rms_total > 0:
            ratio_total = rmse_total / rms_total
            # x-value for temporal = just the temporal lengthscale
            scaled_temporal = temporal_l
            if level == "500 mb":
                temporal_lengthscales_500.append(scaled_temporal)
                rmse_rms_ratios_500.append(ratio_total)
            elif level == "700 mb":
                temporal_lengthscales_700.append(scaled_temporal)
                rmse_rms_ratios_700.append(ratio_total)

        # --- Zonal (row[3] = l2, row[8] = RMS_U, row[9] = RMSE_U) ---
        zonal_l = row[3]
        rms_u = row[8]
        rmse_u = row[9]
        if rms_u > 0:
            ratio_u = rmse_u / rms_u
            scaled_zonal = zonal_l / rms_u
            if level == "500 mb":
                zonal_lengthscales_500.append(scaled_zonal)
                rmse_u_rms_ratios_500.append(ratio_u)
            elif level == "700 mb":
                zonal_lengthscales_700.append(scaled_zonal)
                rmse_u_rms_ratios_700.append(ratio_u)

        # --- Meridional (row[4] = l3, row[10] = RMS_V, row[11] = RMSE_V) ---
        meridional_l = row[4]
        rms_v = row[10]
        rmse_v = row[11]
        if rms_v > 0:
            ratio_v = rmse_v / rms_v
            scaled_meridional = meridional_l / rms_v
            if level == "500 mb":
                meridional_lengthscales_500.append(scaled_meridional)
                rmse_v_rms_ratios_500.append(ratio_v)
            elif level == "700 mb":
                meridional_lengthscales_700.append(scaled_meridional)
                rmse_v_rms_ratios_700.append(ratio_v)

# Gather all y-values together for a common y-range
zonal_all       = rmse_u_rms_ratios_500 + rmse_u_rms_ratios_700
meridional_all  = rmse_v_rms_ratios_500 + rmse_v_rms_ratios_700
temporal_all    = rmse_rms_ratios_500    + rmse_rms_ratios_700

# If you only expect non-negative values, you can start at 0
combined_ratios = zonal_all + meridional_all + temporal_all

common_y_min = 0
common_y_max = 0
if combined_ratios:
    data_max = max(combined_ratios)
    common_y_max = data_max * 1.05     # add buffer so top points are fully visible

###############################################################################
# ZONAL PLOT
###############################################################################
plt.figure()
plt.scatter(zonal_lengthscales_500, rmse_u_rms_ratios_500, label='500 mb')
plt.scatter(zonal_lengthscales_700, rmse_u_rms_ratios_700, label='700 mb')

plt.ylim([common_y_min, common_y_max])
plt.xlabel("Zonal Lengthscale / Zonal RMS")
plt.ylabel("Zonal RMSE / Zonal RMS")
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig("results/rmse_vs_zonal.png")
plt.close()

###############################################################################
# MERIDIONAL PLOT
###############################################################################
plt.figure()
plt.scatter(meridional_lengthscales_500, rmse_v_rms_ratios_500, label='500 mb')
plt.scatter(meridional_lengthscales_700, rmse_v_rms_ratios_700, label='700 mb')

plt.ylim([common_y_min, common_y_max])
plt.xlabel("Meridional Lengthscale / Meridional RMS")
plt.ylabel("Meridional RMSE / Meridional RMS")
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig("results/rmse_vs_meridional.png")
plt.close()

###############################################################################
# TEMPORAL PLOT
###############################################################################
plt.figure()
plt.scatter(temporal_lengthscales_500, rmse_rms_ratios_500, label='500 mb')
plt.scatter(temporal_lengthscales_700, rmse_rms_ratios_700, label='700 mb')

plt.ylim([common_y_min, common_y_max])
plt.xlabel("Temporal Lengthscale / RMS")
plt.ylabel("RMSE / RMS")
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig("results/rmse_vs_temporal.png")
plt.close()
