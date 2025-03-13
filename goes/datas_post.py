import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import linregress

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
    data for data in datas_fit]

frames = [2]

for data in filtered_fit:
    date_dir = os.path.join(f"results/plots", str(data.date))
    minute_dir = os.path.join(date_dir, str(data.minutes))
    os.makedirs(minute_dir, exist_ok=True)

    for frame_num in frames:
        indices = data.indices[:600]
        
        fig = data.plot_both(indices = indices, frame=frame_num)[0]
        plot_path = os.path.join(minute_dir, f"true_plot_frame_{frame_num}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        
