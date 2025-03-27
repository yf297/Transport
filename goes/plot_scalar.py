import warnings
warnings.filterwarnings('ignore')

import pickle
import matplotlib.pyplot as plt
import os
import sys

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import tools

# Path to the pickled fitted data
fit_file_path = 'datas/datas_pre.pkl'

# Load the existing fitted data
with open(fit_file_path, 'rb') as f:
    datas_fit = pickle.load(f)
    

for data in datas_fit:

    date_dir = os.path.join(f"results/plots", str(data.date))
    level_dir = os.path.join(date_dir, str(data.level))
    os.makedirs(level_dir, exist_ok=True)

    fig = data.plot_observations(frame=0)
    plot_path = os.path.join(level_dir, f"obs_{0}.png")
    fig.savefig(plot_path)
    plt.close(fig)