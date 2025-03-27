import warnings
warnings.filterwarnings('ignore')

import sys
import os
import time
import pickle
import torch
import numpy as np
import random
import gc
from collections import defaultdict

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import model, optimize, net


SEED = 23
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


# File paths for preprocessed data and fitted data
pre_file_path = 'datas/datas_pre.pkl'
fit_file_path = 'datas/datas_fit.pkl'

# Load preprocessed data
with open(pre_file_path, 'rb') as f:
    datas_pre = pickle.load(f)

# Load existing fitted data if available, else start empty
'''if os.path.exists(fit_file_path):
    with open(fit_file_path, 'rb') as f:
        datas_fit = pickle.load(f)
else:
    datas_fit = []'''

datas_fit = []
# Identify which dates are already fitted
fitted_dates = {data.date for data in datas_fit}
# Filter new data by excluding already fitted dates
new_datas = [data for data in datas_pre if data.date not in fitted_dates]

if not new_datas:
    print("No new data to fit.")
else:
    i = 1
    for data in new_datas:
        data.flow = net.Flow(L = 4)
        
        print(i)
        i += 1
        gc.collect()
        torch.cuda.empty_cache()
        start_time = time.time()
        optimize.fit(data, num_epochs=100, fix_t=False)
        end_time = time.time()
        data.time = end_time - start_time
        

    # Extend existing fits with any newly fitted data
    datas_fit.extend(new_datas)

    for path in [fit_file_path, pre_file_path]:
        with open(path, 'wb') as f:
            pickle.dump(datas_fit, f)