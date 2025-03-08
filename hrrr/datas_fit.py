import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import gpytorch
import torch
from main import model, optimize, net, tools

sat = "hrrr"
pre_file_path = f'datas/datas_pre.pkl'
fit_file_path = f'datas/datas_fit.pkl'

with open(pre_file_path, 'rb') as f:
    datas_pre = pickle.load(f)
if os.path.exists(fit_file_path):
    with open(fit_file_path, 'rb') as f:
        datas_fit = pickle.load(f)
else:
    datas_fit = []

fitted_dates = {data.date for data in datas_fit}
new_datas = [data for data in datas_pre if data.date not in fitted_dates]

if not new_datas:
    print("No new data to fit.")
else:
    indices = torch.randperm(datas_pre[0].m)

    for data in new_datas:
        data.indices = indices
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=5/2, ard_num_dims=3))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp = model.GP(kernel, likelihood)
        flow = net.Flow(L=5)

        data.gp = gp
        data.flow = flow
        print("fitting GP")
        optimize.gp(data, num_epochs=200)
        print("fitting flow")
        optimize.fl_vecchia(data, num_epochs=100)

    datas_fit.extend(new_datas)
    with open(fit_file_path, 'wb') as f:
        pickle.dump(datas_fit, f)
