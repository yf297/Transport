import warnings
warnings.filterwarnings('ignore')

import pickle
import gpytorch
from main import model, optimize, net, ode, tools

sat = "hrrr" # or goes

with open(f'datas/{sat}/datas_pre.pkl', 'rb') as f:
    datas = pickle.load(f)

indices = tools.point_sampling(datas[0].XY, min_dist=15000, max_samples=1500)

for i in range(len(datas)):
    datas[i].indices = indices
    
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu = 5/2, ard_num_dims = 3))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = model.GP(kernel, likelihood)
    flow = net.Flow(L = 4)
    
    datas[i].gp = gp
    datas[i].flow = flow
    optimize.gp(datas[i], num_epochs=100)
    optimize.fl_vecchia(datas[i], num_epochs=100)
    
with open(f'datas/{sat}/datas_fit.pkl', 'wb') as f:
    pickle.dump(datas, f)