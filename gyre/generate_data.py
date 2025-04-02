import sys
import torch
import os
import gpytorch

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import  model, tools, ode, simulate

def vel(t, xy, A=1.0, omega= 2 * 3.14159, epsilon=0.25):
        x = xy[..., 0]  + 0.5
        y = xy[..., 1]

        a = epsilon * torch.sin(torch.tensor(0))
        b = 1 - 2 * a  

        f = a * x**2 + b * x
        df_dx = 2 * a * x + b

        u = -3.14159 * A * torch.sin(3.14159 * f) * torch.cos(3.14159 * y)
        v = 3.14159 * A * torch.cos(3.14159 * f) * torch.sin(3.14159 * y) * df_dx

        return torch.stack([u, v], dim=-1)/7.5

def gyre(l = -1,r = 1,
         u = 1,d = -1, 
         k1 = 50, k2 = 50,
         n = 5,
         l1 = 1.0, l2 = 1.0, l3 = 1.0):
    
    x = torch.linspace(l,r,k1)
    y = torch.linspace(d,u,k2)
    t = torch.linspace(0,1,n)

    X,Y = torch.meshgrid(x,y,indexing='xy')
    XY = torch.stack([X,Y], dim = -1).reshape(-1,2)
    T = t
   

    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu = 5/2, ard_num_dims = 3))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 0.01
    hypers = {
        'base_kernel.lengthscale': torch.tensor([l1, l2, l3]),
        'outputscale': torch.tensor(1),
        }
    kernel.initialize(**hypers);
    flow = ode.Flow(vel)
    gp = model.GP(kernel, likelihood)
    Z = simulate.observations(gp, flow, T, XY)
    data = model.data(T,XY,Z)
    return data