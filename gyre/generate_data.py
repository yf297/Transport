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

def gyre(l = -1, r = 1,
         u = 1, d = -1, 
         k1 = 50, k2 = 50,
         n = 5,
         l1 = 1.0, l2 = 1.0, l3 = 1.0):
    
    x = torch.linspace(l,r,k1)
    y = torch.linspace(u,d,k2)
    t = torch.linspace(0,1,n)

    X,Y = torch.meshgrid(x,y,indexing='xy')
    XY = torch.stack([X,Y], dim = -1).reshape(-1,2)
    T = t
   
    T_repeated = T.repeat_interleave(XY.shape[0]) 
    XY_tiled = XY.repeat(T.shape[0], 1)  
    TXY = torch.cat([T_repeated.unsqueeze(1), XY_tiled], dim=1)
    flow = ode.Flow(vel)
    spaceTimeKernel = model.SpaceTimeKernel(l0 = 2, l1 = 0.1, l2 = 0.1)
    gpFlow = model.GPFlow(spaceTimeKernel, flow)
    gpFlow.eval();  
    with gpytorch.settings.prior_mode(True):
        Z = gpFlow(TXY ).sample()
    Z = Z.reshape(T.shape[0], -1)
    
    data = model.data(T,XY,Z, grid_size = 1, k0 = 0, k1 = 1)
    return data