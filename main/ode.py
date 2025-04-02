import torch
from torchdiffeq import odeint
import torch.nn.functional as F
import torch.nn as nn
def flow_instant(t, xy, vel):
    t0 = torch.zeros(1, device=t.device) + 1e-8
    t = torch.cat([t, t0])
    solution = odeint(vel, xy, t, method='dopri8')[-1]
    return solution

def Flow(vel):
    def flow_func(t, xy):
        return flow_instant(t.unsqueeze(0), xy, vel)
    return flow_func

def D_flow(t,XY,flow, mean=0, std=1, time_normalize=1):
    f = lambda txy: flow(txy[0]/time_normalize, (torch.atleast_2d(txy[1:]) - mean)/std )[0] 
    tXY = torch.cat([t.repeat(XY.shape[0],1), 
                            XY], dim = -1)
    return torch.vmap(torch.func.jacrev(f))(tXY)


def Vel_hat(data, scale = True):
    input_mean = 0
    input_normalize = 1
    time_normalize = 1
    if scale:
        input_mean = data.input_mean
        input_normalize = data.input_normalize
        time_normalize = data.time_normalize
    def vel_func(t,XY): 
        Jacobians = D_flow(t, XY, data.flow, input_mean,input_normalize, time_normalize)
        Dt = Jacobians[..., 0:1]
        Dx = Jacobians[..., 1:]
        v = torch.linalg.solve(Dx,-1*Dt).squeeze(-1)
        return v
    return vel_func


def D_vel_norm(t, XY, vel, L1, L2): 
    f  = lambda txy: vel(txy[0], torch.atleast_2d(txy[1:])  )[0] 
    
    tXY = torch.cat([t.repeat(XY.shape[0],1), 
                            XY], dim = -1)
    
    Jacobians = torch.vmap(torch.func.jacrev(f))(tXY)

    col0 = Jacobians[:, :, 0]
    col0_norms = torch.linalg.vector_norm(col0, ord=2, dim=1)
    zero = torch.zeros(1, device=XY.device)
    col0_norms = torch.max(zero, col0_norms - L1) ** 2
    
    block2x2 = Jacobians[:, :, 1:]
    block2x2_norms = torch.linalg.matrix_norm(block2x2, ord=2, dim=(1, 2))
    block2x2_norms =  torch.max(zero, block2x2_norms - L2) ** 2
    return col0_norms.mean(), block2x2_norms.mean()

def vel_norm(t, XY, vel, M=1.0):
    vs = vel(t, XY)            
    norms = torch.linalg.vector_norm(vs, ord = float('inf'), dim=1)  
    zero = torch.zeros(1, device=XY.device)
    norms = torch.max(zero, norms - M) ** 2
    return norms.mean()

def det(t, XY, flow, threshold=0.0):
    f = lambda txy: flow(txy[0], torch.atleast_2d(txy[1:]))[0]
    tXY = torch.cat([t.repeat(XY.shape[0], 1), XY], dim=-1)
    
    jacobians = torch.vmap(torch.func.jacrev(f))(tXY)
    dets = torch.linalg.det(jacobians[..., 1:])
    zero = torch.zeros(1, device=XY.device)
    dets = torch.max(zero, threshold - dets)**2
    return dets.mean()

