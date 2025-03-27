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

def D_flow(t,XY,flow, mean=0, std=1, time_std = 1):
    f = lambda txy: flow(txy[0]/time_std, (torch.atleast_2d(txy[1:]) - mean)/std )[0] 
    tXY = torch.cat([t.repeat(XY.shape[0],1), 
                            XY], dim = -1)
    return torch.vmap(torch.func.jacrev(f))(tXY)


def Vel_hat(data, scale = True):
    input_mean = 0
    input_std = 1
    time_std = 1
    if scale:
        input_mean = data.input_mean
        input_std = data.input_std 
        time_std = data.time_std
    def vel_func(t,XY): 
        Jacobians = D_flow(t, XY, data.flow, input_mean,input_std, time_std)
        Dt = Jacobians[..., 0:1]
        Dx = Jacobians[..., 1:]
        v = torch.linalg.solve(Dx,-1*Dt).squeeze(-1)
        return v
    return vel_func


def D_vel_norm(t, XY, vel): 
    f  = lambda txy: vel(txy[0], torch.atleast_2d(txy[1:])  )[0] 
    
    tXY = torch.cat([t.repeat(XY.shape[0],1), 
                            XY], dim = -1)
    
    Jacobians = torch.vmap(torch.func.jacrev(f))(tXY)

    # First column (∂f/∂t)
    col0 = Jacobians[:, :, 0]
    col0_norm = torch.mean(torch.linalg.vector_norm(col0, ord=2, dim=1))

    # Remaining 2x2 block (∂f/∂XY)
    block2x2 = Jacobians[:, :, 1:]
    block2x2_norm = torch.mean(torch.linalg.matrix_norm(block2x2, ord=2, dim=(1, 2)))

    return col0_norm, block2x2_norm

def pen_D_vel_mean(T, XY, vel, L1 = 1, L2 = 1):
    col0_norms, block2x2_norms = zip(*[D_vel_norm(t, XY, vel) for t in T])
    col0_mean = torch.mean(torch.stack(col0_norms))
    block2x2_mean = torch.mean(torch.stack(block2x2_norms))
    z = torch.zeros(1, device=XY.device)
    return torch.max(z,col0_mean - L1)**2, torch.max(z,block2x2_mean - L2)**2


def vel_norm(t, XY, vel):
    vs = vel(t,XY)
    return torch.mean(torch.norm(vs, dim =1))

def pen_vel_mean(T, XY, vel, M = 1):
    ns = [vel_norm(t,XY, vel) for t in T]
    z = torch.zeros(1, device=XY.device)
    return torch.max(z,torch.mean(torch.stack(ns)) - M)**2


def log_det(t, XY, flow): 
    f  = lambda txy: flow(txy[0], torch.atleast_2d(txy[1:])  )[0] 
    
    tXY = torch.cat([t.repeat(XY.shape[0],1), 
                            XY], dim = -1)
    
    Jacobians = torch.vmap(torch.func.jacrev(f))(tXY)
    dets = torch.linalg.det(Jacobians[..., 1:])  

    epsilon = 1e-6 
    log_dets = torch.log(torch.abs(dets) + epsilon)

    return torch.mean(log_dets)

def pen_det_mean(T, XY, flow):
    ns = [log_det(t,XY, flow) for t in T]
    return -1*torch.mean(torch.stack(ns))