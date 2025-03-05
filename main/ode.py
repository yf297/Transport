import torch
from torchdiffeq import odeint
import torch.nn.functional as F

def flow_instant(t, xy, vel):
    t0 = torch.zeros(1, device=t.device) + 1e-8
    t = torch.cat([t, t0])
    solution = odeint(vel, xy, t, method='dopri8')[-1]
    return solution

def Flow(vel):
    def flow_func(t, xy):
        return flow_instant(t.unsqueeze(0), xy, vel)
    return flow_func

def D_flow(tXY,data):
    XY = data.XY[data.indices,:]
    mean = XY.mean(dim=-2, keepdim=True)
    std = std =  XY.view(-1).std() + 1e-6
    
    f = lambda txy: data.flow(txy[0], (torch.atleast_2d(txy[1:]) - mean)/std )[0]
    return torch.vmap(torch.func.jacrev(f))(tXY)

def Vel_hat(data):
    def vel_func(t,xy): 
        tXY = torch.cat([t.repeat(xy.shape[0],1), 
                              xy], dim = -1)
        
        Jacobians = D_flow(tXY, data)
        Dt = Jacobians[..., 0:1]
        Dx = Jacobians[..., 1:]
        v = torch.linalg.solve(Dx,-1*Dt).squeeze()
        return v
    return vel_func
