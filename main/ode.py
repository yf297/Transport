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

'''def Vel_hat(flow_hat, data):
    XY = data.XY[data.indices,:]
    mean = XY.mean(dim=-2, keepdim=True)
    std = std =  XY.view(-1).std() + 1e-6
    
    def vel_func(t,xy):
        xy = (xy - mean)/std
        txy = torch.cat([t.repeat(xy.shape[0],1),
                        xy], dim = -1)
        return flow_hat.net(txy)
    return vel_func

def discretization(T, k):
    n = T.shape[0]
    if k < n:
        S = T
    else:
        points_per_interval = max( k // (n - 1), 1)
        equispaced_points = []
        for i in range(n - 1):
            interval_points = torch.linspace(
            T[i].item(), 
            T[i + 1].item(), 
            points_per_interval + 1 
            )
            if i < n - 2:  
                interval_points = interval_points[:-1]
            equispaced_points.append(interval_points)
        S = torch.cat(equispaced_points)
    return S'''