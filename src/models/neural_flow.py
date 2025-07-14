import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import train.optim

class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.W =  nn.Linear(width, width)
        self.V = nn.Linear(width, width)

    def forward(self, h):
        return F.tanh(h + self.V(F.tanh(self.W(h))))  
    
class SpaceTime(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.W = nn.Linear(3, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        self.V = nn.Linear(width, 2)
        
    def forward(self, TXY):
        h = F.tanh(self.W(TXY) )
        for block in self.blocks:
            h = block(h)
        return self.V(h)
    
##########################################################################################

class Velocity(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.space_time = SpaceTime(width, depth)
        
    def forward(self, TXY):
        out = self.space_time(TXY)
        return out
    
##########################################################################################
class ODE(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = velocity
    
    def func(self, t, XY):
        tXY = torch.cat([t.repeat(XY.size(0),1), XY], dim=-1)
        return self.velocity(tXY)

    def flow_instant(self, t, XY):
        t0 = torch.zeros(1, device=t.device)
        t_array = torch.cat([t.unsqueeze(0), t0], dim = -1)
        if torch.allclose(t, t0):
            A = XY
        else:
            A = odeint(self.func, XY, t_array, method="euler", options={"step_size":0.05})[-1]
        return A
    
    def forward(self, TXY):
        TXY = torch.atleast_2d(TXY)
        XY = TXY[..., 1:]
        A = XY.clone()  

        T = TXY[..., 0]  
        T_unique = torch.unique(T.detach())  
        
        for t in T_unique:
            idx = torch.nonzero(T == t, as_tuple=True)[0]  
            result = self.flow_instant(t, XY[idx, :])
            A[idx, :] = result
        return A
