import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.W = nn.Linear(width, width)
        self.V = nn.Linear(width, width)
    def forward(self, h):
        return self.V(F.relu(self.W(h)))

class Velocity(nn.Module):
    def __init__(self, width=32, depth=4):
        super().__init__()
        self.W = nn.Linear(2, width, bias=False)
        self.b = nn.Linear(1, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        self.V = nn.Linear(width, 2)

    def forward(self, TXY):
        T = TXY[..., 0:1]
        XY = TXY[..., 1:3]
        h = F.relu(self.W(XY) + self.b(T))
        for block in self.blocks:
            h = block(h)
        return self.V(h)
    

class ODE(nn.Module):
    def __init__(self, velocity, method='euler', step_size=1/9):
        super().__init__() 
        self.velocity = velocity
        self.method = method
        self.step_size = step_size
        
    def func(self, t, XY):
        tXY = torch.cat([t.repeat(XY.size(0),1), XY], dim=-1)
        return self.velocity(tXY)

    def flow_instant(self, t, XY):
        t0 = torch.zeros(1, device=t.device) 
        t_array = torch.cat([t.unsqueeze(0), t0], dim = -1)
        if torch.allclose(t, t0):
            A = XY
        else:
            A = odeint(self.func, XY, t_array, method=self.method)[-1]
        tA = torch.cat([t.repeat(A.size(0),1), A], dim=-1)
        return tA

    def __call__(self, TXY):
        T = torch.unique(TXY[:, 0:1])
        idx = [torch.nonzero(TXY[:,0] == t_i, as_tuple=False).squeeze(1) for t_i in T]
        XY = TXY[..., 1:3]
        TA = torch.cat([self.flow_instant(T[i], XY[idx[i],:]) for i in range(T.size(0))], dim = 0)
        return TA

    
class Flow(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = velocity
        self.flow = ODE(self.velocity)
    def forward(self, TXY):
        TA = self.flow(TXY)
        return TA