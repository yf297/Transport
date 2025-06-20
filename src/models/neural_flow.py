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
        return F.tanh(h + self.V(F.tanh(self.W(h))))

class Time(nn.Module):
    def __init__(self, width, depth, out):
        super().__init__()
        self.W = nn.Linear(1, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        self.V = nn.Linear(width, out)
    def forward(self, T):
        h = F.tanh(self.W(T))
        for block in self.blocks:
            h = block(h)
        return self.V(h)

class Velocity(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        self.W = nn.Linear(2, width, bias=False)
        self.b = nn.Linear(1, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        self.V = nn.Linear(width, 2)
        self.time0 = Time(16, 1, 2)
        self.time1 = Time(16, 1, 2)
    
    def forward(self, TXY):
        T = TXY[..., 0:1]
        XY = TXY[..., 1:3]
        h = F.tanh(self.W(XY + self.time0(T)))
        for block in self.blocks:
            h = block(h)
        return self.V(h) + self.time1(T)

class NeuralFlow(nn.Module):
    def __init__(self, width=32, depth=2):
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
        A = XY + self.V(h)
        TA = torch.cat([T, A], dim=-1)
        return TA
    
class ODEFlow(nn.Module):
    def __init__(self, velocity, method='euler', dt=0.05):
        super().__init__()
        self.method = method
        self.dt = dt
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
            A = odeint(self.func, XY, t_array, method=self.method, options={"step_size":self.dt})[-1]
        return A
    def forward(self, TXY):
        XY = TXY[..., 1:]
        TA = TXY.clone()  

        t_all = TXY[..., 0]  
        t_unique = torch.unique(t_all.detach())  

        for t in t_unique:
            idx = torch.nonzero(t_all == t, as_tuple=True)[0]  
            result = self.flow_instant(t_all[idx][0], XY[idx, :])
            TA[idx, 1:] = result
        return TA


    
class Flow(nn.Module):
    def __init__(self, velocity=None):
        super().__init__()
        if velocity is None:
            self.velocity = Velocity()
            self.flow = NeuralFlow()
        else:
            self.velocity = velocity
            self.flow = ODEFlow(velocity)
    def forward(self, TXY):
        return self.flow(TXY)