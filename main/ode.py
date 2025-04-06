import torch
from torchdiffeq import odeint

class Flow:
    def __init__(self, vel):
        self.vel = vel

    def __call__(self, T, XY):
        t0 = torch.zeros(1, device=T.device) + 1e-8
        solution = []
        T = torch.atleast_1d(T)
        for t in T:
            tt0 = torch.cat([t.unsqueeze(0), t0])
            solution.append(odeint(self.vel, torch.atleast_2d(XY), tt0, method='dopri8')[-1])
        return torch.cat(solution, dim = 0)

class DFlow:
    def __init__(self, flow, normalizeTime, normalizeSpace):
        self.flow = flow
        self.normalizeTime = normalizeTime
        self.normalizeSpace = normalizeSpace

    def __call__(self, t, XY):
        def f(txy):
            t_norm = self.normalizeTime(txy[0])
            xy_norm = self.normalizeSpace(torch.atleast_2d(txy[1:]))
            return self.flow(t_norm, xy_norm)[0]
        
        tXY = torch.cat([t.repeat(XY.shape[0], 1), XY], dim=-1)
        return torch.vmap(torch.func.jacrev(f))(tXY)

class Vel:
    def __init__(self, flow, normalizeTime, normalizeSpace):
        self.flow = flow
        self.normalizeTime = normalizeTime
        self.normalizeSpace = normalizeSpace

    def __call__(self, t, XY):
        Jacobians = DFlow(
            self.flow, 
            self.normalizeTime, 
            self.normalizeSpace
        )(t, XY)

        Dt = Jacobians[..., 0:1]
        Dx = Jacobians[..., 1:]
        v = torch.linalg.solve(Dx, -Dt).squeeze(-1)
        return v


