import torch
import torch.nn.functional as F




'''def PDE(TXY, flow, vel):
    loss = torch.nn.L1Loss(reduction="mean").to(TXY.device) 

    v = vel(TXY)
    Ds = D_flow(TXY, flow)
    
    Dt = Ds[:, :, 0]
    Dx = Ds[:, :, 1:]
    rhs = torch.matmul(Dx,v.unsqueeze(-1)).squeeze(2)
    lhs = Dt

    return loss(lhs, -1*rhs)


def check(TXY, flow):
    f = lambda txy: txy[1:] - flow(torch.atleast_2d(txy))[0]
    Ds =  torch.vmap(torch.func.jacrev(f))(TXY)
    Dx = Ds[:, :, 1:]
    return Dx'''

