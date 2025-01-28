import torch

def D_phi(TXY, flow):
    func = lambda txy: flow(txy)
    return torch.vmap(torch.func.jacrev(func))(TXY)

def v_hat(TXY, flow):
    Jacobians = D_phi(TXY, flow)
    Jacobians_t = Jacobians[..., 0]
    Jacobians_x = Jacobians[..., 1:]

    v = torch.linalg.solve(-1*Jacobians_x, Jacobians_t)
    return v


def PDE(TXY, flow, vel):

    loss = torch.nn.L1Loss(reduction="mean")
    
    v =  torch.vmap(vel)(TXY)
    D =  torch.vmap(D_phi, in_dims=(0, None))(TXY, flow)
    
    Dt = D[:, :, 0]
    Dx = D[:, :, 1:]
    rhs = torch.matmul(Dx,v.unsqueeze(-1)).squeeze(2)
    lhs = Dt
    
    return loss(lhs, -1*rhs)

