import torch

def D_phi(xyt, flow):
    func = lambda xyt: flow(xyt)
    return torch.func.jacrev(func)(xyt)

def v_hat(xyt, flow):
    Jacobians = D_phi(xyt, flow)
    Jacobians_t = Jacobians[:, 2]
    Jacobians_x = Jacobians[:, 0:2]

    v = torch.linalg.solve(-1*Jacobians_x, Jacobians_t)
    return v


def PDE(spacetime, flow, vel):

    loss = torch.nn.L1Loss(reduction="mean")
    
    v =  torch.vmap(vel)(spacetime)
    D =  torch.vmap(D_phi, in_dims=(0, None))(spacetime, flow)
    
    Dt = D[:, :, 2]
    Dx = D[:, :, 0:2]
    rhs = torch.matmul(Dx,v.unsqueeze(-1)).squeeze(2)
    lhs = Dt
    
    return loss(lhs, -1*rhs) 

