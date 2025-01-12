import torch

def D_phi(tx, flow):
    func = lambda tx: flow(tx)
    return torch.func.jacrev(func)(tx)



def PDE(tx, flow, vel):

    loss = torch.nn.L1Loss()
    v =  torch.vmap(vel)(tx)
    
    D =  torch.vmap(D_phi, in_dims=(0, None))(tx, flow)
    Dt = D[:, :, 0]
    Dx = D[:, :, 1:]
    rhs = torch.matmul(Dx,v.unsqueeze(-1)).squeeze(2)
    lhs = Dt
    
    return loss(lhs, -1*rhs) 