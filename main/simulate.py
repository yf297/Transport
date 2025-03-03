import gpytorch
import torch

def observations(gp, flow, T, XY):
    gp.eval()
    points = torch.cat(
        [torch.cat(
        [t.repeat(XY.shape[0],1), flow(t,XY)], dim = -1) 
         for t in T])
    with gpytorch.settings.prior_mode(True):
        Z = gp(points).sample()
    gp.train()
    return Z.reshape(T.shape[0], -1)