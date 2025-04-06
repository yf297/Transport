import gpytorch
import torch

def observations(gpFlow, T, XY):
    gpFlow.eval()
    TXY = torch.cat([torch.cat([t.repeat(XY.shape[0],1), XY], dim = -1) for t in T], dim = 0)
    with gpytorch.settings.prior_mode(True):
        Z = gpFlow(TXY).sample()
    gpFlow.train()
    return Z.reshape(T.shape[0], -1)