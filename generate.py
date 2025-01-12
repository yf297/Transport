import torch

def time(n):
    return torch.linspace(0, n, n + 1)

def space_time(t,x):
    n = t.shape[0] - 1
    t_expanded = t[:, None].repeat_interleave(x.shape[0], dim=0)
    x_expanded = x.repeat(n + 1, 1)
    return torch.cat([t_expanded, x_expanded], dim=1)