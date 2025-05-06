import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthoLinear(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.W_raw = nn.Parameter(torch.empty(width, 2))
        nn.init.orthogonal_(self.W_raw, gain=1.0)
        self.bias = nn.Parameter(torch.zeros(width)) 

    def forward(self, x):
        # compute Q once per forward
        Q, _ = torch.linalg.qr(self.W_raw, mode='reduced')
        # y = Q x + b
        return F.linear(x, Q)

    def inverse(self, y):
        Q, _ = torch.linalg.qr(self.W_raw, mode='reduced')
        # undo: x = Qᵀ(y − b) = Qᵀ y + (−Qᵀ b)
        b_inv = -Q.T @ self.bias
        return F.linear(y, Q.T)

class Displacement(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        act = nn.ReLU()

        b_layers = [nn.Linear(1, width, bias = False), act]
        b_layers.append(nn.Linear(width, width, bias = False))
        self.b_net = nn.Sequential(*b_layers)
        
        s_layers = [nn.Linear(1, width, bias = False), act]
        s_layers.append(nn.Linear(width, width, bias = False))
        self.s_net = nn.Sequential(*s_layers)

        f_layers = []
        for _ in range(depth-1):
            lin = nn.Linear(width, width, bias = False)
            f_layers += [lin, act]
        lin = nn.Linear(width, width, bias = False)
        f_layers.append(lin)
        self.f_net = nn.Sequential(*f_layers)

        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.weight.data /= 2 * torch.linalg.matrix_norm(m.weight.data, ord=2)
        
        self.f_net.apply(init_fn)
        self.b_net.apply(init_fn)
        self.s_net.apply(init_fn)


    def forward(self, T, P):
        return self.f_net(self.s_net(T)*P + self.b_net(T))


class NeuralFlow(nn.Module):
    def __init__(self, width=32, depth=2, nets=3):
        super().__init__()
        self.displacements = nn.ModuleList([
            Displacement(width, depth) for _ in range(nets)
        ])
        self.proj = OrthoLinear(width)

    def project_weights(self):
        with torch.no_grad():
            for disp in self.displacements:
                layers = [m for m in disp.f_net.modules() if isinstance(m, nn.Linear)]
                norms = torch.tensor([
                    torch.linalg.matrix_norm(m.weight, ord=2) for m in layers
                ], device=layers[0].weight.device)
                prod = norms.prod()
                if prod > 1.0:
                    L = norms.numel()
                    scale = (1.0 / (prod + 1e-12)) ** (1.0 / L)
                    for m in layers:
                        m.weight.data.mul_(scale)

    def inspect_weights(self):
        for i, disp in enumerate(self.displacements):
            norms = []
            for m in disp.f_net.modules():
                if isinstance(m, nn.Linear):
                    norms.append(torch.linalg.matrix_norm(m.weight, ord=2))
            prod = torch.prod(torch.stack(norms))
            print(f"Displacement #{i} → product of spectral‐norms = {prod.item():.4f}")

    def forward(self, T, XY):
        P = self.proj(XY)
        for disp in self.displacements:
            P = P + disp(T, P)    
        A = self.proj.inverse(P)
        return A