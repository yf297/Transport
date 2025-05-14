import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowBlock(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
  
        self.s = nn.Sequential(nn.Linear(1, width), 
                            nn.Tanh(), 
                            nn.Linear(width, 2, bias = False))
        
        self.W = nn.Linear(2, width, bias=False)
        self.b = nn.Linear(1, width)
        hidden_layers = [nn.ReLU()]
        for i in range(depth - 1):
            lin = nn.Linear(width, width)
            hidden_layers += [lin, nn.ReLU()]
        self.hidden = nn.Sequential(*hidden_layers)
        self.V = nn.Linear(width, 2)

    def project_weight_norms(self, max_norms):
        norm = torch.linalg.matrix_norm(self.W.weight, ord=2)
        if norm > max_norms[0]:
            self.W.weight.data.mul_(max_norms[0] / norm)

        j = 1 
        for layer in self.hidden:
            if hasattr(layer, "weight"):
                norm = torch.linalg.matrix_norm(layer.weight, ord=2)
                if norm > max_norms[j]:
                    layer.weight.data.mul_(max_norms[j] / norm)
                j += 1

        norm = torch.linalg.matrix_norm(self.V.weight, ord=2)
        if norm > max_norms[-1]:
            self.V.weight.data.mul_(max_norms[-1] / norm)

        
    def forward(self, T, XY):
        T = torch.atleast_1d(T)
        return self.s(T)*self.V(self.hidden(self.W(XY) + self.b(T)))

    
class Warp(nn.Module):
    def __init__(self, width,depth, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            FlowBlock(width, depth) for _ in range(blocks)
        ])

    def forward(self, T, XY):
        for block in self.blocks:
            XY = XY + block(T, XY)
        return XY



class NeuralFlow(nn.Module):
    def __init__(self, width=32, depth=2, blocks=5):
        super().__init__()
        self.warp = Warp(width,depth, blocks)

        inner_norm = 2
        outer_norm = 1
        hidden_norm = (1.0/ (inner_norm * outer_norm)) ** (1 / (depth - 1))  
        self.max_norms = [inner_norm] + [hidden_norm] * (depth - 1) + [outer_norm]
    
    def project_all_weight_norms(self):
        for block in self.warp.blocks:
            block.project_weight_norms(self.max_norms)
            
    def check_weight_norm_products(self):
        for i, block in enumerate(self.warp.blocks):
            norms = []
            norms.append(torch.linalg.matrix_norm(block.W.weight, ord=2).item())
            for layer in block.hidden:
                if hasattr(layer, "weight"):
                    norms.append(torch.linalg.matrix_norm(layer.weight, ord=2).item())
            norms.append(torch.linalg.matrix_norm(block.V.weight, ord=2).item())

            product = 1.0
            for n in norms:
                product *= n

            print(f"Block {i}: product of normalized weight norms = {product:.6f}")

    def forward(self, T, XY):
        T = torch.atleast_1d(T)
        XY = torch.atleast_2d(XY)
        T = T.unsqueeze(1).unsqueeze(2)
        XY = XY.unsqueeze(0).expand(T.size(0), -1, 2)

        A = self.warp(T, XY)
        return A.reshape(2) if A.size(1) == 1 else A.reshape(-1, 2)
