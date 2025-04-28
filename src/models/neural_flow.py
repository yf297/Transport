import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class SpaceTimeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        width: int,
        depth: int,
        p0: float = 0.0,  
        p: float = 0.5
    ):
        super().__init__()

        # ── Spatial net h(x) → ℝ^width ───────────────────────────────
        spatial_layers: list[nn.Module] = []
        lin = nn.Linear(dim, width, bias = False)
        nn.init.xavier_uniform_(lin.weight, gain = 0.8)
        spatial_layers += [lin]
        self.h_net = nn.Sequential(*spatial_layers)

        # ── Time net s(t) → ℝ^width ─────────────────────────────────
        time_layers: list[nn.Module] = []
        lin = nn.Linear(1, width, bias = False)
        nn.init.constant_(lin.weight, 0)
        time_layers += [lin]
        self.s_net = nn.Sequential(*time_layers)

        # ── Combine & project f(h+s) → ℝ^dim ─────────────────────────
        f_layers: list[nn.Module] = []
        f_layers += [nn.ReLU(), nn.Dropout(p0)]        
        for _ in range(depth-1):
            lin = nn.Linear(width, width)
            nn.init.xavier_uniform_(lin.weight, gain = 0.8)
            f_layers += [lin, nn.ReLU(), nn.Dropout(p)]
        lin = nn.Linear(width, dim)
        nn.init.xavier_uniform_(lin.weight, gain = 0.2)
        f_layers += [lin]
        self.f_net = nn.Sequential(*f_layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.h_net(x)         
        s = self.s_net(t)        
        z = h + s                 
        return self.f_net(z)

class NeuralFlow(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        width: int = 48,
        depth: int = 3,
        num_blocks: int = 3
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(dim, width, depth)
            for _ in range(num_blocks)
        ])
        
    def inspect_weights(self):
        norms = []
        for block in self.blocks:
            block_norms = []
            for net in (block.h_net, block.f_net):
                for module in net.modules():
                    if isinstance(module, nn.Linear):
                        block_norms.append(
                            torch.linalg.matrix_norm(module.weight, ord=2)
                        )
            prod = torch.prod(torch.stack(block_norms)).item()
            norms.append(prod)
        print(norms)

    def forward(self, TXY: torch.Tensor) -> torch.Tensor:
        t = TXY[..., :1]
        xy = TXY[..., 1:]
        out = xy
        for block in self.blocks:
            out = out + t*block(t, out)
        return out
