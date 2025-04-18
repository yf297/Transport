import torch
from torch import Tensor, nn

class NeuralFlow(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        depth: int = 3,
        width: int = 32,
        blocks: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dim        = dim
        self.depth      = depth
        self.width      = width
        self.blocks     = blocks
        self.dropout    = dropout
        self.blocks_net = nn.ModuleList([self._make_block() for _ in range(blocks)])

    def _make_block(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers += [nn.Linear( self.dim + 1, self.width)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(self.dropout)]
        for _ in range(self.depth - 1):
            layers += [nn.Linear(self.width, self.width)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(self.dropout)]
        layers += [nn.Linear(self.width, self.dim)]
        return nn.Sequential(*layers)


    def project_weights(self, eps: float = 1e-12) -> None:
        for net in self.blocks_net:
            norms = []
            for module in net:
                if isinstance(module, nn.Linear):
                    norms.append(torch.linalg.matrix_norm(module.weight, ord=2))
            prod = torch.prod(torch.stack(norms))
            if prod > self.blocks + eps:
                alpha = (self.blocks / (prod + eps)) ** (1.0 / len(norms))
                with torch.no_grad():
                    for module in net:
                        if isinstance(module, nn.Linear):
                            module.weight.mul_(alpha)

    def forward(self, 
                TXY: Tensor
    ) -> Tensor:
        T = TXY[..., :1]
        XY = TXY[..., 1:]
        for net in self.blocks_net:
            XY = XY + (T / self.blocks) * net(TXY)
            TXY = torch.cat([T, XY], dim = -1)  
        return XY