import torch
class NeuralFlow(torch.nn.Module):
    def __init__(
        self,
        dim: int = 2,
        depth: int = 3,
        width: int = 32,
        blocks: int = 3,
        dropout: float = 0.8,
    ):
        super().__init__()
        self.dim        = dim
        self.depth      = depth
        self.width      = width
        self.blocks     = blocks
        self.dropout    = dropout
        self.blocks_net = torch.nn.ModuleList([self._make_block() for _ in range(blocks)])

    def _make_block(self) -> torch.nn.Sequential:
        layers: list[torch.nn.Module] = []

        lin = torch.nn.Linear(self.dim + 1, self.width)
        torch.nn.init.xavier_uniform_(lin.weight, gain=0.8)
        layers.append(lin)
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(self.dropout))

        for _ in range(self.depth - 1):
            lin = torch.nn.Linear(self.width, self.width)
            torch.nn.init.xavier_uniform_(lin.weight, gain=0.8)
            layers.append(lin)
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(self.dropout))

        lin = torch.nn.Linear(self.width, self.dim)
        torch.nn.init.xavier_uniform_(lin.weight, gain=0.2)
        layers.append(lin)
        
        return torch.nn.Sequential(*layers)
    
    def project_weights(self, eps: float = 1e-12) -> None:
        for net in self.blocks_net:
            linear_layers = [m for m in net if isinstance(m, torch.nn.Linear)]
            for i, module in enumerate(linear_layers):
                w = module.weight
                if i == 0:
                    w_sub = w[:, -2:]
                    norm = torch.linalg.matrix_norm(w_sub, ord=2)
                    if norm > 1 + eps:
                        with torch.no_grad():
                            scale = 1 / (norm + eps)
                            module.weight.data[:, -2:].mul_(scale)
                else:
                    norm = torch.linalg.matrix_norm(w, ord=2)
                    if norm > 1 + eps:
                        with torch.no_grad():
                           scale = 1 / (norm + eps)
                           module.weight.data.mul_(scale)
                            
    def inspect_weights(self) -> list[float]:
        norms = []
        for net in self.blocks_net:
            net_norms = []
            linear_layers = [m for m in net if isinstance(m, torch.nn.Linear)]
            for i, module in enumerate(linear_layers):
                weight = module.weight
                if i == 0:
                    weight = weight[:, -2:]  
                net_norms.append(torch.linalg.matrix_norm(weight, ord=2))
            norms.append(torch.prod(torch.stack(net_norms)).item())
        print(norms)

    def forward(self, 
                TXY: torch.Tensor
    ) -> torch.Tensor:
        T = TXY[..., :1]
        XY = TXY[..., 1:]
        for net in self.blocks_net:
            XY = XY + T * net(TXY)
            TXY = torch.cat([T, XY], dim = -1)  
        return XY
    
