import torch
class NeuralFlow(torch.nn.Module):
    def __init__(
        self,
        dim: int = 2,
        depth: int = 3,
        width: int = 32,
        blocks: int = 4,
        dropout: float = 0.3,
        init_gain: float = 0.8
    ):
        super().__init__()
        self.dim        = dim
        self.depth      = depth
        self.width      = width
        self.blocks     = blocks
        self.dropout    = dropout
        self.init_gain = init_gain
        self.blocks_net = torch.nn.ModuleList([self._make_block() for _ in range(blocks)])

    def _make_block(self) -> torch.nn.Sequential:
        layers: list[torch.nn.Module] = []

        lin = torch.nn.Linear(self.dim + 1, self.width)
        torch.nn.init.xavier_uniform_(lin.weight, gain=self.init_gain)
        layers.append(lin)
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(self.dropout))

        for _ in range(self.depth - 1):
            lin = torch.nn.Linear(self.width, self.width)
            torch.nn.init.xavier_uniform_(lin.weight, gain=self.init_gain)
            layers.append(lin)
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(self.dropout))

        lin = torch.nn.Linear(self.width, self.dim)
        torch.nn.init.xavier_uniform_(lin.weight, gain=self.init_gain)
        layers.append(lin)
        
        return torch.nn.Sequential(*layers)


    def project_weights(self, eps: float = 1e-12) -> None:
        for net in self.blocks_net:
            norms = []
            linear_layers = [m for m in net if isinstance(m, torch.nn.Linear)]
            for i, module in enumerate(linear_layers):
                weight = module.weight
                if i == 0:
                    weight = weight[:, -2:]
                norms.append(torch.linalg.matrix_norm(weight, ord=2))
            
            prod = torch.prod(torch.stack(norms))
            if prod > self.blocks + eps:
                alpha = (self.blocks / (prod + eps)) ** (1.0 / len(norms))
                with torch.no_grad():
                    for i, module in enumerate(linear_layers):
                        if i == 0:
                            module.weight[:, -2:] *= alpha 
                        else:
                            module.weight *= alpha
    
    def inspect_weight_norms(self) -> list[float]:
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
        return norms

    def forward(self, 
                TXY: torch.Tensor
    ) -> torch.Tensor:
        T = TXY[..., :1]
        XY = TXY[..., 1:]
        for net in self.blocks_net:
            XY = XY + (T / self.blocks) * net(TXY)
            TXY = torch.cat([T, XY], dim = -1)  
        return XY