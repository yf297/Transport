import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.linear1 = nn.Linear(h, h)
        self.ln1 = nn.LayerNorm(h)
        self.activation = nn.Mish()
        self.linear2 = nn.Linear(h, h)
        self.ln2 = nn.LayerNorm(h)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.linear1.weight,  a=0.0003)
        nn.init.kaiming_normal_(self.linear2.weight,  a=0.0003)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.ln2(x)
        return self.activation(x + residual)

                
class Flow(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Flow, self).__init__()
        
        layers = [nn.Linear(d + 1, h), nn.Mish()]
        for _ in range(L - 1):
            layers.append(ResidualBlock(h))
        layers.append(nn.Linear(h, d))
        
        self.network = nn.Sequential(*layers)
                
    def forward(self, txy):
        xy = txy[...,1:]
        t = txy[...,0:1]
        return xy + t*self.network(txy)
    

class Mean(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Mean, self).__init__()
        
        layers = [nn.Linear(d, h), nn.Mish()]
        for _ in range(L - 1):
            layers.append(ResidualBlock(h))
        layers.append(nn.Linear(h, 1))
        
        self.network = nn.Sequential(*layers)
                
    def forward(self, a):
        return self.network(a).squeeze()
    



