import torch
import torch.nn as nn
import torch.nn.functional as F

class Flow(nn.Module):
    def __init__(self, d=2, L=1, w=32):
        super().__init__()
              
        layers = []
        layers.append(nn.Linear(d + 1, w))
        layers.append(nn.Tanh())
        for _ in range(L - 1):
            layers.append(nn.Linear(w, w))
            layers.append(nn.Tanh())
        final_layer = nn.Linear(w, d)
        layers.append(final_layer)
        
        self.net = nn.Sequential(*layers)
        
        for module in list(self.net.children()):
            if isinstance(module, nn.Linear):
               nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.constant_(final_layer.weight, 0.0)
        
        
    def forward(self, t, xy):
        tXY = torch.cat([t.repeat(xy.shape[0],1), xy], dim=-1)
        net_out = self.net(tXY)
        return xy + t*net_out



