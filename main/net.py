import torch
import torch.nn as nn
    
class Flow(nn.Module):
    def __init__(self, d=2, L=1, w=32):
        super(Flow, self).__init__()
              
        layers = [nn.Linear(d + 1, w), nn.Tanh()]
        for _ in range(L - 1):
            layers += [nn.Linear(w, w), nn.Tanh()]
        final_layer = nn.Linear(w, d)
        layers.append(final_layer)
        self.net = nn.Sequential(*layers)
        
        for module in list(self.net.children())[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="tanh")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.constant_(final_layer.weight, 0.0)
                    
                
    def forward(self, t, XY):
        tXY = torch.cat([t.repeat(XY.shape[0],1),
                             XY], dim = -1)
        return XY + t*self.net(tXY)
