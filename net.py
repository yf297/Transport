import torch.nn as nn
import torch
def initialize(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class Flow(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Flow, self).__init__()
    
        layers = [nn.Linear(d + 1, h), nn.Mish()]
        for _ in range(L - 1):
            layers += [nn.Linear(h, h), nn.Mish()]
        layers.append(nn.Linear(h, d))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, xyt):
        xy = xyt[0:2]
        t = xyt[2:]
        return xy + t*self.network(xyt) 
    
class Vel(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Vel, self).__init__()
        
        layers = [nn.Linear(d + 1, h), nn.Mish()]
        for _ in range(L - 1):
            layers += [nn.Linear(h, h), nn.Mish()]
        layers.append(nn.Linear(h, d))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, xyt):
        return self.network(xyt)

class Mean(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Mean, self).__init__()
        layers = [nn.Linear(d, h) , nn.Mish()]   
        for _ in range(L - 1):
            layers += [nn.Linear(h, h) , nn.Mish()]
        layers.append( nn.Linear(h, 1) )
        self.network = nn.Sequential(*layers)
        
    def forward(self, a):
        return self.network(a)