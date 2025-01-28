import torch.nn as nn
import torch

class Flow(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Flow, self).__init__()
        
        layers = [nn.Linear(d+1, h), nn.Mish()]
        for _ in range(L - 1):
            layers += [nn.Linear(h, h), nn.Mish()]
        layers.append(nn.Linear(h, d))
        
        self.network = nn.Sequential(*layers)
        
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                 nn.init.xavier_uniform_(module.weight,gain = nn.init.calculate_gain("relu"))
                
    def forward(self, txy):
        xy = txy[...,1:]
        t = txy[...,0:1]
        return xy + t*self.network(txy)
    
    

class Mean(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Mean, self).__init__()
        
        layers = [nn.Linear(d, h) , nn.Mish()]   
        for _ in range(L - 1):
            layers += [nn.Linear(h, h) , nn.Mish()]
        layers.append( nn.Linear(h, 1) )
        self.network = nn.Sequential(*layers)
        
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                 nn.init.xavier_uniform_(module.weight,gain = nn.init.calculate_gain("relu"))
        
    def forward(self, a):
        return self.network(a).squeeze()
    



