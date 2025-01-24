import torch.nn as nn

class Flow(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Flow, self).__init__()
    
        layers = [nn.Linear(d + 1, h), nn.Mish()]
        for _ in range(L - 1):
            layers += [nn.Linear(h, h), nn.Mish()]
        
        layers.append(nn.Linear(h, d))
        self.network = nn.Sequential(*layers)
        
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight,gain = nn.init.calculate_gain("relu"))

  
    def forward(self, xyt):
        xy = xyt[..., 0:2]
        t = xyt[..., 2:3]
        return xy + t*self.network(xyt)
    
class Vel(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Vel, self).__init__()
        
        layers = [nn.Linear(d+1, h), nn.Mish()]
        for _ in range(L - 1):
            layers += [nn.Linear(h, h), nn.Mish()]
        layers.append(nn.Linear(h, d))
        
        self.network = nn.Sequential(*layers)
        
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                 nn.init.xavier_uniform_(module.weight,gain = nn.init.calculate_gain("relu"))
                
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
        
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                 nn.init.xavier_uniform_(module.weight,gain = nn.init.calculate_gain("relu"))
        
    def forward(self, a):
        return self.network(a).squeeze()
    



