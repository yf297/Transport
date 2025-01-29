import torch.nn as nn

class Flow(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Flow, self).__init__()
        
        layers = [nn.Linear(d+1, h) , nn.GELU()]   
        for _ in range(L - 1):
            layers += [nn.Linear(h, h) , nn.GELU()]
        layers.append( nn.Linear(h, d) )
        self.network = nn.Sequential(*layers)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
                
    def forward(self, txy):
        xy = txy[...,1:]
        t = txy[...,0:1]
        return xy + t*self.network(txy)
    

class Mean(nn.Module):
    def __init__(self, d=2, L=1, h=32):
        super(Mean, self).__init__()
        layers = [nn.Linear(d, h) , nn.GELU()]   
        for _ in range(L - 1):
            layers += [nn.Linear(h, h) , nn.GELU()]
        layers.append( nn.Linear(h, 1) )
        self.network = nn.Sequential(*layers)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, a):
        return self.network(a).squeeze()


