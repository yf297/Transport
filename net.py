import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize(model):
    for weight, bias in zip(model.hidden_weights, model.hidden_biases):
        nn.init.kaiming_uniform_(weight, nonlinearity="relu")
        nn.init.constant_(bias,0)
    
    nn.init.kaiming_uniform_(model.output_weight, nonlinearity="relu")
    nn.init.constant_(model.output_bias,0)

class Flow(nn.Module):
    def __init__(self, L=1, h=48, d=2):
        super(Flow, self).__init__()
                
        self.hidden_weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()
        
        weight = nn.Parameter(torch.Tensor(h, d))
        self.hidden_weights.append(weight)
        
        bias = nn.Parameter(torch.Tensor(h))
        self.hidden_biases.append(bias)
        
        for _ in range(L-1):
            weight = nn.Parameter(torch.Tensor(h, h))
            self.hidden_weights.append(weight)
            
            bias = nn.Parameter(torch.Tensor(h))
            self.hidden_biases.append(bias)
        
        self.output_weight = nn.Parameter(torch.Tensor(d, h))
        self.output_bias = nn.Parameter(torch.Tensor(d))
        
    def forward(self, tx):
        t = tx[0:1]
        x = tx[1:]
        out = x
        
        weight = self.hidden_weights[0]
        bias = self.hidden_biases[0]
        out = F.gelu(F.linear(out, weight) + t * bias)

        for weight, bias in zip(self.hidden_weights[1:], self.hidden_biases[1:]):
            out = F.gelu(F.linear(out, weight) + bias)

        weight = self.output_weight
        bias = self.output_bias
        out = F.linear(out,weight) + bias

        return x + t * out


class Vel(nn.Module):
    def __init__(self, L=1, h=48, d=2):
        super(Vel, self).__init__()
                
        self.hidden_weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()
        
        weight = nn.Parameter(torch.Tensor(h, d))
        self.hidden_weights.append(weight)
        
        bias = nn.Parameter(torch.Tensor(h))
        self.hidden_biases.append(bias)
        
        for _ in range(L-1):
            weight = nn.Parameter(torch.Tensor(h, h))
            self.hidden_weights.append(weight)
            
            bias = nn.Parameter(torch.Tensor(h))
            self.hidden_biases.append(bias)
        
        self.output_weight = nn.Parameter(torch.Tensor(d, h))
        self.output_bias = nn.Parameter(torch.Tensor(d))
        
        
    def forward(self, tx):
        
        t = tx[0:1]
        x = tx[1:]
        out = x
        
        weight = self.hidden_weights[0]
        bias = self.hidden_biases[0]
        
        out = F.gelu(F.linear(out, weight) + t * bias)

        for weight, bias in zip(self.hidden_weights[1:], self.hidden_biases[1:]):
            out = F.gelu(F.linear(out, weight) + bias)

        weight = self.output_weight
        bias = self.output_bias
        out = F.linear(out,weight) + bias
        return out



class Mean(nn.Module):
    def __init__(self, L=1, h=48, d=2):
        super(Mean, self).__init__()
                
        self.hidden_weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()
        
        weight = nn.Parameter(torch.Tensor(h, d))
        self.hidden_weights.append(weight)
        
        bias = nn.Parameter(torch.Tensor(h))
        self.hidden_biases.append(bias)
        
        for _ in range(L-1):
            weight = nn.Parameter(torch.Tensor(h, h))
            self.hidden_weights.append(weight)
            
            bias = nn.Parameter(torch.Tensor(h))
            self.hidden_biases.append(bias)
        
        self.output_weight = nn.Parameter(torch.Tensor(1, h))
        self.output_bias = nn.Parameter(torch.Tensor(1))
        
        
    def forward(self, x):
        out = x
        
        weight = self.hidden_weights[0]
        bias = self.hidden_biases[0]
        out = F.gelu(F.linear(out, weight) + bias)

        for weight, bias in zip(self.hidden_weights[1:], self.hidden_biases[1:]):
            out = F.gelu(F.linear(out, weight) + bias)

        weight = self.output_weight
        bias = self.output_bias
        out = F.linear(out,weight) + bias
        return out
