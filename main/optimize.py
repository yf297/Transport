import torch
import gpytorch
from . import model
import gc

from main import ode, LBFGS, neighbors, net
from gpytorch.distributions import MultivariateNormal
import random 

    
    
def fit(data, num_epochs=100):
    flow = net.Flow(L = 4)
    spaceTimeKernel = model.SpaceTimeKernel(l0 = 5, l1 = 0.1, l2 = 0.1).kernel
    gpFlow = model.GPFlow(spaceTimeKernel, flow)
    gpFlow.eval()

    data.device = torch.device('cuda:0')
    gpFlow = gpFlow.to(data.device)

    precomputed_data = []
    for i in range(1, data.n):
        for cell in data.cells:
            TXY0, Z0 = data.conditining(cell, i)
            TXY1, Z1 = data.prediction(cell, i)
            precomputed_data.append((TXY0, Z0, TXY1, Z1))


    optimizer = torch.optim.Adam([
            {'params': gpFlow.flow.parameters(), 'lr':  0.01},
            {'params': gpFlow.kernel.parameters(), 'lr': 0.1},
            {'params': gpFlow.likelihood.parameters(), 'lr': 0.1},
            ])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gpFlow.likelihood, gpFlow)
    
    with gpytorch.settings.detach_test_caches(state=False),\
        gpytorch.settings.cholesky_max_tries(7),\
        gpytorch.settings.fast_computations(log_prob=False, 
                                    covar_root_decomposition=True, 
                                    solves=False):
            

        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()
            gpFlow.train()
            gpFlow.likelihood.train()
            
            loss = 0.0
                
            for TXY0, Z0, TXY1, Z1 in precomputed_data:
                indices0 = torch.randperm(TXY0.shape[0])[:5000]
                indices1 = torch.randperm(TXY1.shape[0])[:5000]
                
                TXY0, Z0 = TXY0[indices0,:].to(data.device), Z0[indices0].to(data.device)
                TXY1, Z1 = TXY1[indices1,:].to(data.device), Z1[indices1].to(data.device)
                
                gpFlow.set_train_data(TXY0, Z0, strict=False)
                gpFlow.eval()
                gpFlow.likelihood.eval()
                
                output = gpFlow(TXY1)
                loss += -mll(output, Z1) / len(precomputed_data)
                #loss.backward()
                del TXY0, Z0, TXY1, Z1, output
                torch.cuda.empty_cache()
                
            loss.backward()
            optimizer.step()
            
            print(f"Epoch: {epoch} - Likelihood: {loss.item():.3f}")
            del loss
    
    data.flow = gpFlow.flow.cpu()

            
