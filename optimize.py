import torch
import gpytorch
from likelihoods import ll
import gc
import penalty

import torch
import gpytorch
import gc

def initial(model, num_epochs=75):
    
    optimizer = torch.optim.AdamW([
        {'params': model.gp.Mean.parameters()},
        {'params': model.flow.parameters()},
        {'params': model.vel.parameters()},
        {'params': model.gp.Kernel.parameters(), 'lr': 0.1, 'weight_decay':0},
        {'params': model.gp.likelihood.parameters(),'lr': 0.1, 'weight_decay':0},
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    if torch.cuda.is_available():   
        model.gp = model.gp.cuda()
        model.likelihood = model.likelihood.cuda()
        model.flow = model.flow.cuda()
        model.vel = model.vel.cuda()
        model.TXY = model.TXY.cuda()
        model.Z = model.Z.cuda()
        
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model.gp)
    model.gp.train()
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        output = model.gp(model.TXY)
        ll = -mll(output, model.Z)
        
        loss = ll + 0.001 * penalty.PDE(model.TXY, model.flow, model.vel)
        loss.backward()
        optimizer.step()
        
        #scheduler.step()
        
        if epoch % 1 == 0:
            print(f'Epoch: {epoch} - Likelihood: {loss.item():.2f}')
    
    if torch.cuda.is_available():   
        model.gp = model.gp.cpu()
        model.likelihood = model.likelihood.cpu()
        model.flow = model.flow.cpu()
        model.vel = model.vel.cpu()
        model.TXY = model.TXY.cpu()
        model.Z = model.Z.cpu()
        gc.collect()
        torch.cuda.empty_cache()



