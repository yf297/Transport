import torch
import penalty
import gpytorch
import gc 

def flow(model, num_epochs=75):
    
    if torch.cuda.is_available():
        model.TXY = model.TXY.cuda()
        model.Z = model.Z.cuda()
        model.gp = model.gp.cuda()
        model.likelihood = model.likelilhood.cuda()

    optimizer_mean = torch.optim.Adam([
        {'params': model.gp.Mean.parameters(), 'lr': 0.001}
    ])
    
    optimizer_flow = torch.optim.Adam([
        {'params': model.flow.parameters(), 'lr': 0.001}
    ])
   
    optimizer_cov = torch.optim.Adam([
        {'params': model.gp.Kernel.parameters(), 'lr': 0.1},
        {'params': model.gp.likelihood.parameters(), 'lr': 0.1}
    ])
    
    scheduler_mean = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_mean, 
        mode='min', 
        factor=0.1, 
        patience=1, 
        min_lr=1e-5
    )
    
    scheduler_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_flow, 
        mode='min', 
        factor=0.1, 
        patience=1, 
        min_lr=1e-5
    )
    
    scheduler_cov = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cov, 
        mode='min', 
        factor=0.1, 
        patience=1, 
        min_lr=1e-5
    )

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model.gp)


    for epoch in range(1, num_epochs+1):
        
        optimizer_mean.zero_grad()
        optimizer_cov.zero_grad()
        optimizer_flow.zero_grad()
        
        output = model.gp(model.TXY)
        ll = -mll(output, model.Z) 
                        
        loss = ll
        loss.backward()

        optimizer_mean.step()
        optimizer_cov.step()
        optimizer_flow.step()
        
        scheduler_mean.step(ll)
        scheduler_flow.step(ll)
        scheduler_cov.step(ll)
        
        current_lrs = [round(group['lr'], 6) for group in optimizer_flow.param_groups]

        if epoch % 5 == 0:
            print(f"Epoch: {epoch} - Likelihood: {ll.item():.3f} - Learning Rates: {current_lrs}")

        if 1e-5 in current_lrs:
            if torch.cuda.is_available():
                model.TXY = model.TXY.cpu()
                model.Z = model.Z.cpu()
                model.gp = model.gp.cpu()
                model.likelihood = model.likelilhood.cpu()
            break
        
    if torch.cuda.is_available():   
        model.TXY = model.TXY.cpu()
        model.Z = model.Z.cpu()
        model.gp = model.gp.cpu()
        model.likelihood = model.likelilhood.cpu()

        
