import torch
import penalty
import gpytorch

def flow(spacetime, z, model, num_epochs=75):
    
    model.gp.set_train_data(spacetime, z, strict=False)
    
    
    optimizer_net = torch.optim.Adam([
        {'params': model.flow.parameters(),'lr': 0.01},
        {'params': model.gp.Mean.Mean_W.parameters(),'lr': 0.01}])
   
    optimizer_cov = torch.optim.Adam([            
        {'params': model.gp.Kernel.Kernel_W.parameters(), 'lr': 0.1},
        {'params': model.gp.likelihood.parameters(), 'lr': 0.1}
    ])

    scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_net, 
        mode='min', 
        factor=0.1, 
        patience=2, 
        min_lr=1e-5
    )
    
    scheduler_cov = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_net, 
        mode='min', 
        factor=0.1, 
        patience=2, 
        min_lr=1e-5
    )

    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.gp.likelihood, model.gp)

    for epoch in range(1, num_epochs+1):
        optimizer_net.zero_grad()
        optimizer_cov.zero_grad()
        
        output = model.gp(spacetime)
        ll = -mll(output, z) 
                       
        loss = ll
        loss.backward()

        optimizer_net.step()
        optimizer_cov.step()
        
        scheduler_net.step(ll)
        scheduler_cov.step(ll)
        
        current_lrs = [round(group['lr'], 6) for group in optimizer_net.param_groups]

        if epoch % 5 == 0:
            print(f"Epoch: {epoch} - Likelihood: {ll.item():.3f} - Learning Rates: {current_lrs}")

        if 1e-5 in current_lrs:
            break