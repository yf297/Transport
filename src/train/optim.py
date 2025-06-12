import gpytorch
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import math


def mse(TXY, UV, velocity, batch_size, epochs):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_iters = math.ceil(TXY.size(0) / batch_size) * epochs
    
    mse_loss = torch.nn.MSELoss()
    optim_vel = torch.optim.Adam([{"params": velocity.parameters(), "lr": 0.001}])
    
    velocity.to(device)
    velocity.train()
    
    for iter in range(1, total_iters + 1):
        optim_vel.zero_grad()

        idx = torch.randperm(TXY.size(0))[:batch_size]
        TXY_sub = TXY[idx,:].to(device)
        input = velocity(TXY_sub)
        target = UV[idx,:].to(device)
        loss = mse_loss(input, target) 
        loss.backward()
        optim_vel.step()
        
        if iter % 1 == 0:  
            print(f"Iter {iter}/{total_iters} — loss: {loss.item():.4f}")
    
    velocity.eval()
    velocity.to("cpu")



def mle(TXY, Z, gp, batch_size, epochs):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_iters = math.ceil(TXY.size(0) / batch_size) * epochs
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    optim_vel = torch.optim.AdamW([
        {"params": gp.flow.velocity.parameters(), "lr":0.01}
        ])
    optim_gp = torch.optim.Adam([
        {"params": gp.mean.parameters(), "lr": 0.01},
        {"params": gp.kernel.kernel.parameters(), "lr": 0.01},
        {"params": gp.likelihood.parameters(), "lr": 0.01}
    ])

    gp.to(device)
    gp.train()
    
    with gpytorch.settings.fast_computations(log_prob=False,
                                             covar_root_decomposition=False,
                                             solves=False):
        for iter in range(1, total_iters + 1):
            optim_gp.zero_grad()
            optim_vel.zero_grad()
            
            idx, _ = torch.sort(torch.randperm(TXY.size(0))[:batch_size])
            TXY_sub = TXY[idx,:].to(device)
            Z_sub = Z[idx].to(device)

            gp.set_train_data(inputs=TXY_sub, targets=Z_sub, strict=False)
            output = gp(TXY_sub)
            ll = -mll(output, Z_sub)

            loss = ll
            loss.backward()
            
            optim_vel.step()
            optim_gp.step()
        
            if iter % 1 == 0:
                noise_var = gp.likelihood.noise_covar.noise.item()

                print(
                    f"Iter {iter}/{total_iters} — "
                    f"Likelihood: {ll.item():.4f} — "
                    f"noise_var: {noise_var:.4f}"
                )
    

    gp.eval()
    gp.likelihood.eval()
    gp.to("cpu")