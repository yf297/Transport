import gpytorch.constraints
import torch
import gpytorch.models
import train.vecchia 
import gc
import random


def mle(
    T,
    XY, 
    Z,
    gp,
    epochs,
    size, 
    warmup_epochs = 5
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T, XY, Z = T.to(device), XY.to(device), Z.to(device)
    gp.to(device)
    gp.flow.to(device)
    print("moved to gpu")
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    optim_gp = torch.optim.Adam([
        {"params": gp.mean_module.parameters(),  "lr": 0.1},
        {"params": gp.covar_module.parameters(), "lr": 0.1},
    ])

    optim_flow = torch.optim.AdamW([
        {"params": gp.flow.parameters(),  "lr": 0.005, "weight_decay":0.01},
        {"params": gp.likelihood.parameters(), "lr": 0.1},
    ])
    
    gp.train()
    gp.likelihood.train()
    gp.flow.train()
    
    with gpytorch.settings.fast_computations(log_prob=False,
                                             covar_root_decomposition=False,
                                             solves=False):
        for epoch in range(1, epochs + 1):
            optim_gp.zero_grad()
            optim_flow.zero_grad()
            
            idx = torch.randperm(Z.size(0))[:size]
            T_train = T[idx]
            XY_train = XY[idx]
            Z_train = Z[idx]
            gp.set_train_data(inputs=(T_train, XY_train), targets=Z_train, strict=False)
            
            if epoch == warmup_epochs + 1:
                optim_flow = torch.optim.AdamW([
                    {"params": gp.flow.parameters(),  "lr": 0.01, "weight_decay": 0.01},
                    {"params": gp.likelihood.parameters(), "lr": 0.1},
                ])

            
            output = gp(T_train, XY_train)
            ll = -mll(output, Z_train)
            
            ll.backward()
            optim_flow.step()
            if epoch > warmup_epochs:
                optim_gp.step()

            gp.flow.project_weights()
            gp.flow.inspect_weights()
            ls = gp.covar_module.base_kernel.lengthscale.view(-1).tolist()
            ls_str = ", ".join(f"{v:.2f}" for v in ls)
            noise_var = gp.likelihood.noise_covar.noise.item()
            outputscale = gp.covar_module.outputscale.item()
            print(
                f"Epoch {epoch}/{epochs} — "
                f"Avg likelihood: {ll.item():.4f} — "
                f"outputscale: {outputscale:.4f} — "
                f"lengthscales: {ls_str} — "
                f"noise_var: {noise_var:.4f}"
            )

    gp.eval()
    gp.likelihood.eval()
    gp.flow.eval()
    gp.flow.to("cpu")
    gp.to("cpu")
