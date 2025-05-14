import gpytorch
import torch
import torch.nn.functional as F
import random

def mle(
    T,
    XY, 
    Z,
    gp,
    flow,
    epochs,
    size
):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gp.to(device)
    flow.to(device)
    print("moved to gpu")
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    optim_gp = torch.optim.Adam([
        {"params": gp.mean_module.parameters(),  "lr": 0.1},
        {"params": gp.covar_module.parameters(), "lr": 0.1},
        {"params": gp.likelihood.parameters(),   "lr": 0.1}
    ])
    
    optim_flow = torch.optim.Adam([
         {"params": flow.warp.parameters(),  "lr": 0.01}
    ])
    
    gp.train()
    gp.likelihood.train()
    flow.train()
    
    
    print("sampling from", XY.size(0), "points")
    with gpytorch.settings.fast_computations(log_prob=False,
                                             covar_root_decomposition=False,
                                             solves=False):
        for epoch in range(1, epochs + 1):
            optim_gp.zero_grad()
            optim_flow.zero_grad()
            
            idx = torch.randperm(XY.size(0))[:size]
            
            T_train = T.to(device)
            XY_train = XY[idx,:].to(device)
            Z_train = Z[:,idx].reshape(-1).to(device)
            A_train = flow(T_train, XY_train)
            TA_train = torch.cat([T_train.repeat_interleave(XY_train.size(0)).unsqueeze(1), 
                                  A_train], dim = -1)

            gp.set_train_data(inputs=TA_train, targets=Z_train, strict=False)

            output = gp(TA_train)
            ll = -mll(output, Z_train)
            loss = ll 
            
            loss.backward()
            optim_flow.step()
            optim_gp.step()
            with torch.no_grad():
                flow.project_all_weight_norms()
            del T_train, XY_train, Z_train
            torch.cuda.empty_cache()

            if epoch % 1 == 0:
                flow.check_weight_norm_products()
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
    flow.eval()
    flow.to("cpu")
    gp.to("cpu")