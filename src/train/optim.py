import gpytorch.constraints
import torch
import gpytorch.models
import train.vecchia 
import gc
import random

def mle(
    T: torch.Tensor,
    XY: torch.Tensor,
    Z: torch.Tensor,
    gp: gpytorch.models.ExactGP,
    epochs: int,
    nn: int, 
    k: int,
    size: int
    )-> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gp.to(device)
    gp.flow.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    vecchia_blocks = train.vecchia.VecchiaBlocks(T, XY, Z)

    optimizer = torch.optim.Adam([
                            {"params": gp.flow.parameters(), "lr": 0.01},
                            {"params": gp.covar_module.parameters(), "lr": 0.01},
                            {"params": gp.mean_module.parameters(), "lr": 0.01},
                            {"params": gp.likelihood.parameters(), "lr":0.1}
                            ])
    print("fitting flow")
    with gpytorch.settings.detach_test_caches(state=False),\
        gpytorch.settings.fast_computations(log_prob=False, 
                                    covar_root_decomposition=False, 
                                    solves=False):
        
        for epoch in range(1, epochs + 1):
            def compute_vecchia_ll():
                gp.train()
                gp.likelihood.train()
                optimizer.zero_grad()
                idx = torch.randperm(XY.size(0))[:size]                
                ll = 0.0
                
                i_sub = 1 + torch.randperm(T.size(0)-1)[:k]
                for i in i_sub:             
                    TXY_pred, Z_pred = vecchia_blocks.prediction(i, idx)
                    TXY_pred, Z_pred = TXY_pred.to(device), Z_pred.to(device)
                
                    TXY_cond, Z_cond = vecchia_blocks.conditioning(i, idx, nn)
                    TXY_cond, Z_cond = TXY_cond.to(device), Z_cond.to(device)
                
                    gp.set_train_data(
                        inputs=TXY_cond,
                        targets=Z_cond,
                        strict=False)
                    gp.eval()
                    gp.likelihood.eval()
                    output = gp(TXY_pred)
                    ll += -mll(output, Z_pred)
                return ll/k
            

            ll = compute_vecchia_ll()
            ll.backward()
            optimizer.step()
            
            if epoch % 1 == 0:
                gp.flow.inspect_weights()
                ls = gp.covar_module.base_kernel.lengthscale.view(-1).tolist()
                ls_str = ", ".join(f"{v:.2f}" for v in ls)
                print(f"Epoch {epoch}/{epochs} — Avg NLL: {ll.item():.4f} — lengthscales: {ls_str}")
    
    gp.flow.eval()
    gp.to("cpu")

