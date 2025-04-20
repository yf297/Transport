import torch
import gpytorch.models
import train.vecchia 
import gc
def mle(
    T: torch.Tensor,
    XY: torch.Tensor,
    Z: torch.Tensor,
    gp: gpytorch.models.ExactGP,
    epochs: int,
    sample_size: int,
    nn: int
)-> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gp.to(device)
    gp.flow.to(device)
    gp.flow.train()
    gp.train()
    gp.likelihood.train()
    
    optimizer = torch.optim.Adam([
        {"params": gp.flow.parameters(),           "lr": 0.01},
        {"params": gp.covar_module.parameters(),   "lr": 0.1},
        {"params": gp.mean_module.parameters(),    "lr": 0.1},
        {"params": gp.likelihood.parameters(),     "lr": 0.1},
    ])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    vecchia_blocks = train.vecchia.VecchiaBlocks(T, XY, Z)
    T_steps = T.size(0)

    with gpytorch.settings.detach_test_caches(state=False),\
        gpytorch.settings.fast_computations(log_prob=False, 
                                    covar_root_decomposition=False, 
                                    solves=False):
            
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            gp.train()
            gp.likelihood.train()
            idx = torch.randperm(XY.size(0))[:sample_size]
            
            TXY_pred, Z_pred = vecchia_blocks.prediction(i=0, idx=idx)  
            TXY_pred, Z_pred = TXY_pred.to(device), Z_pred.to(device)

            gp.set_train_data(TXY_pred, Z_pred, strict=False)
            
            output = gp(TXY_pred)
            ll = -mll(output, Z_pred) /  T_steps

            for i in range(1,T_steps):
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
                ll += -mll(output, Z_pred) / T_steps

            ll.backward()
            optimizer.step()
            with torch.no_grad():
                gp.flow.project_weights()
                
            print(f"Epoch {epoch}/{epochs} — Avg NLL: {(ll.item()):.4f}")

    gc.collect()
    torch.cuda.empty_cache()
    gp.flow.eval()
    gp.eval()
    gp.likelihood.eval()
    gp.to("cpu")
