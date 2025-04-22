import gpytorch.constraints
import torch
import gpytorch.models
import train.vecchia 
import gc
from .LBFGS import FullBatchLBFGS
import random

def mle(
    T: torch.Tensor,
    XY: torch.Tensor,
    Z: torch.Tensor,
    gp: gpytorch.models.ExactGP,
    epochs: int,
    nn: int, 
    stride: int = 4
)-> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gp.to(device)
    gp.flow.to(device)
    gp.flow.train()
    gp.train()
    gp.likelihood.train()
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    vecchia_blocks = train.vecchia.VecchiaBlocks(T, XY, Z, stride)


    optimizer1 = torch.optim.Adam([
                            {"params": gp.flow.parameters(), "lr": 0.003},
                            {"params": gp.covar_module.parameters(), "lr": 0.01},
                            {"params": gp.mean_module.parameters(), "lr": 0.01},
                            {"params": gp.likelihood.parameters(),"lr": 0.1},
                            ])
    
    with gpytorch.settings.detach_test_caches(state=False),\
        gpytorch.settings.fast_computations(log_prob=False, 
                                    covar_root_decomposition=False, 
                                    solves=False):
        
        for epoch in range(1, epochs + 1):
            def compute_ll():
                gp.train()
                gp.likelihood.train()
                idx = random.randint(0, (stride**2)-1)
                TXY_pred, Z_pred = vecchia_blocks.prediction(i=0, idx=idx)
                TXY_pred, Z_pred = TXY_pred.to(device), Z_pred.to(device)
            
                gp.set_train_data(TXY_pred, Z_pred, strict=False)
                output = gp(TXY_pred)
                ll = -mll(output, Z_pred) /  T.size(0)

                for i in range(1, T.size(0)):
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
                    ll += -mll(output, Z_pred) / T.size(0)
                return ll
            

            optimizer1.zero_grad()
            ll = compute_ll()
            ll.backward()
            optimizer1.step()
            
            with torch.no_grad():
                #gp.flow.project_weights()
                gp.flow.inspect_weights()
            
            ls = gp.covar_module.base_kernel.lengthscale.view(-1).tolist()
            ls_str = ", ".join(f"{v:.2f}" for v in ls)
            print(f"Epoch {epoch}/{epochs} — Avg NLL: {ll.item():.4f} — lengthscales: {ls_str}")
    
                
    gp.flow.eval()
    gp.eval()
    gp.likelihood.eval()
    gp.to("cpu")
    del vecchia_blocks, mll, optimizer1
    gc.collect()
    torch.cuda.empty_cache()
