import gpytorch.constraints
import torch
import gpytorch.models
import train.vecchia 
import gc
from .LBFGS import FullBatchLBFGS

def mle0(
    T: torch.Tensor,
    XY: torch.Tensor,
    Z: torch.Tensor,
    gp: gpytorch.models.ExactGP,
    epochs: int,
    sample_size: int
)-> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gp.to(device)
    gp.flow.to(device)
    gp.flow.train()
    gp.train()
    gp.likelihood.train()

    vecchia_blocks = train.vecchia.VecchiaBlocks(T, XY, Z)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    optimizer = FullBatchLBFGS(gp.parameters(), lr=0.1)

    def closure():
        optimizer.zero_grad()
        gp.train(); gp.likelihood.train()

        idx = torch.randperm(XY.size(0))[:sample_size]
        TXY_pred, Z_pred = vecchia_blocks.prediction(i=0, idx=idx, exact=False)
        TXY_pred, Z_pred = TXY_pred.to(device), Z_pred.to(device)

        gp.set_train_data(TXY_pred, Z_pred, strict=False)
        out = gp(TXY_pred)
        return -mll(out, Z_pred)

    loss = closure()
    loss.backward()

    with gpytorch.settings.detach_test_caches(state=False), \
        gpytorch.settings.fast_computations(log_prob=False,
                                            covar_root_decomposition=False,
                                            solves=False):

        for epoch in range(1, epochs+1):
            options = {
                "closure": closure,
                "current_loss": loss,       
                "max_ls": 10,               
                "tolerance_grad": 1e-10,
                "tolerance_change": 1e-10
            }
            loss, *_ , fail = optimizer.step(options)

            ls = gp.covar_module.base_kernel.lengthscale.view(-1).tolist()
            ls_str = ", ".join(f"{v:.2f}" for v in ls)
            print(
                f"Epoch {epoch}/{epochs} — "
                f"NLL: {loss.item():.4f} — "
                f"lengthscales: {ls_str}"
            )
            if fail:
                print("Convergence reached!")
                break

    gp.flow.eval()
    gp.eval()
    gp.likelihood.eval()
    gp.to("cpu")



def mle(
    T: torch.Tensor,
    XY: torch.Tensor,
    Z: torch.Tensor,
    gp: gpytorch.models.ExactGP,
    epochs: int,
    sample_size: int,
    nn: int,
    exact: bool
)-> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gp.to(device)
    gp.flow.to(device)
    gp.flow.train()
    gp.train()
    gp.likelihood.train()
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    vecchia_blocks = train.vecchia.VecchiaBlocks(T, XY, Z)
    if exact:
        steps = 1
    else:
        steps = T.size(0)

    optimizer0 = torch.optim.AdamW([
                            {"params": gp.flow.parameters(), "lr": 0.005},
                            ])
    
    optimizer1 = torch.optim.Adam([
                            {"params": gp.covar_module.parameters(), "lr": 0.01},
                            {"params": gp.mean_module.parameters(), "lr": 0.01},
                            {"params": gp.likelihood.parameters(),"lr": 0.1},
                            ])
    
    with gpytorch.settings.detach_test_caches(state=False),\
        gpytorch.settings.fast_computations(log_prob=False, 
                                    covar_root_decomposition=False, 
                                    solves=False):
        
        for epoch in range(1, epochs + 1):
            def compute_ll(idx):
                gp.train()
                gp.likelihood.train()
                idx = torch.randperm(XY.size(0))[:sample_size]
            
                TXY_pred, Z_pred = vecchia_blocks.prediction(i=0, idx=idx, exact=exact)
                TXY_pred, Z_pred = TXY_pred.to(device), Z_pred.to(device)
            
                gp.set_train_data(TXY_pred, Z_pred, strict=False)
                output = gp(TXY_pred)
                ll = -mll(output, Z_pred) /  steps

                for i in range(1,steps):
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
                    ll += -mll(output, Z_pred) / steps
                return ll
            
            idx = torch.randperm(XY.size(0))[:sample_size]
            optimizer0.zero_grad()
            ll = compute_ll(idx)
            ll.backward()
            optimizer0.step()

            optimizer1.zero_grad()
            ll = compute_ll(idx)
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
