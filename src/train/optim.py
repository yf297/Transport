import gpytorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm


def mle(TXY, Z, gp, batch_size, epochs, verbose, cvf, dvf):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gp.to(device)
    gp.train()
            
    train_dataset = TensorDataset(TXY, Z)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    optim_flow = torch.optim.AdamW(gp.flow.parameters(), lr=0.005)
    optim_mean = torch.optim.Adam(gp.mean.parameters(), lr=0.1)
    optim_kernel = torch.optim.Adam(gp.kernel.kernel.parameters(), lr=0.1)
    optim_noise = torch.optim.Adam(gp.likelihood.parameters(), lr=0.1)
    
        
    with gpytorch.settings.fast_computations(log_prob=False,
                                             covar_root_decomposition=False,
                                             solves=False):
        for epoch in range(epochs):
            avg_ll = 0
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                optim_flow.zero_grad()
                optim_mean.zero_grad()
                optim_kernel.zero_grad()  
                optim_noise.zero_grad()
   
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                gp.set_train_data(inputs=x_batch, targets=y_batch, strict=False)
                output = gp(x_batch)
                ll =  -1*mll(output, y_batch) 
                
                loss = ll 
                loss.backward()

                avg_ll += ll.item()/len(minibatch_iter)
                optim_flow.step()
                optim_mean.step()
                optim_kernel.step()
                optim_noise.step()
                minibatch_iter.set_postfix(loss=loss.item())

            mean = gp.mean.constant.item()
            sigma2 = gp.kernel.kernel.outputscale.item()
            l0 = gp.kernel.kernel.base_kernel.lengthscale[0][0].item()
            l1 = gp.kernel.kernel.base_kernel.lengthscale[0][1].item()
            l2 = gp.kernel.kernel.base_kernel.lengthscale[0][2].item()
            tau2 = gp.likelihood.noise.item()

            if verbose:
                if (epoch+1) % 10 == 0 or epoch == epochs - 1 or epoch == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} — "
                        f"rmse: {cvf.RMSE(dvf):.4f} - "
                        f"ll: {avg_ll:.4f} - "
                        f"mean: {mean:.4f} - "
                        f"sigma2: {sigma2:.4f} - "
                        f"tau2: {tau2:.4f} - "
                        f"l0: {l0:.4f} - "
                        f"l1: {l1:.4f} - "
                        f"l2: {l2:.4f}"
                    )

    gp.eval()
    gp.to("cpu")









def pde(TXY, flow, UV, batch_size, epochs):
    n = TXY.shape[0] 
    idx = torch.randperm(TXY.shape[0])[0:n]
    TXY = TXY[idx]
    UV = UV[idx]
    
    optim_flow = torch.optim.AdamW([{"params": flow.parameters(), "lr": 0.04}])
    flow.train()

    l1 = torch.nn.HuberLoss(delta=1)
    zeros = torch.zeros(batch_size, 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    flow.to(device)
    TXY = TXY.to(device)
    UV = UV.to(device)
    zeros = zeros.to(device)

    train_dataset = TensorDataset(TXY, UV)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)    
    
    for epoch in range(epochs):
        avg_loss = 0.0
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        for (x_batch, uv_batch) in minibatch_iter:
            optim_flow.zero_grad()
            
            x_batch = x_batch
            uv_batch = uv_batch

            res = flow.vel_psi(x_batch) - uv_batch
            loss = l1(res, zeros)
            loss.backward(retain_graph=True)
            optim_flow.step()
            
            avg_loss += loss.item() / len(minibatch_iter)
        if (epoch+1) % 10 == 0 or epoch == epochs - 1 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")
    
    flow.to("cpu")
    
