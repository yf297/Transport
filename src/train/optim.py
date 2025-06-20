import gpytorch
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import math
from torch.utils.data import TensorDataset, DataLoader, BatchSampler
import tqdm


def mle(TXY, Z, gp, batch_size, epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gp.to(device)
    gp.train()

    optim_vel = torch.optim.Adam([
        {"params": gp.flow.velocity.parameters(), "lr":0.01},
    ])
    optim_gp = torch.optim.Adam([
        {"params": gp.mean.parameters(), "lr": 0.01},
        {"params": gp.kernel.kernel.parameters(), "lr": 0.1},
        {"params": gp.likelihood.parameters(), "lr": 0.1}
    ])

    train_dataset = TensorDataset(TXY, Z)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    sched_vel = torch.optim.lr_scheduler.OneCycleLR(optim_vel, max_lr=0.01,
                                                    div_factor=10,
                                                    final_div_factor = 1000,  
                                                    epochs=epochs,
                                                    steps_per_epoch=len(train_loader))
    

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    print("Training GP with MLE...")
    with gpytorch.settings.fast_computations(log_prob=False,
                                             covar_root_decomposition=False,
                                             solves=False):
        for epoch in range(epochs):
            avg_loss = 0
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                optim_gp.zero_grad()
                optim_vel.zero_grad()
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                gp.set_train_data(inputs=x_batch, targets=y_batch, strict=False)
                output = gp(x_batch)
                ll =  mll(output, y_batch)
                loss = -1*ll
                loss.backward()
                avg_loss += loss.item()/len(minibatch_iter)
                optim_vel.step()
                optim_gp.step()
                sched_vel.step()
                minibatch_iter.set_postfix(loss=loss.item())

            noise_var = gp.likelihood.noise_covar.noise.item()
            lengthscale = gp.kernel.kernel.base_kernel.lengthscale.detach().cpu().numpy().squeeze()
            print(
                f"Epoch {epoch+1}/{epochs} — "
                f"loss: {avg_loss:.4f} — "
                f"noise_var: {noise_var:.4f} — "
                f"lengthscale: {lengthscale}"
            )
            

    gp.eval()
    gp.to("cpu")



def mse_vector(TXY, UV, velocity, batch_size, epochs):
    
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

    velocity.eval()
    velocity.to("cpu")
