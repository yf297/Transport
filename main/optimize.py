import torch
import gpytorch

def gp(data, num_epochs=75):
    gp = data.gp
    T = data.T
    
    Z = [z[data.indices] for z in data.Z]
    mean = torch.cat(Z).reshape(-1).mean()
    std =  torch.cat(Z).reshape(-1).std()
    Z0 =  (Z[0] - mean)/std
    data.output_std = std
    
    XY = data.XY[data.indices,:]
    mean = XY.mean(dim=-2, keepdim=True)
    std = XY.view(-1).std() + 1e-6
    XY = (XY - mean) / std
    data.input_std = std

    gp.train()
    gp.likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': gp.kernel.parameters(), 'lr': 0.1},
        {'params': gp.mean.parameters(), 'lr': 0.01},
        {'params': gp.likelihood.parameters(), 'lr': 0.1}
    ])
    points = torch.cat([T[0].repeat(XY.shape[0], 1), XY], dim=-1)

    if torch.cuda.is_available():
        gp = gp.cuda()
        Z0 = Z0.cuda()
        points = points.cuda()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    with torch.no_grad():
        gp.set_train_data(points, Z0, strict=False)
        
    with gpytorch.settings.fast_computations(log_prob=False):
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()

            prior = gp(points)
            ll = -mll(prior, Z0)
            loss = ll

            loss.backward()
            optimizer.step()

            if epoch % 25 == 0:
                print(f'Epoch: {epoch} - Likelihood: {ll.item():.3f}')

    if torch.cuda.is_available():
        data.gp = gp.cpu()
        
        
        
  
def fl_vecchia(data, num_epochs=100):
    gp = data.gp
    flow = data.flow
    T = data.T

    Z = [z[data.indices] for z in data.Z]
    mean = torch.cat(Z).reshape(-1).mean()
    std =  torch.cat(Z).reshape(-1).std()
    Z = [(z - mean)/std for z in Z]
    
    XY = data.XY[data.indices,:]
    mean = XY.mean(dim=-2, keepdim=True)
    std = XY.view(-1).std() + 1e-6
    XY = (XY - mean) / std
        
    
    if torch.cuda.is_available():
        gp = gp.cuda()
        
        
    optimizer = torch.optim.AdamW([
        {'params': flow.parameters(), 'lr':  0.001, "weight_decay": 0.1},
        {'params': gp.kernel.base_kernel.parameters(), 'lr': 0.1},
        ])
        
    fixed1 = gp.kernel.base_kernel.lengthscale[0,1].item()
    fixed2 = gp.kernel.base_kernel.lengthscale[0,2].item()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    
    with gpytorch.settings.detach_test_caches(state=False):
        with gpytorch.settings.fast_computations(log_prob=False, 
                                                covar_root_decomposition=False, 
                                                solves=False):
            
            for epoch in range(1, num_epochs + 1):
                optimizer.zero_grad()
                points = [torch.cat(
                        [t.repeat(XY.shape[0],1), flow(t,XY)], dim = -1)
                        for t in T]
                
                loss = 0
                for i in range(1,data.n):
                    j = max(0, i-2)
                    with torch.no_grad():
                        points0 = torch.cat(points[j:i]).cuda()
                        Z0 = torch.cat(Z[j:i]).reshape(-1).cuda()
                        gp.set_train_data(points0, Z0, strict=False)                    
                        gp.eval()
                        gp.likelihood.eval()
                    
                    points1 = points[i].cuda()
                    Z1 = Z[i].cuda()
                    posterior = gp(points1)
                    loss += -mll(posterior,Z1)/(data.n-1)
                    
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    gp.train()
                    gp.likelihood.train()
                    lengthscale = gp.kernel.base_kernel.lengthscale
                    lengthscale[0,1] = fixed1
                    lengthscale[0,2] = fixed2
                    gp.kernel.base_kernel.lengthscale = lengthscale

                if epoch % 10 == 0:
                    print(f'Epoch: {epoch} - Likelihood: {loss.item():.3f}')
                    
    if torch.cuda.is_available():
        data.gp = gp.cpu()
        data.flow = flow.cpu()