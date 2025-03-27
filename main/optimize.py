import torch
import gpytorch
from . import model
import gc

from main import ode

def fit(data, num_epochs=100, fix_t = False):
    
    device = torch.device('cuda:0')
    T = data.T_normalized.contiguous()
    XY = data.XY_normalized.contiguous()
    points = torch.cat([T[0].repeat(XY.shape[0], 1), XY], dim=-1).contiguous()
    Z0 = data.Z0_normalized.contiguous()
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel  = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.keops.MaternKernel(nu=5/2,
                                          ard_num_dims=3))
    


    kernel.base_kernel.initialize(lengthscale=torch.tensor([1/data.T[-1] , 0.1, 0.1]))
    gp = model.GP(kernel, likelihood, points, Z0)
    
    points = points.to(device)
    Z0 = Z0.to(device)
    likelihood = likelihood.to(device)
    gp = gp.to(device)
    
    gp.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    
    with gpytorch.settings.fast_computations(log_prob=False, 
                                        covar_root_decomposition=False, 
                                        solves=False):
        for epoch in range(1, 100 + 1):
            optimizer.zero_grad()
            prior = gp(points)
            ll = -mll(prior, Z0)
            loss = ll
            loss.backward()
            optimizer.step()

            if epoch % 25 == 0:
                print(f'Epoch: {epoch} - Likelihood: {ll.item():.3f}')
                        

    del Z0, points, prior, loss, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    
    Z = data.Z_normalized
    XY = XY.to(device)
    T = T.to(device)
    Z = [z.to(device) for z in Z]
    #Z = torch.cat(Z).view(-1)
    flow = data.flow.to(device)
    
    print("fitting flow")
    optimizer = torch.optim.Adam([
        {'params': flow.parameters(), 'lr':  0.01},
        {'params': gp.kernel.base_kernel.parameters(), 'lr': 0.01},
        ])
    
    fixed0 = gp.kernel.base_kernel.lengthscale[0,0].item()
    fixed1 = gp.kernel.base_kernel.lengthscale[0,1].item()
    fixed2 = gp.kernel.base_kernel.lengthscale[0,2].item()
    
    gp.train()
    likelihood.train()
    grid_x = torch.linspace(-2,2,30)
    grid_y = torch.linspace(-2,2,30)
    grid_X,grid_Y = torch.meshgrid(grid_x,grid_y,indexing='xy')
    grid = torch.stack([grid_X,grid_Y], dim = -1).reshape(-1,2).to(device)

    
    with gpytorch.settings.detach_test_caches(state=False),\
        gpytorch.settings.fast_computations(log_prob=False, 
                                        covar_root_decomposition=False, 
                                        solves=False):
        for epoch in range(1, num_epochs + 1):

            optimizer.zero_grad()
            points = [torch.cat(
                    [t.repeat(XY.shape[0],1), flow(t,XY)], dim = -1)
                    for t in T]

            ll = 0
            for i in range(1,data.n):
                j = max(0, i-1)
                points0 = torch.cat(points[j:i])
                Z0 = torch.cat(Z[j:i]).view(-1)
                
                with torch.no_grad():
                    gp.set_train_data(points0, Z0, strict=False)
                    gp.eval()
                    likelihood.eval()
                    
                points1 = points[i]
                Z1 = Z[i]
                posterior = gp(points1)
                ll += -mll(posterior,Z1)/(data.n-1) 
                del points0, Z0
            
            vel = ode.Vel_hat(data, scale = False)
            det_pen =  ode.pen_det_mean(T, grid, flow)
            vel_pen =  ode.pen_vel_mean(T, grid, vel, 0.5)
            dt_vel_pen, dx_vel_pen = ode.pen_D_vel_mean(T, grid, vel, 0.1, 0.75)
            loss = ll + 0.5*det_pen +\
                        0.5*vel_pen+\
                        0.5*dt_vel_pen + \
                        0.5*dx_vel_pen
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                gp.train()
                likelihood.train()
                lengthscale = gp.kernel.base_kernel.lengthscale
                lengthscale[0,1] = fixed1
                lengthscale[0,2] = fixed2
                if fix_t:
                    lengthscale[0,0] = fixed0
                gp.kernel.base_kernel.lengthscale = lengthscale
                
            del loss, points
            if epoch % 10 == 0:
                transformed_ls0 = round(gp.kernel.base_kernel.lengthscale[0][0].item() * data.time_std.item(), 2)
                transformed_ls1 = round(gp.kernel.base_kernel.lengthscale[0][1].item() * data.input_std[0][0].item(), 2)
                transformed_ls2 = round(gp.kernel.base_kernel.lengthscale[0][2].item() * data.input_std[0][1].item(), 2)
                print(f"Epoch: {epoch} - Likelihood: {ll.item():.3f}"
                    f" - Lengthscales: [{transformed_ls0}, {transformed_ls1}, {transformed_ls2}]")
        
        
        gp = gp.cpu()
        likelihood = likelihood.cpu()
        data.flow = flow.cpu()
        data.sigma2 = round(gp.kernel.outputscale.item() * data.output_std.item(), 2)
        data.l1     = round(gp.kernel.base_kernel.lengthscale[0][0].item() * data.time_std.item(), 2)
        data.l2     = round(gp.kernel.base_kernel.lengthscale[0][1].item() * data.input_std[0][0].item(), 2)
        data.l3     = round(gp.kernel.base_kernel.lengthscale[0][2].item() * data.input_std[0][1].item(), 2)
        data.tau2   = round(likelihood.noise.item() * data.output_std.item(), 2)
        
