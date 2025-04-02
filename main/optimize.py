import torch
import gpytorch
from . import model
import gc

from main import ode, LBFGS
from gpytorch.distributions import MultivariateNormal
import random 

def vecchia_loss(Z, points, gp, likelihood, mll, i):
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
    del points0, Z0

    return -mll(posterior, Z1)
    
    
def fit(data, indices, num_epochs=100, fix_t = False, num_batches = 1):
    
    device = torch.device('cuda:0')
    T = data.T_normalized.contiguous()
    XY = data.XY_normalized[indices, :].contiguous()
    points = torch.cat([T[0].repeat(XY.shape[0], 1), XY], dim=-1).contiguous()
    Z0 = data.Z0_normalized[indices].contiguous()
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel  = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.keops.MaternKernel(nu=5/2,
                                          ard_num_dims=3))
    
    kernel.base_kernel.initialize(lengthscale=torch.tensor([4, 0.1, 0.1]))
        
    gp = model.GP(kernel, likelihood)
    gp.set_train_data(points, Z0, strict=False)
    
    points = points.to(device)
    Z0 = Z0.to(device)
    likelihood = likelihood.to(device)
    gp = gp.to(device)
    
    gp.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    optimizer = LBFGS.FullBatchLBFGS(gp.parameters(), lr=0.1)    
    def closure():
            optimizer.zero_grad()
            prior = gp(points)
            loss = -mll(prior, Z0)
            return loss
    loss = closure()
    loss.backward()
        
    with gpytorch.settings.fast_computations(log_prob=False, 
                                             covar_root_decomposition=False, 
                                             solves=False):
        for epoch in range(1, 50 + 1):
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)
            
            if epoch % 1 == 0:
                transformed_ls0 = round(gp.kernel.base_kernel.lengthscale[0][0].item() * data.time_normalize.item(), 2)
                transformed_ls1 = round(gp.kernel.base_kernel.lengthscale[0][1].item() * data.input_normalize.item(), 2)
                transformed_ls2 = round(gp.kernel.base_kernel.lengthscale[0][2].item() * data.input_normalize.item(), 2)
                print(f"Epoch: {epoch} - Likelihood: {loss.item():.3f}"
                    f" - Lengthscales: [{transformed_ls0}, {transformed_ls1}, {transformed_ls2}]")
        
            if fail:
                print('Convergence reached!')
                break
    del Z0, points, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    
    Z = data.Z_normalized
    XY = XY.to(device)
    T = T.to(device)
    Z = [z[indices].to(device) for z in Z]
    flow = data.flow.to(device)
    
    sigma2 =  gp.kernel.outputscale.item()
    l0 = gp.kernel.base_kernel.lengthscale[0,0].item()
    l1 = gp.kernel.base_kernel.lengthscale[0,1].item()
    l2 = gp.kernel.base_kernel.lengthscale[0,2].item()
    lengthscale_init = torch.tensor([l0, l1, l2])
    
    eps0 = 2
    eps1 = 1e-5
    eps2 = 1e-5
    if fix_t:
        eps0 = 1e-5
        
    eps = torch.tensor([eps0, eps1, eps2])
    lengthscale_constraint = gpytorch.constraints.Interval(
        lower_bound=lengthscale_init - eps,
        upper_bound=lengthscale_init + eps
    )

    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.keops.MaternKernel(nu=5/2,
                                          ard_num_dims=3,
                                          lengthscale_constraint=lengthscale_constraint))

    kernel.base_kernel.initialize(lengthscale=lengthscale_init)
    kernel.initialize(outputscale=sigma2)
    gp = model.GP(kernel, likelihood)
    gp = gp.to(device)

    grid_x = torch.linspace(-3,3,30)
    grid_y = torch.linspace(-3,3,30)
    grid_X,grid_Y = torch.meshgrid(grid_x,grid_y,indexing='xy')
    
    grid = torch.stack([grid_X,grid_Y], dim = -1).reshape(-1,2).to(device)

    gp.train()
    likelihood.train()
    
    print("fitting flow")
    optimizer = torch.optim.Adam([
        {'params': flow.parameters(), 'lr':  0.01},
        {'params': gp.kernel.base_kernel.parameters(), 'lr': 0.1},
        ])

    with gpytorch.settings.detach_test_caches(state=False),\
         gpytorch.settings.cholesky_max_tries(7),\
         gpytorch.settings.fast_computations(log_prob=False, 
                                        covar_root_decomposition=False, 
                                        solves=False):
        for epoch in range(1, num_epochs + 1):
            gp.train()
            likelihood.train()
            optimizer.zero_grad()
            
            points = [torch.cat(
                    [t.repeat(XY.shape[0],1), flow(t,XY)], dim = -1)
                    for t in T]
            
            
            indices = random.sample(range(1, data.n), num_batches)
            vel = ode.Vel_hat(data, scale = False)
            ll = 0
            det_pen = 0
            vel_pen = 0
            dt_vel_pen = 0
            dx_vel_pen = 0
            for i in indices:
                det_pen +=  ode.det(T[i], grid, flow, threshold = 0.0) / num_batches
                vel_pen +=  ode.vel_norm(T[i], grid, vel, M = 2) / num_batches
                tup = ode.D_vel_norm(T[i], grid, vel, L1 = 0.0, L2 = 1.5)
                dt_vel_pen +=tup[0] / num_batches
                dx_vel_pen +=tup[1] / num_batches
                ll += vecchia_loss(Z, points, gp, likelihood, mll, i) / num_batches
            
            loss = ll + 10*det_pen +\
                        10*vel_pen 
                      #  10*dx_vel_pen + \
                      #  10*dt_vel_pen
                        
            loss.backward()
            optimizer.step()

            if epoch % 25 == 0:
                
                #ll_full = 0
                #for i in range(1, data.n):
                    #ll_full += vecchia_loss(Z, points, gp, likelihood, mll, i)/(data.n - 1)
                
                #print(f"Penalties at Epoch {epoch}:")
                #print(f"  det_pen     : {det_pen.item():.5f}")
                #print(f"  vel_pen     : {vel_pen.item():.5f}")
                #print(f"  dt_vel_pen  : {dt_vel_pen.item():.5f}")
                #print(f"  dx_vel_pen  : {dx_vel_pen.item():.5f}")
                transformed_ls0 = round(gp.kernel.base_kernel.lengthscale[0][0].item() * data.time_normalize.item(), 2)
                transformed_ls1 = round(gp.kernel.base_kernel.lengthscale[0][1].item() * data.input_normalize.item(), 2)
                transformed_ls2 = round(gp.kernel.base_kernel.lengthscale[0][2].item() * data.input_normalize.item(), 2)
                print(f"Epoch: {epoch} - Likelihood: {loss.item():.3f}"
                    f" - Lengthscales: [{transformed_ls0}, {transformed_ls1}, {transformed_ls2}]")
                #del ll_full
                
            del loss, points, ll
        
        gp = gp.cpu()
        likelihood = likelihood.cpu()
        data.flow = flow.cpu()
        data.sigma2 = round(gp.kernel.outputscale.item() * data.output_normalize.item(), 2)
        data.l0     = round(gp.kernel.base_kernel.lengthscale[0][0].item() * data.time_normalize.item(), 2)
        data.l1    = round(gp.kernel.base_kernel.lengthscale[0][1].item() * data.input_normalize.item(), 2)
        data.l2     = round(gp.kernel.base_kernel.lengthscale[0][2].item() * data.input_normalize.item(), 2)
        data.tau2   = round(likelihood.noise.item() * data.output_normalize.item(), 2)