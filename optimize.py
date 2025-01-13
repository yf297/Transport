import torch
from likelihoods import ll, full_ll
import generate
import penalty

def initial_state(y, x, model, num_epochs=200):
    
    n = y.shape[0]-1
    t = generate.time(n)
    
    optimizer = torch.optim.Adam([
        {'params': model.Mean.parameters()},
        {'params': model.Kernel.parameters(), 'lr': 0.1},
        {'params': model.likelihood.parameters(), 'lr': 0.1}
    ])
    
    y0 = y[0, :]
    t0x = generate.space_time( t[0].unsqueeze(0), x )
    model.set_train_data(t0x, y0, strict=False)
    
    m = y0.numel()
    model.train()
    
    for epoch in range(1,num_epochs+1):
        
        optimizer.zero_grad() 
        
        mean = model.likelihood(model(t0x)).mean
        cov = model.likelihood(model(t0x)).covariance_matrix 
        likelihood = ll(y0, mean, cov)
        
        loss = likelihood/m
        loss.backward()
        optimizer.step()
            
        if epoch % 50 == 0:
            print(f'Epoch: {epoch:d} - Likelihood: {likelihood.item()/m:.3f}')



def flow(y, x, model, num_epochs=20):

    n = y.shape[0]-1   
    N = y.numel()
    m = x.shape[0]
    
    t = generate.time(n)
    tx = generate.space_time(t,x)

    optimizer = torch.optim.Adam([
        {'params': model.Flow.parameters()},
        {'params': model.Vel.parameters()},
        {'params': model.likelihood.parameters(), 'lr': 0.1}
    ])
    

    model.eval()
    for epoch in range(1,num_epochs+1):
        for i in range(1,n+1):
            optimizer.zero_grad()

            t0x = generate.space_time( t[i-1].unsqueeze(0), x )
            y0 = y[(i-1),:]
            model.set_train_data(t0x, y0, strict=False)
            
            t1x = generate.space_time( t[i].unsqueeze(0), x )            
            y1 = y[i,:]
            
            mean = model.likelihood(model(t1x)).mean
            cov = model.likelihood(model(t1x)).covariance_matrix
            likelihood = ll(y1, mean, cov)
            
            pen = penalty.PDE(tx, model.Flow, model.Vel)
            alpha = 0.01 * torch.abs((likelihood / m) / pen).item()
            
            loss = likelihood/m + alpha*pen
            loss.backward()
            optimizer.step()
            
                
        if epoch % 5 == 0:
            full_likelihood = full_ll(y, x, model)
            print(f'Epoch: {epoch:.2f} - Likelihood: {full_likelihood.item()/N:.3f}')