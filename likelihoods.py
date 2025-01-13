from gpytorch.distributions import MultivariateNormal
import generate

def ll(y, mean, cov):
    mvn_conditional = MultivariateNormal(mean, cov)
    log_prob = mvn_conditional.log_prob(y)
    return -1*log_prob


def full_ll(y, x, model):
    
    n = y.shape[0]-1
    t = generate.time(n)
    likelihood = 0
    
    for i in range(1, n+1):
        t0x = generate.space_time( t[i-1].unsqueeze(0), x )               
        y0 = y[(i-1),:]
        model.set_train_data(t0x, y0, strict=False)
        
        t1x = generate.space_time( t[i].unsqueeze(0), x )
        y1 = y[i,:]
        
        mean = model.likelihood(model(t1x)).mean
        cov = model.likelihood(model(t1x)).covariance_matrix
        likelihood += ll(y1, mean, cov)
        
    return likelihood