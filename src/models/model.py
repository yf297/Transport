
class Transport:
    def __init__(self, Data):
        
        self.Data = Data
        self.T = self.Data.ObservationTimes
        self.XY = self.Data.ObservationLocations
        self.Z = self.Data.Observations
        
        self.NormalizeTime = Normalize.Time(self.T)
        self.NormalizeSpace = Normalize.Space(self.XY)
        self.NormalizeSpaceInverse = Normalize.SpaceInverse(self.XY)
        self.NormalizeObs = Normalize.Obs(self.Z)
        
        self.NetFlow = Net.Flow(L=3,k=5)
        self.NetVelocity = Net.Velocity(L = 3)
        
        self.GaussianProcess = Gaussian.Process(MeanSpaceTime = gpytorch.means.ConstantMean(), 
                                                KernelSpaceTime = Kernel.SpaceTime(l0=6, l1=0.2, l2=0.2).Kernel, 
                                                Likelihood = gpytorch.likelihoods.GaussianLikelihood(), 
                                                Flow = self.NetFlow)

        self.Optimizer = torch.optim.Adam([
            {'params': self.NetFlow.parameters(), 'lr': 0.01},
            {'params': self.GaussianProcess.kernel.parameters(), 'lr': 0.1},
            {'params': self.GaussianProcess.mean.parameters(), 'lr': 0.1},
            {'params': self.GaussianProcess.likelihood.parameters(), 'lr': 0.1}
        ])
            
    
    def TrainMLE(self, Epochs, SubSampleSize, Neighbors):
        Train.MLE(self, Epochs=Epochs, SubSampleSize=SubSampleSize, Neighbors=Neighbors)

    def PlotVelocities(self,
                       Factor=None,
                       SubSampleSize=None,
                       Frame = 0, 
                       Gif = False):
        Velocity = ODE.Velocity(Scale.Flow(self.NetFlow, self.NormalizeTime, self.NormalizeSpace, self.NormalizeSpaceInverse))
        Velocities = []
        
        if Factor is not None:
            ObservationLocations = [self.Data.ObservationLocations.reshape(self.Data.ObservationGrid[0], 
                                              self.Data.ObservationGrid[1],2)[::Factor, ::Factor, :]
                                    for _ in range(self.Data.VelocityTimes.shape[0])]
            
            for frame in range(self.Data.VelocityTimes.shape[0]):
               Velocities.append( Velocity(self.Data.VelocityTimes[frame], ObservationLocations[frame].reshape(-1,2)).reshape(ObservationLocations[frame].shape) )
            Grid = True
        
        elif SubSampleSize is not None:
            idx =  torch.randperm(self.Data.ObservationLocations.shape[0])[:SubSampleSize] 
            ObservationLocations = [self.Data.ObservationLocations[idx,:] for _ in range(self.Data.VelocityTimes.shape[0])]
            
            for frame in range(self.Data.VelocityTimes.shape[0]):
                Velocities.append(Velocity(self.Data.VelocityTimes[frame], ObservationLocations[frame]))
            Grid = False
       
        return Plot.Velocities(ObservationLocations, Velocities, self.Data.VelocityMap, Frame, Gif, Grid)
    
    
    def RMSE(self):
        Velocity = ODE.Velocity(
        Scale.Flow(self.NetFlow,
                   self.NormalizeTime,
                   self.NormalizeSpace,
                   self.NormalizeSpaceInverse))

        errors = []
        for t, locs, true_uv in zip(
                self.Data.VelocityTimes,
                self.Data.VelocityLocations,
                self.Data.Velocities):

            pred_uv = Velocity(t, locs)         
            se = ((pred_uv - true_uv)**2).sum(dim=1)  
            mse = se.mean()                        
            errors.append(torch.sqrt(mse))        

        mean_rmse = torch.stack(errors).mean()      

        return mean_rmse.item()

