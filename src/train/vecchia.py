import torch

class VecchiaBlocks:
    def __init__(self, T, XY, Z):
        self.T = T
        self.XY = XY
        self.Z = Z


    def block(self, i):
        n, m = self.T.size(0), self.XY.size(0)
        m = self.XY.size(0)
        T  = self.T[i].unsqueeze(0).unsqueeze(1).expand(m,1)
        XY = self.XY
        Z  = self.Z[i,:]
        return T, XY, Z
    
    def all(self):
        n, m = self.T.size(0), self.XY.size(0)          
        T = self.T.unsqueeze(0).repeat_interleave(m).unsqueeze(1)
        XY = self.XY.repeat(n, 1)
        Z = self.Z.reshape(-1)
        return T, XY, Z
        
    def prediction(self):
        n, m = self.T.size(0), self.XY.size(0)
        T_pred  = self.T[1:].unsqueeze(1).unsqueeze(2).expand(n-1,m,1) 
        XY_pred = self.XY.unsqueeze(0).expand(n-1,m,2) 
        Z_pred  = self.Z[1:, :]
        return T_pred, XY_pred, Z_pred

    def conditioning(self):
        n, m = self.T.size(0), self.XY.size(0)
        T_cond = self.T[0:-1].unsqueeze(1).unsqueeze(2).expand(n-1,m,1) 
        XY_cond = self.XY.unsqueeze(0).expand(n-1,m,2) 
        Z_cond  = self.Z[0:-1, :]        
        return T_cond, XY_cond, Z_cond
