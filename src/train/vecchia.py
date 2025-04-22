import torch
from typing import Tuple

def shifted_locs(tensor, stride):
    grids = []
    for i in range(stride):
        for j in range(stride):
            sub = tensor[i::stride, j::stride, :]
            grids.append(sub)
    return grids

def shifted_obs(tensor, stride):
    grids = []
    for i in range(stride):
        for j in range(stride):
            sub = tensor[:, i::stride, j::stride]
            grids.append(sub)
    return grids

class VecchiaBlocks:
    def __init__(self, T: torch.Tensor, XY: torch.Tensor, Z: torch.Tensor, stride: int = 4):
        self.T  = T
        self.XY_list = shifted_locs(XY, stride)
        self.Z_list = shifted_obs(Z, stride)
        

    def prediction(self, i: int, 
                   idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:      
        XY = self.XY_list[idx].reshape(-1, 2)
        Z =  self.Z_list[idx].reshape(self.T.size(0), -1)
        M  = XY.size(0)

        T = self.T[i]
        TXY_pred = torch.cat([T.repeat(M, 1), XY], dim=-1) 
        Z_pred = Z[i,:]
        return TXY_pred, Z_pred

    def conditioning(
        self,
        i: int,
        idx: torch.Tensor,
        nn: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        j       = max(0, i - nn)
        T   = self.T[j:i]             
       
        XY = self.XY_list[idx].reshape(-1, 2)
        Z =  self.Z_list[idx].reshape(self.T.size(0), -1)
       
        M  = XY.size(0)  
        L, M = T.size(0), XY.size(0)
        
        TXY_cond = torch.cat([
        T.repeat_interleave(M).unsqueeze(1),
        XY.repeat(L, 1)
    ], dim=-1)         
        
        Z_cond = Z[j:i, :].reshape(-1) 
        return TXY_cond, Z_cond
