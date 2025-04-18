import torch
from torch import Tensor
from typing import Tuple

class VecchiaBlocks:
    def __init__(self, T: Tensor, XY: Tensor, Z: Tensor):
        self.T  = T
        self.XY = XY
        self.Z  = Z

    def prediction(self, i: int, 
                   idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        T = self.T[i]                 
        XY = self.XY[idx,:]              
        M  = XY.size(0)
        
        TXY_pred = torch.cat([T.repeat(M, 1), XY], dim=-1) 
        Z_pred = self.Z[i, idx]
        return TXY_pred, Z_pred

    def conditioning(
        self,
        i: int,
        idx: Tensor,
        nn: int
    ) -> Tuple[Tensor, Tensor]:
        j       = max(0, i - nn)
        T   = self.T[j:i]             
        XY  = self.XY[idx,:]           
        L, M = T.size(0), XY.size(0)
        
        TXY_cond = torch.cat([
        T.repeat_interleave(M).unsqueeze(1),
        XY.repeat(L, 1)
    ], dim=-1)         
        
        Z_cond = self.Z[j:i, idx].reshape(-1) 
        return TXY_cond, Z_cond
