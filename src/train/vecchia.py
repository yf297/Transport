import torch
from typing import Tuple

class VecchiaBlocks:
    def __init__(self, T: torch.Tensor, XY: torch.Tensor, Z: torch.Tensor):
        self.T  = T
        self.XY = XY
        self.Z = Z
        

    def prediction(self, i: int, 
                   idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:    
        T = self.T  
        XY = self.XY[idx, :]
        Z =  self.Z[:, idx]
        M = XY.size(0)

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
        XY = self.XY[idx,:]
        Z =  self.Z[j:i,idx]
        
        TXY_cond = torch.cat([
            T.repeat_interleave(XY.size(0)).unsqueeze(1),
            XY.repeat(T.size(0), 1)
            ], dim=-1)
        Z_cond = Z.reshape(-1)
        return TXY_cond, Z_cond
