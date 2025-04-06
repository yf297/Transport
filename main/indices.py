import torch
from main import neighbors, order

class Prediction:
    def __init__(self, T, XY, Z, grid_size):
        self.T = T
        self.XY = XY
        self.Z = Z
        
        self.grid_size = grid_size
        self.bin_size = 2.0 / grid_size
        
        self.x_idx = ((XY[:, 0] + 1) / self.bin_size).floor().long()
        self.y_idx = ((XY[:, 1] + 1) / self.bin_size).floor().long()
        self.x_idx = torch.clamp(self.x_idx, max=grid_size - 1)
        self.y_idx = torch.clamp(self.y_idx, max=grid_size - 1)
        self.y_idx = grid_size - 1 - self.y_idx

    def __call__(self, cell, i):
        
        indices_in_cell = ((self.y_idx == cell[0]) & (self.x_idx == cell[1])).nonzero(as_tuple=True)[0]
        T1 = torch.atleast_1d(self.T[i])
        XY1 = self.XY[indices_in_cell]
        
        TXY1 = torch.cat(
                (
                    T1.repeat_interleave(XY1.shape[0]).unsqueeze(1),
                    XY1.repeat(T1.shape[0], 1)
                ),
                dim=1
            )
        Z1 =  self.Z[i][indices_in_cell]
        
        return TXY1, Z1

class Conditioning:
    def __init__(self, T, XY, Z, grid_size, k0, k1):
        
        self.T = T
        self.XY = XY
        self.Z = Z
        
        self.grid_size = grid_size
        self.k0 = k0
        self.k1 = k1
        self.bin_size = 2.0 / grid_size
        
        self.x_idx = ((self.XY[:, 0] + 1) / self.bin_size).floor().long()
        self.y_idx = ((self.XY[:, 1] + 1) / self.bin_size).floor().long()
        self.x_idx = torch.clamp(self.x_idx, max=grid_size - 1)
        self.y_idx = torch.clamp(self.y_idx, max=grid_size - 1)
        self.y_idx = grid_size - 1 - self.y_idx
        
        self.cells = order.max_min(grid_size)
        self.NNarray0 = neighbors.prev(k0, self.cells)
        self.NNarray1 = neighbors.full(k1, self.cells)
        self.k0 = k0

    def __call__(self, cell, i):
        
        neighbor_cells_i = self.NNarray0[cell]
        neighbor_cells_i_1 = self.NNarray1[cell]

        cell_check = cell in self.cells[1:] and self.k0 > 0
        time_check = i != 0
        

        if cell_check:
            XY_i_dict = {}
            Z_i_dict = {}
            
            for nb_cell in neighbor_cells_i:
                idx = ((self.y_idx == nb_cell[0]) & (self.x_idx == nb_cell[1])).nonzero(as_tuple=True)[0]
                XY_i_dict[nb_cell] = self.XY[idx]
                Z_i_dict[nb_cell] = self.Z[i][idx]
                
            T_i = torch.atleast_1d(self.T[i])
            XY_i = torch.cat([XY_i_dict[nb] for nb in neighbor_cells_i], dim=0)
            TXY_i = torch.cat(
                (
                    T_i.repeat_interleave(XY_i.shape[0]).unsqueeze(1),
                    XY_i.repeat(T_i.shape[0], 1)
                ),
                dim=1
            )
            
            
            Z_i = torch.cat([Z_i_dict[nb] for nb in neighbor_cells_i], dim=0)

        
        if time_check:
            XY_i_1_dict = {}
            Z_i_1_dict = {}
            for nb_cell in neighbor_cells_i_1:
                idx = ((self.y_idx == nb_cell[0]) & (self.x_idx == nb_cell[1])).nonzero(as_tuple=True)[0]
                XY_i_1_dict[nb_cell] = self.XY[idx]
                Z_i_1_dict[nb_cell] = self.Z[i-1][idx]
            
            T_i_1 = torch.atleast_1d(self.T[i-1])
            XY_i_1 = torch.cat([XY_i_1_dict[nb] for nb in neighbor_cells_i_1], dim=0)
            TXY_i_1 = torch.cat(
                (
                    T_i_1.repeat_interleave(XY_i_1.shape[0]).unsqueeze(1),
                    XY_i_1.repeat(T_i_1.shape[0], 1)
                ),
                dim=1
            )
            Z_i_1 = torch.cat([Z_i_1_dict[nb] for nb in neighbor_cells_i_1], dim=0)
            
            
        if time_check and cell_check:
            TXY0 = torch.cat([TXY_i_1, TXY_i], dim = 0)
            Z0 = torch.cat([Z_i_1,Z_i]) 

        if not time_check and cell_check:
            TXY0 = TXY_i
            Z0 = Z_i
            
        if time_check and not cell_check:
            TXY0 = TXY_i_1
            Z0 = Z_i_1
            
        return TXY0, Z0

    

