import xarray as xr 
from datetime import date, datetime, timedelta
import numpy as np
import random
import torch

from . import ode

def generate_time_ranges(date, minutes, start_time="00:00", end_time="23:59"):

    start = datetime.strptime(date + " " + start_time, "%Y-%m-%d %H:%M")
    end = datetime.strptime(date + " " + end_time, "%Y-%m-%d %H:%M")

    time_list = [start]
    while time_list[-1] + timedelta(minutes=minutes) <= end:
        time_list.append(time_list[-1] + timedelta(minutes=minutes))
    
    return time_list


def generate_dates(n, month = 1):

    start_date = date(2024, month, 1)
    end_date = date(2024, month, 30)

    all_dates = [
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end_date - start_date).days + 1)
    ]

    random.shuffle(all_dates)
    return all_dates[:n]


def partition_into_coarse_grids(grid, k):
    coarse_grids = []

    for i in range(k):
        for j in range(k):
            coarse = grid[i::k, j::k, :]  
            coarse_grids.append(coarse)

    return coarse_grids
        
class NormalizeTime:
    def __init__(self, T):
        self.max = T.max()

    def __call__(self, t):
        return t / self.max


class NormalizeSpace:
    def __init__(self, XY):
        self.min = XY.min(dim=0).values
        self.max = XY.max(dim=0).values

    def __call__(self, xy):
        return 2 * (xy - self.min) / (self.max - self.min) - 1


class NormalizeObs:
    def __init__(self, Z):
        flat = Z.reshape(-1)
        self.mean = flat.mean()
        self.std = flat.std()

    def __call__(self, z):
        return (z - self.mean) / self.std


def rescale_temporal_lengthscale(T, temporal_lengthscale):
    return temporal_lengthscale *T.max()

def rescale_spatial_lengthscales(XY, spatial_lengthscales):
    min = XY.min(dim=0).values
    max = XY.max(dim=0).values
    return spatial_lengthscales * ( (min - max) / 2)

def rescale_variance(Z, variance_normalized):
    std_Z = Z.reshape(-1).std()
    return variance_normalized * std_Z**2

def rmse(data, scale=1, mag=1):

    T = data.T
    XY = data.XY
    n_frames = T.shape[0]
    n_points = XY.shape[0]

    # Compute predicted velocities for each frame
    vel = ode.Vel_hat(data)
    UV = torch.zeros((n_frames, n_points, 2))
    for frame in range(n_frames):
        UV[frame] = vel(T[frame], XY) * scale * mag

    XY_UV = [torch.cat((XY, UV[frame]), dim=-1).detach() for frame in range(n_frames)]
    XY_UV_ = [
        torch.index_select(XYP_UV, dim=1, index=torch.tensor([0, 1, 3, 4]))
        for XYP_UV in data.XYP_UV
    ]

    errors = []
    for i in range(data.n):
        true_U = XY_UV_[i][:, 2]
        true_V = XY_UV_[i][:, 3]
        pred_U = XY_UV[i][:, 2]
        pred_V = XY_UV[i][:, 3]

        mse_U = torch.mean((true_U - pred_U) ** 2)
        mse_V = torch.mean((true_V - pred_V) ** 2)
        rmse_val = torch.sqrt(mse_U + mse_V)
        errors.append(rmse_val)

    mean_rmse = torch.stack(errors).mean()
    return mean_rmse


def rmse_U(data, scale=1, mag=1):

    T = data.T
    XY = data.XY
    n_frames = T.shape[0]
    n_points = XY.shape[0]

    vel = ode.Vel_hat(data)
    UV = torch.zeros((n_frames, n_points, 2))
    for frame in range(n_frames):
        UV[frame] = vel(T[frame], XY) * scale * mag

    XY_UV = [torch.cat((XY, UV[frame]), dim=-1).detach() for frame in range(n_frames)]
    XY_UV_ = [
        torch.index_select(XYP_UV, dim=1, index=torch.tensor([0, 1, 3, 4]))
        for XYP_UV in data.XYP_UV
    ]

    errors = []
    for i in range(data.n):
        true_U = XY_UV_[i][:, 2]
        pred_U = XY_UV[i][:, 2]
        mse_U = torch.mean((true_U - pred_U) ** 2)
        rmse_val = torch.sqrt(mse_U)
        errors.append(rmse_val)

    mean_rmse = torch.stack(errors).mean()
    return mean_rmse


def rmse_V(data, scale=1, mag=1):

    T = data.T
    XY = data.XY
    n_frames = T.shape[0]
    n_points = XY.shape[0]

    # Compute predicted velocities for each frame
    vel = ode.Vel_hat(data)
    UV = torch.zeros((n_frames, n_points, 2))
    for frame in range(n_frames):
        UV[frame] = vel(T[frame], XY) * scale * mag

    XY_UV = [torch.cat((XY, UV[frame]), dim=-1).detach() for frame in range(n_frames)]
    XY_UV_ = [
        torch.index_select(XYP_UV, dim=1, index=torch.tensor([0, 1, 3, 4]))
        for XYP_UV in data.XYP_UV
    ]

    errors = []
    for i in range(data.n):
        true_V = XY_UV_[i][:, 3]
        pred_V = XY_UV[i][:, 3]
        mse_V = torch.mean((true_V - pred_V) ** 2)
        rmse_val = torch.sqrt(mse_V)
        errors.append(rmse_val)

    mean_rmse = torch.stack(errors).mean()
    return mean_rmse



def rmse_datas(data1, data2, t, scale=1, mag=1):

    XY = data1.XY
    n_points = XY.shape[0]

    vel1 = ode.Vel_hat(data1)
    vel2 = ode.Vel_hat(data2)
    UV1 = torch.zeros((1, n_points, 2))
    UV2 = torch.zeros((1, n_points, 2))
    n_frames = 1
    for frame in range(n_frames):
        UV1[frame] = vel1(t, XY) * scale * mag
        UV1[frame] = vel2(t, XY) * scale * mag
        

    XY_UV1 = [torch.cat((XY, UV1[frame]), dim=-1).detach() for frame in range(n_frames)]
    XY_UV2 = [torch.cat((XY, UV2[frame]), dim=-1).detach() for frame in range(n_frames)]


    errors = []
    for i in range(n_frames):
        U1 = XY_UV1[i][:, 2]
        U2 = XY_UV2[i][:, 3]
        V1 = XY_UV1[i][:, 2]
        V2 = XY_UV2[i][:, 3]

        mse_U = torch.mean((U1 - U2) ** 2)
        mse_V = torch.mean((V1 - V2) ** 2)
        rmse_val = torch.sqrt(mse_U + mse_V)
        errors.append(rmse_val)

    mean_rmse = torch.stack(errors).mean()
    return mean_rmse

