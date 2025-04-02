import xarray as xr 
from datetime import date, datetime, timedelta
import numpy as np
import random
import torch

from . import ode

def generate_time_ranges(date, minutes, start_time="00:00", end_time="23:59"):
    """
    Generate a list of datetime objects at a fixed interval (in minutes) 
    between a given start and end time for a particular date.

    Parameters:
    -----------
    date : str
        Date string in 'YYYY-MM-DD' format (e.g., '2023-01-01').
    minutes : int
        Interval in minutes between each generated time.
    start_time : str, optional
        Start time in 'HH:MM' format (default '00:00').
    end_time : str, optional
        End time in 'HH:MM' format (default '23:59').

    Returns:
    --------
    time_list : list of datetime
        A list of datetime objects from start to end at the specified interval.
    """

    start = datetime.strptime(date + " " + start_time, "%Y-%m-%d %H:%M")
    end = datetime.strptime(date + " " + end_time, "%Y-%m-%d %H:%M")

    time_list = [start]
    while time_list[-1] + timedelta(minutes=minutes) <= end:
        time_list.append(time_list[-1] + timedelta(minutes=minutes))
    
    return time_list


def generate_dates(n, month = 1):
    """
    Generate a list of `n` random dates as strings (YYYY-MM-DD) within the range
    2024-01-01 to 2024-12-30.

    Args:
        n (int): Number of dates to return.

    Returns:
        list of str: Randomly shuffled dates of length `n`.
                     Format: "YYYY-MM-DD".
    """
    start_date = date(2024, month, 1)
    end_date = date(2024, month, 30)

    all_dates = [
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end_date - start_date).days + 1)
    ]

    random.shuffle(all_dates)
    return all_dates[:n]


def rmse(data, scale=1, mag=1):
    """
    Compute the overall RMSE (root mean squared error) in both U and V velocity
    components, compared to ground-truth data.

    Args:
        data: An object containing:
              - data.T: Time steps (tensor of shape [n_frames]).
              - data.XY: Spatial coordinates (tensor of shape [n_points, 2]).
              - data.XYP_UV: Ground-truth data (list of tensors, each containing
                             columns [X, Y, P, U, V], or similar).
              - data.n: Number of data items to evaluate (int).
              - Possibly other attributes used by `ode.Vel_hat`.
        scale (float, optional): Scaling factor for predicted velocities. Defaults to 1.
        mag (float, optional): Additional multiplier for predicted velocities. Defaults to 1.

    Returns:
        torch.Tensor: Mean RMSE (scalar) across all frames and data items.
    """
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
    """
    Compute the RMSE in the U velocity component only.

    Args:
        data: An object containing:
              - data.T: Time steps (tensor of shape [n_frames]).
              - data.XY: Spatial coordinates (tensor of shape [n_points, 2]).
              - data.XYP_UV: Ground-truth data (list of tensors, each containing
                             columns [X, Y, P, U, V], or similar).
              - data.n: Number of data items to evaluate (int).
        scale (float, optional): Scaling factor for predicted U. Defaults to 1.
        mag (float, optional): Additional multiplier for predicted U. Defaults to 1.

    Returns:
        torch.Tensor: Mean RMSE (scalar) across all frames/items for the U component.
    """
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
        pred_U = XY_UV[i][:, 2]
        mse_U = torch.mean((true_U - pred_U) ** 2)
        rmse_val = torch.sqrt(mse_U)
        errors.append(rmse_val)

    mean_rmse = torch.stack(errors).mean()
    return mean_rmse


def rmse_V(data, scale=1, mag=1):
    """
    Compute the RMSE in the V velocity component only.

    Args:
        data: An object containing:
              - data.T: Time steps (tensor of shape [n_frames]).
              - data.XY: Spatial coordinates (tensor of shape [n_points, 2]).
              - data.XYP_UV: Ground-truth data (list of tensors, each containing
                             columns [X, Y, P, U, V], or similar).
              - data.n: Number of data items to evaluate (int).
        scale (float, optional): Scaling factor for predicted V. Defaults to 1.
        mag (float, optional): Additional multiplier for predicted V. Defaults to 1.

    Returns:
        torch.Tensor: Mean RMSE (scalar) across all frames/items for the V component.
    """
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
