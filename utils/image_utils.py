#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def nll_kernel_density(pred_rgbs, pred_std, ground_truth):
    n = pred_std.numel()
    eps = 1e-05
    H_sqrt = pred_std.detach() * torch.pow(0.8/n,torch.tensor(-1/7)) + eps # (N_rays, 3)
    H_sqrt = H_sqrt[...,None] # (N_rays, 3, 1)
    r_P_C_1 = torch.exp( -((pred_rgbs - ground_truth[...,None])**2) / (2*H_sqrt*H_sqrt)) # [N_rays, 3, k]
    r_P_C_2 = torch.pow(torch.tensor(2*math.pi),-1.5) / H_sqrt # [N_rays, 3, 1]
    r_P_C = r_P_C_1 * r_P_C_2 # [N_rays, 3, k]
    r_P_C_mean = r_P_C.mean(-1) + eps
    loss_nll = - torch.log(r_P_C_mean).mean()
    return loss_nll

def ause_br(unc_vec, err_vec, err_type='rmse'):
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    # Sort the error
    err_vec_sorted, _ = torch.sort(err_vec)

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        if err_type == 'rmse':
            ause_err.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae' or err_type == 'mse':
            ause_err.append(err_slice.mean().cpu().numpy())

    # Sort by variance
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = []
    for r in ratio_removed:
        
        err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        if err_type == 'rmse':
            ause_err_by_var.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae'or err_type == 'mse':
            ause_err_by_var.append(err_slice.mean().cpu().numpy())
    
    #Normalize and append
    max_val = max(max(ause_err), max(ause_err_by_var))
    ause_err = ause_err / max_val
    ause_err = np.array(ause_err)
    
    ause_err_by_var = ause_err_by_var / max_val
    ause_err_by_var = np.array(ause_err_by_var)
    ause = np.trapz(ause_err_by_var - ause_err, ratio_removed)

    return ause, ause_err, ause_err_by_var

