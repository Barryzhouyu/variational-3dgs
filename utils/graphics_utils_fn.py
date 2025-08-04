# graphics_utils_fn.py

import torch
import math
import numpy as np

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 1e-7  # ✅ Prevent division by zero
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    """Computes World-to-View transformation matrix"""
    Rt = np.zeros((4, 4), dtype=np.float32)

    Rt[:3, :3] = R.T  # Correct NumPy transpose
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    Computes World-to-View matrix with NO normalization applied.
    Use this if your poses and point cloud are already in raw space.
    """
    Rt = np.zeros((4, 4), dtype=np.float32)

    # Convert from torch to NumPy if needed
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()

    Rt[:3, :3] = R.T  # transpose rotation (world to view)
    Rt[:3, 3] = t     # translation remains as-is (in raw space)
    Rt[3, 3] = 1.0

    # World to camera = inverse of camera-to-world
    C2W = np.linalg.inv(Rt)

    # NO normalization step — use raw center
    # cam_center = (cam_center + translate) * scale   ← removed
    # Rt = np.linalg.inv(C2W)

    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Generates a perspective projection matrix."""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

