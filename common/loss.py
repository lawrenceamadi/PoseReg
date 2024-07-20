# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
import numpy as np

def mpjpe(predicted, target, n_samples_oi=None, shapeMatch=True):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert (not shapeMatch or predicted.shape==target.shape), '{} vs. {}'.format(predicted.shape, target.shape) # (?,f,p,3)
    if n_samples_oi is None:
        return torch.mean(torch.linalg.norm(predicted - target, dim=-1))
    else:
        n_samples, n_frames, n_joints = predicted.shape[:3]
        return torch.sum(torch.linalg.norm(predicted - target, dim=-1)) / (n_samples_oi*n_frames*n_joints)
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.linalg.norm(predicted - target, dim=-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape # (?*f,p,3)
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape # (?,f,p,3)
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape # (?,f,p,3)
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=-1))


def torch_p_mpjpe(predicted, target, with_scale_align=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert (predicted.shape == target.shape and len(predicted.shape) == 3) # (?*f,p,3)
    with torch.no_grad():
        muX = torch.mean(target, dim=1, keepdim=True)
        muY = torch.mean(predicted, dim=1, keepdim=True)

        X0 = target - muX
        Y0 = predicted - muY

        normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdim=True))
        normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdim=True))

        X0 /= normX
        Y0 /= normY

        H = torch.matmul(torch.transpose(X0, 2, 1), Y0)
        U, s, Vt = torch.linalg.svd(H)
        V = torch.transpose(Vt, 2, 1)
        R = torch.matmul(V, torch.transpose(U, 2, 1))

        # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
        sign_detR = torch.sign(torch.unsqueeze(torch.linalg.det(R), dim=1))
        V[:, :, -1] *= sign_detR
        s[:, -1] *= torch.flatten(sign_detR)
        R = torch.matmul(V, torch.transpose(U, 2, 1)) # Rotation

        tr = torch.unsqueeze(torch.sum(s, dim=1, keepdim=True), dim=2)

        a = tr * normX / normY # Scale
        t = muX - a*torch.matmul(muY, R) # Translation

    # Perform rigid transformation on the input
    if with_scale_align:
        predicted_aligned = a*torch.matmul(predicted, R) + t
    else: predicted_aligned = torch.matmul(predicted, R) + t

    # Return MPJPE
    return torch.linalg.norm(predicted_aligned - target, dim=-1)

# def get_outlier_thresh(input_tensor, quantiles, dim=None, keepdim=False):
#     per_dim_quantiles = torch.quantile(input_tensor.view(-1,17), quantiles, dim=dim, keepdim=keepdim)
#     per_batch_iqr = per_dim_quantiles[1] - per_dim_quantiles[0] # IQR = Q3 - Q1
#     upper_thresh = per_dim_quantiles[1] + 1.5*per_batch_iqr # Upper = Q3 + 1.5*IQR
#     return torch.unsqueeze(upper_thresh, dim=1)


