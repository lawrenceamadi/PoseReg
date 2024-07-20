# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Extensive modification of VideoPose3D source code
# by researchers at the Visual Computing Lab @ IIT

import sys
import torch
import numpy as np

sys.path.append('../')
from agents.helper import BONE_CHILD_KPTS_IDXS, BONE_PARENT_KPTS_IDXS

def mpjpe(predicted, target, ret_per_kpt=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    #assert(predicted.shape==target.shape), '{} vs {}'.format(predicted.shape, target.shape) # (?,f,j,3)
    pos_err = torch.linalg.norm(predicted - target, dim=-1)
    if ret_per_kpt:
        return torch.mean(pos_err), torch.mean(pos_err, dim=(0,1))
    return torch.mean(pos_err)
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    #assert(predicted.shape == target.shape)
    #assert(w.shape[0] == predicted.shape[0])
    return torch.mean(w * torch.linalg.norm(predicted - target, dim=-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert (predicted.shape == target.shape) # (?*f,j,3)
    
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
    procrustes_err = np.linalg.norm(predicted_aligned - target, axis=-1) # (?*f,j)
    return np.mean(procrustes_err), np.mean(procrustes_err, axis=0)
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert (predicted.shape == target.shape), '{} vs. {}'.format(predicted.shape, target.shape) # (?,f,j,3)
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target, ret_per_kpt=True)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert (predicted.shape == target.shape) # (?,f,j,3)
    
    velocity_predicted = np.diff(predicted, axis=0) # (f,j,3)
    velocity_target = np.diff(target, axis=0)  # (f,j,3)
    velocity_err = np.linalg.norm(velocity_predicted - velocity_target, axis=-1) # (f,j)
    return np.mean(velocity_err), np.mean(velocity_err, axis=0)


def torch_p_mpjpe(predicted, target, with_scale_align=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    #assert(predicted.shape == target.shape and len(predicted.shape) == 3) # (?*f,j,3)
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


def mpboe(predicted, target, fb_obj, qset_kpt_idxs, v2o_bone_idxs,
          fb_oi_indexes=None, swap_01_axes=True, ret_fb_dist=False, ret_per_kpt=True):
    if swap_01_axes:
        predicted = torch.swapaxes(predicted, 0, 1) # (?,f,j,3) or (f,?,j,3)
        target = torch.swapaxes(target, 0, 1) # (?,f,j,3) or (f,?,j,3)

    # compute target poses' bone lengths
    target_dist = target[:,:,BONE_CHILD_KPTS_IDXS] - target[:,:,BONE_PARENT_KPTS_IDXS] # (?,f,b,3) or (f,?,b,3)
    targ_bone_len = torch.linalg.norm(target_dist, dim=3, keepdim=True) # (?,f,b,1) or (f,?,b,1)
    targ_bone_len = targ_bone_len[:,:,v2o_bone_idxs] # (?,f,b,1) or (f,?,b,1)
    # compute poses' free bone unit-vector orientation
    pred_fb_uvecs = fb_obj(predicted[:,:,qset_kpt_idxs,:]) # (?,f,b,3) or (f,?,b,3)
    targ_fb_uvecs = fb_obj(target[:,:,qset_kpt_idxs,:]) # (?,f,b,3) or (f,?,b,3)
    # scale free-bone unit vectors to target bone lengths
    pred_fb_uvecs = targ_bone_len * pred_fb_uvecs
    targ_fb_uvecs = targ_bone_len * targ_fb_uvecs
    fb_uvecs_dist = torch.linalg.norm(pred_fb_uvecs - targ_fb_uvecs, dim=-1) # (?,f,b) or (f,?,b)

    if ret_per_kpt:
        if fb_oi_indexes is not None: fb_uvecs_dist = fb_uvecs_dist[:,:,fb_oi_indexes] # drop Face bone (ie. b=15)
        return torch.mean(fb_uvecs_dist), torch.mean(fb_uvecs_dist, dim=(0,1)) # (?,f,b)->(1,), (b,) where b = 16 or 15
    if ret_fb_dist: return fb_uvecs_dist # (?,f,b=16)
    return torch.mean(fb_uvecs_dist) # (?,f,b=16) >> (1,)


def j_mpboe(predicted, target, fb_obj, qset_kpt_idxs, v2o_bone_idxs,
            joint_2_bone_mapping, kpt_2_idx, processor, fb_oi_indexes=None):
    # first compute MPBOE
    fb_uvecs_dist = mpboe(predicted, target, fb_obj, qset_kpt_idxs, v2o_bone_idxs,
                          fb_oi_indexes, swap_01_axes=True, ret_fb_dist=True, ret_per_kpt=False)
    drop_nose_kpt = fb_oi_indexes is not None

    # propagate MPBOE to joints
    nbsz, nfrm, nfbs = fb_uvecs_dist.shape # (?,f,b) or (f,?,b)
    njnt = 16 if drop_nose_kpt else 17 # exclude 'Nse' kpt if drop_nose_kpt
    jnt_mpbo_errs = torch.zeros((nbsz,nfrm,njnt), dtype=torch.float32, device=torch.device(processor)) # (?,f,j)

    for jnt in joint_2_bone_mapping.keys():
        if drop_nose_kpt and jnt=='Nse': continue
        jnt_idx = kpt_2_idx[jnt]
        fb_indexes, fb_kpt_wgt_contrib = joint_2_bone_mapping[jnt][0]

        for idx, fb_idx in enumerate(fb_indexes):
            assert (not drop_nose_kpt or fb_idx!=0), 'idx:{} fb_idx:{}'.format(idx, fb_idx) # drop_nose_kpt--> fb_idx is never 0 (0 is the index of UFace)
            jnt_mpbo_errs[:,:,jnt_idx] += fb_uvecs_dist[:,:,fb_idx] * fb_kpt_wgt_contrib[idx] # (?,f) jnt_weighted_err_sum
    return torch.mean(jnt_mpbo_errs), torch.mean(jnt_mpbo_errs, dim=(0,1)) # (?,f,j)->(1,), (j,)
