# -*- coding: utf-8 -*-
# @Time    : 4/28/2021 9:33 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : rbo_transform_np.py
# @Software: pose.reg

import os
import math
from scipy.spatial.transform import Rotation as sR

from agents.helper import *


def deg2rad(degree_angles):
    return np.deg2rad(degree_angles).astype(np.float32)

def is_unit_vector(v):
    # Checks if a vector is a valid unit vector
    magnitude = np.linalg.norm(v) # should be 1
    diff = abs(1. - magnitude)
    return diff < 1e-4

def are_unit_vectors(v):
    # Checks if a vector is a valid unit vector
    magnitude = np.linalg.norm(v, axis=-1) # should be 1
    diff = np.abs(1. - magnitude)
    return np.all(diff < 1e-4)

def are_translation_matrices(T, is_transposed=False):
    # Checks if the last two axis in multi-dimensional array T looks like a translation matrix
    # 4x4 translation matrix: |1  0  0  tx|  -  4x4 transposed translation matrix: | 1  0  0  0|
    #                         |0  1  0  ty|                                        | 0  1  0  0|
    #                         |0  0  1  tz|                                        | 0  0  1  0|
    #                         |0  0  0   1|                                        |tx ty tz  1|
    assert (np.ndim(T)==5), 'T should be ?xfxjx4x4, but ndim=={}'.format(np.ndim(T))
    assert (T.shape[3:]==(4,4)), 'Translation should be 4x4 matrix, not {}'.format(T.shape[3:])
    diagonals = T[:,:,:,[0,1,2,3],[0,1,2,3]]
    diagonals_are_1s = np.abs(diagonals - 1.) < 1e-3
    if is_transposed:
        zero_terms = T[:,:,:,[0,0,0,1,1,1,2,2,2],[1,2,3,0,2,3,0,1,3]] # for transposed translation matrix
    else: zero_terms = T[:,:,:,[0,0,1,1,2,2,3,3,3],[1,2,0,2,0,1,0,1,2]] # for translation matrix
    zero_terms_are_0s = np.abs(zero_terms) < 1e-3
    are_translations = np.all(diagonals_are_1s) and np.all(zero_terms_are_0s)
    if not are_translations:
        print ('Not translation matrix:\n{}'.format(np.concatenate((diagonals_are_1s, zero_terms_are_0s), axis=-1)))
    return are_translations

def are_freebones_similar(rmtx_free_bone_uvecs, quat_free_bone_uvecs):
    fb_diffs = rmtx_free_bone_uvecs - quat_free_bone_uvecs
    fb_dists = np.linalg.norm(fb_diffs, axis=-1, keepdims=True)
    are_similar_fbs = np.isclose(fb_dists[...,0], 0, atol=5e-04)
    all_are_similar_fbs = np.all(are_similar_fbs)
    if not all_are_similar_fbs:
        are_not_similar_fbs = np.logical_not(are_similar_fbs.flatten()) # (?*f*j,)
        fb_dists = np.tile(fb_dists, (1,1,1,3))
        fb_stack = np.stack([rmtx_free_bone_uvecs, quat_free_bone_uvecs, fb_diffs, fb_dists], axis=-2) # (?,f,j,4,3)
        not_similar_fbs = fb_stack.reshape((-1,4,3))[are_not_similar_fbs,:,:]
        print('\n[Assert Warning] {:,} (or {:.2%}) of RMtx-Fb and Quat-Fb are not similar (norms >{})\n{}\n'.format(
            not_similar_fbs.shape[0], not_similar_fbs.shape[0]/ are_not_similar_fbs.shape[0], 5e-04, not_similar_fbs))
    return all_are_similar_fbs

def is_rotation_matrix(R):
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    shouldBeIdentity = np.around(shouldBeIdentity, 1)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 10 # 1->1mm, 0.5->0.5mm (other options: 1e-3, 1e-6)

def normal_vec(vec1, vec2):
    # compute the normal (orthogonal vector) to vec1 and vec2
    normal_vec = np.cross(vec1, vec2)
    if np.sum(normal_vec)==0:
        # vec1 is parallel to vec2 (either same or opposite directions)
        # As quick fix we adjust the y-component of vec2 to break parallelism
        print('Correcting parallelism..')
        vec2[1] += 10 # shift y-component by 10 millimeters
        normal_vec = np.cross(vec1, vec2)
    return normal_vec

def rotation_matrix(unit_vec_a, unit_vec_b, I_mtx=np.identity(3, dtype=np.float32)):
    # Derive the rotation matrix that rotates unit_vec_a onto unit_vec_b
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    c = np.dot(unit_vec_a, unit_vec_b) # cosine of angle
    if c==-1: # implies the angle is +/-180" (a flip) and hence a case where the rotation matrix is not 1-1
        # for our application clipping the angle at +-179" may be a sufficient work-around this case
        c += 1e-4 # ideally 0.00015
    const = 1. / (1.+c)
    v = np.cross(unit_vec_a, unit_vec_b)
    v_mtx = np.float32([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    #s = np.linalg.norm(v) # sine of angle
    #assert (const-((1-c)/s**2)) < 1e-0, '{} vs. {}'.format(const, (1-c)/s**2) # should be equivalent
    R_mtx = I_mtx + v_mtx + np.matmul(v_mtx, v_mtx)*const
    return R_mtx

def rotation_matrix_to_xyz_euler_angles(R):
    # Calculates xyz-euler-angles from rotation matrix. The result is the same as
    # MATLAB except the order of the euler angles (x and z are swapped).
    # https://learnopencv.com/rotation-matrix-to-euler-angles/
    #assert(is_rotation_matrix(R)), 'R not orthogonal:\n{}\n{}'.format(R, np.dot(R.T, R))

    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6

    if  not singular:
        x_phi = math.atan2(R[2,1] , R[2,2])
        assert (not (abs(R[2,1])<1e-6 and abs(R[2,2])<1e-6)), 'y:{} and x:{}'.format(R[2,1], R[2,2])
        y_theta = math.atan2(-R[2,0], sy)
        assert (not (abs(-R[2,0])<1e-6 and sy<1e-6)), 'y:{} and x:{}'.format(-R[2,0], sy)
        z_sai = math.atan2(R[1,0], R[0,0])
        assert (not (abs(R[1,0])<1e-6 and abs(R[0,0])<1e-6)), 'y:{} and x:{}'.format(R[1,0], R[0,0])
    else:
        x_phi = math.atan2(-R[1,2], R[1,1])
        assert (not (abs(-R[1,2])<1e-6 and abs(R[1,1])<1e-6)), 'y:{} and x:{}'.format(-R[1,2], R[1,1])
        y_theta = math.atan2(-R[2,0], sy)
        assert (not (abs(-R[2,0])<1e-6 and sy<1e-6)), 'y:{} and x:{}'.format(-R[2,0], sy)
        z_sai = 0

    euler_angles = [x_phi, y_theta, z_sai]
    #assert(np.all(-3.143<euler_angles) and np.all(euler_angles<3.143)), 'euler_angles:{}'.format(euler_angles)
    return euler_angles  # angles returned in radians

def numpy_atan2(y_tensor, x_tensor):
    # arctan2 function is a piece-wise arctan function and is only differentiable
    # when x>0 or y!=0 see https://en.wikipedia.org/wiki/Atan2
    x_is_zero = np.abs(x_tensor) < 1e-1
    y_is_zero = np.abs(y_tensor) < 1e-1
    assert (np.any(np.logical_not(np.logical_and(y_is_zero, x_is_zero)))), 'y-x:\n{}'.format(np.stack([y_tensor, x_tensor], axis=-1))
    atan2_yx = np.arctan2(y_tensor, x_tensor)
    return atan2_yx

def numpy_cross(a_vec_tensors, b_vec_tensors):
    # cross product of vectors in dim=2
    # equivalent to np.cross(a_vec_tensors, b_vec_tensors)
    # a_vec_tensor-->(?,f,3)
    # b_vec_tensor-->(?,f,3)
    cross_prod_0 = np.cross(a_vec_tensors, b_vec_tensors)
    return cross_prod_0

def numpy_vecdot(a_vec_tensors, b_vec_tensors, ndims=3, vsize=3):
    # dot product of vectors in dim=2
    # equivalent to np.dot(a_vec_tensors, b_vec_tensors)
    dot_prod_0 = np.sum(a_vec_tensors * b_vec_tensors, axis=-1)
    return dot_prod_0 # (?,f) or (?,f,n)

def numpy_matdot(mtx_tensor, vec_tensors):
    # dot product between matrices in dim=2&3 and vectors in dim=3
    # equivalent to np.dot(mtx_tensor, vec_tensors.T).T
    # mtx_tensor-->(?,f,4,4)
    # vec_tensor-->(?,f,n,4)
    axes = (0,1,3,2) if np.ndim(vec_tensors)==4 else (0,1,2,4,3)
    mat_dot_0 = np.transpose(np.matmul(mtx_tensor, np.transpose(vec_tensors, axes=axes)), axes=axes)
    return mat_dot_0


def numpy_matmul(a_mtx_tensor, b_mtx_tensor, n=3):
    # matrix multiplication between two matrices
    # equivalent to np.matmul(a_mtx_tensor, b_mtx_tensor)
    # a_mtx_tensor-->(?,f,n,n)
    # b_mtx_tensor-->(?,f,n,n)
    mat_mul_0 = np.matmul(a_mtx_tensor, b_mtx_tensor)
    return mat_mul_0


def save_jmc_priors(fb_orientation_vecs, jme_ordered_tags, n_fb_jnts, args,
                    supr_subset_tag, frm_rate_tag, jmc_joint_config, from_torch=True):
    op_tag = '_tch' if from_torch else '_npy'
    aug_tag = 'wxaug' if args.data_augmentation else 'nouag'
    grp_tag = 'grpjnt' if args.group_jmc else 'perjnt'
    fbq_tag = 5 if args.quintuple else 4
    file_path = os.path.join('priors', supr_subset_tag, 'properties', 'fbj_orients{}_{}{}_{}_{}_{}{}.pickle'.
                        format(frm_rate_tag, args.jmc_fbo_ops_type, fbq_tag, aug_tag, n_fb_jnts, grp_tag, op_tag))
    jme_prior_src = {'joint_align_config':jmc_joint_config, 'keypoint_indexes':KPT_2_IDX, 'q_kpt_set':fbq_tag,
                     'joint_order':jme_ordered_tags, 'group':args.group_jmc, 'augment':args.data_augmentation}
    for joint_id, joint_orient_vecs in fb_orientation_vecs.items():
        jme_prior_src[joint_id] = np.concatenate(joint_orient_vecs, axis=0)
    pickle_write(jme_prior_src, file_path)


def save_bone_prior_properties(proportions_metadata, bpc_ordered_tags, blen_std, augment,
                               supr_subset_tag, frm_rate_tag, n_ratios, from_torch=True):
    properties_dir = os.path.join('priors', supr_subset_tag, 'properties')
    op_tag = '_tch' if from_torch else '_npy'
    aug_tag = 'wxaug' if augment else 'nouag'
    std_tag = '' if blen_std is None else '_blstd{}'.format(str(blen_std).replace("0.", "."))
    if len(proportions_metadata['bone_lengths'])>0:
        bone_len = np.concatenate(proportions_metadata['bone_lengths'])
        file_path = os.path.join(properties_dir, 'bone_lengths{}_16_m.npy'.format(frm_rate_tag))
        np.save(file_path, bone_len)
        print('min bone lengths: {} - {} m'.format(np.min(bone_len, axis=(0,1)), np.min(bone_len)))
        print('avg bone lengths: {} - {} m'.format(np.mean(bone_len, axis=(0,1)), np.mean(bone_len)))
        print('max bone lengths: {} - {} m'.format(np.max(bone_len, axis=(0,1)), np.max(bone_len)))
    if len(proportions_metadata['bone_symms'])>0:
        bone_symm = np.concatenate(proportions_metadata['bone_symms'])
        file_path = os.path.join(properties_dir, 'bone_symms{}_6_m.npy'.format(frm_rate_tag))
        np.save(file_path, bone_symm)
        print('avg bone symmetry fractions: {}'.format(np.mean(bone_symm, axis=0, keepdims=False)))
    if len(proportions_metadata['bone_ratios'])>0:
        bone_ratio = np.concatenate(proportions_metadata['bone_ratios'])
        assert (n_ratios==bone_ratio.shape[-1]), '{} vs. {}'.format(n_ratios, bone_ratio.shape[-1])
        bpc_prior_src = {'bone_ratios':bone_ratio, 'keypoint_indexes':KPT_2_IDX, 'bone_kpt_pairs':BONE_KPT_PAIRS,
                         'ratio_order':bpc_ordered_tags, 'augment':augment, 'induced_blen_std': blen_std,
                         'rgt_sym_bones':RGT_SYM_BONES, 'lft_sym_bones':LFT_SYM_BONES}
        file_path = os.path.join(properties_dir, 'bone_ratios_{}{}_{}{}_m{}.pickle'.
                                 format(aug_tag, std_tag, n_ratios, frm_rate_tag, op_tag))
        pickle_write(bpc_prior_src, file_path)
        print('avg bone proportion ratio:\n{}'.format(np.mean(bone_ratio, axis=0, keepdims=False)))


# Quaternions
# -------------------------------------------------------------------------------------------------

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult_v1(q1, v1):
    # generic q*v*q' implementation
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def qv_mult_v2(q1, v1):
    q = np.tile(q1, (17, 1)) # (17,4)
    # from quaternion.py>>qro()
    qvec = q[:, 1:]
    uv = np.cross(qvec, v1, axis=-1)
    uuv = np.cross(qvec, uv, axis=-1)
    return v1 + 2 * (q[:, :1] * uv + uuv)

def rotmtx_quaternion_rotate(pose3d, rot_mtx, pvt_kpt):
    '''
    Converts 3x3 rotation matrix to quaternion and uses the derived quaternion to transform poses
    Args:
        pose3d: (j,3) 3D keypoint coordinates of a single pose
        rot_matrix: (3,3) corresponding rotation matrix to transform align pose
    Returns: (j,3) poses after translation and rotation
    '''
    # translate pose to origin first
    vpose = pose3d - pvt_kpt

    # quaternion from rotation matrix
    # option 1
    quat_obj = sR.from_matrix(rot_mtx)
    quaternion = quat_obj.as_quat()
    sr_quat = np.roll(quaternion, 1)
    # option 2
    rot_mtx = rot_mtx.T
    if rot_mtx[2,2] < 0:
        if rot_mtx[0,0] > rot_mtx[1,1]:
            t = 1 + rot_mtx[0,0] - rot_mtx[1,1] - rot_mtx[2,2]
            q = [t, rot_mtx[0,1]+rot_mtx[1,0], rot_mtx[2,0]+rot_mtx[0,2], rot_mtx[1,2]-rot_mtx[2,1]]
        else:
            t = 1 - rot_mtx[0,0] + rot_mtx[1,1] - rot_mtx[2,2]
            q = [rot_mtx[0,1]+rot_mtx[1,0], t, rot_mtx[1,2]+rot_mtx[2,1], rot_mtx[2,0]-rot_mtx[0,2]]
    else:
        if rot_mtx[0,0] < -rot_mtx[1,1]:
            t = 1 - rot_mtx[0,0] - rot_mtx[1,1] + rot_mtx[2,2]
            q = [rot_mtx[2,0]+rot_mtx[0,2], rot_mtx[1,2]+rot_mtx[2,1], t, rot_mtx[0,1]-rot_mtx[1,0]]
        else:
            t = 1 + rot_mtx[0,0] + rot_mtx[1,1] + rot_mtx[2,2]
            q = [rot_mtx[1,2]-rot_mtx[2,1], rot_mtx[2,0]-rot_mtx[0,2], rot_mtx[0,1]-rot_mtx[1,0], t]
    q = np.float32(q)
    q *= 0.5 / np.sqrt(t)
    op_quat = np.roll(q, 1) # (4,)

    print('op_quat:{} vs.\nsr_quat:{}\n---------'.format(op_quat, sr_quat))
    return qv_mult_v2(sr_quat, vpose)


def quaternion_rotate(pose3d, q1, q2, pvt_kpt):
    # translate pose to origin first
    vpose = pose3d - pvt_kpt
    vpose = qv_mult_v2(q1, vpose)
    vpose = qv_mult_v2(q2, vpose)
    return vpose


