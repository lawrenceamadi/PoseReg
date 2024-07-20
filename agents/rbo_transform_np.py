# -*- coding: utf-8 -*-
# @Time    : 4/28/2021 9:33 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : rbo_transform_np.py
# @Software: videopose3d

import os
import math
import numpy as np
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

def is_rotation_matrix(R):
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    shouldBeIdentity = np.around(shouldBeIdentity, 1)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    #print('n:{}'.format(n))
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
    #print(unit_vec_a, unit_vec_b, v, s, c, const)
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
    # gradient when x>0 or y!=0
    #   partial_derivative_wrt_x = -y / (x^2 + y^2)
    #   partial_derivative_wrt_y =  x / (x^2 + y^2)
    # gradient when x==0 and y!=0
    #   partial_derivative_wrt_x = partial_derivative_wrt_y = 0
    # gradient when x==0 and y==0
    #   partial_derivative_wrt_x = partial_derivative_wrt_y = undefined (unless hard-reset to 0)
    #print(y_tensor, x_tensor)
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
    # i_components = a_vec_tensors[:,:,1]*b_vec_tensors[:,:,2] - a_vec_tensors[:,:,2]*b_vec_tensors[:,:,1] # (?,f)
    # j_components = a_vec_tensors[:,:,2]*b_vec_tensors[:,:,0] - a_vec_tensors[:,:,0]*b_vec_tensors[:,:,2] # (?,f)
    # k_components = a_vec_tensors[:,:,0]*b_vec_tensors[:,:,1] - a_vec_tensors[:,:,1]*b_vec_tensors[:,:,0] # (?,f)
    # cross_prod_1 =  np.stack([i_components, j_components, k_components], axis=2) # (?,f,3)
    # assert (np.all(cross_prod_1==cross_prod_0)), 'cross - python:\n{}\nnumpy:\n{}'.format(cross_prod_1, cross_prod_0)
    return cross_prod_0

def numpy_vecdot(a_vec_tensors, b_vec_tensors, ndims=3, vsize=3):
    # dot product of vectors in dim=2
    # equivalent to np.dot(a_vec_tensors, b_vec_tensors)
    dot_prod_0 = np.sum(a_vec_tensors * b_vec_tensors, axis=-1)

    # if ndims==3:
    #     # a_vec_tensor-->(?,f,3)
    #     # b_vec_tensor-->(?,f,3)
    #     dot_prod_1 = a_vec_tensors[:,:,0] * b_vec_tensors[:,:,0]
    #     for v_comp_idx in range(1, vsize):
    #         dot_prod_1 += a_vec_tensors[:,:,v_comp_idx] * b_vec_tensors[:,:,v_comp_idx]
    #
    #     b, f = b_vec_tensors.shape[:2]
    #     dot_prod_2 = np.zeros((b,f), dtype=np.float32)
    #     for b_idx in range(b):
    #         for f_idx in range(f):
    #             dot_prod_2[b_idx, f_idx] = np.dot(a_vec_tensors[b_idx, f_idx], b_vec_tensors[b_idx, f_idx])
    #
    # else: # ndims==4:
    #     # a_vec_tensor-->(?,f,m,3), m=1
    #     # b_vec_tensor-->(?,f,n,3)
    #     dot_prod_1 = a_vec_tensors[:,:,:,0] * b_vec_tensors[:,:,:,0] + \
    #                  a_vec_tensors[:,:,:,1] * b_vec_tensors[:,:,:,1] + \
    #                  a_vec_tensors[:,:,:,2] * b_vec_tensors[:,:,:,2] + \
    #                  a_vec_tensors[:,:,:,3] * b_vec_tensors[:,:,:,3]
    #
    #     b, f, n = b_vec_tensors.shape[:3]
    #     dot_prod_2 = np.zeros((b,f,n), dtype=np.float32)
    #     for b_idx in range(b):
    #         for f_idx in range(f):
    #             for n_idx in range(n):
    #                 dot_prod_2[b_idx,f_idx,n_idx] = np.dot(a_vec_tensors[b_idx,f_idx,0], b_vec_tensors[b_idx,f_idx,n_idx])
    #
    # print('dot_prod_2.shape:{}'.format(dot_prod_2.shape))
    # assert (np.all(np.abs(dot_prod_1-dot_prod_2)<1e-1)), 'python-vs-numpy dot product:\n{}'.format(np.stack([dot_prod_1, dot_prod_2, np.abs(dot_prod_1-dot_prod_2)<1e-3], axis=-1))
    # assert (np.all(np.abs(dot_prod_0-dot_prod_2)<1e-1)), 'python-vs-numpy dot product:\n{}'.format(np.stack([dot_prod_0, dot_prod_2, np.abs(dot_prod_0-dot_prod_2)<1e-3], axis=-1))
    return dot_prod_0 # (?,f) or (?,f,n)

def numpy_matdot(mtx_tensor, vec_tensors):
    # dot product between matrices in dim=2&3 and vectors in dim=3
    # equivalent to np.dot(mtx_tensor, vec_tensors.T).T
    # mtx_tensor-->(?,f,4,4)
    # vec_tensor-->(?,f,n,4)
    #mat_dot_0 = np.transpose(np.matmul(mtx_tensor, np.transpose(vec_tensors, axes=(0,1,3,2))), axes=(0,1,3,2))
    axes = (0,1,3,2) if np.ndim(vec_tensors)==4 else (0,1,2,4,3)
    mat_dot_0 = np.transpose(np.matmul(mtx_tensor, np.transpose(vec_tensors, axes=axes)), axes=axes)
    # row0_dot_prod = numpy_vecdot(mtx_tensor[:,:,[0],:], vec_tensors, ndims=4, vsize=4) # (?,f,n)
    # row1_dot_prod = numpy_vecdot(mtx_tensor[:,:,[1],:], vec_tensors, ndims=4, vsize=4) # (?,f,n)
    # row2_dot_prod = numpy_vecdot(mtx_tensor[:,:,[2],:], vec_tensors, ndims=4, vsize=4) # (?,f,n)
    # row3_dot_prod = numpy_vecdot(mtx_tensor[:,:,[3],:], vec_tensors, ndims=4, vsize=4) # (?,f,n)
    # mat_dot_1 = np.stack([row0_dot_prod, row1_dot_prod, row2_dot_prod, row3_dot_prod], axis=3) # (?,f,n,4)
    # assert (np.all(np.abs(mat_dot_1-mat_dot_0)<1e-1)), 'python-vs-numpy matrix dot:\n{}'.format(np.stack([mat_dot_1, mat_dot_0, np.abs(mat_dot_1-mat_dot_0)<1e-3], axis=-1))
    return mat_dot_0


def numpy_matmul(a_mtx_tensor, b_mtx_tensor, n=3):
    # matrix multiplication between two matrices
    # equivalent to np.matmul(a_mtx_tensor, b_mtx_tensor)
    # a_mtx_tensor-->(?,f,n,n)
    # b_mtx_tensor-->(?,f,n,n)
    mat_mul_0 = np.matmul(a_mtx_tensor, b_mtx_tensor)
    # row_list, col_list = list(), list()
    # #print('rot_mtx.shape:{}, tra_mtx.shape:{}'.format(a_mtx_tensor.shape, b_mtx_tensor.shape))
    # for i in range(n):
    #     for j in range(n):
    #         mij_dot_prod = numpy_vecdot(a_mtx_tensor[:,:,i,:], b_mtx_tensor[:,:,:,j], vsize=n) # (?,f)
    #         col_list.append(mij_dot_prod)
    #
    #     col_tensor = np.stack(col_list, axis=2) # (?,f,n)
    #     row_list.append(col_tensor)
    #     col_list.clear()
    #
    # mat_mul_1 = np.stack(row_list, axis=2) # (?,f,n,n)
    # assert (np.all(np.abs(mat_mul_1-mat_mul_0)<1e-1)), 'python-vs-numpy matrix mul:\n{}'.format(np.stack([mat_mul_1, mat_mul_0, np.abs(mat_mul_1-mat_mul_0)<1e-3], axis=-1))
    return mat_mul_0


def per_bone_orientation(quadruplet_kpts, z_axis_a_idx, z_axis_b_idx,
                         yx_axis_a_idx, yx_axis_b_idx, xy_axis_dir, xy_idx, yx_idx, z_idx=2,
                         pivot_kpt_idx=0, xy_axis_kpt_idx=1, yx_plane_kpt_idx=2, free_kpt_idx=3):
    '''Computes free bone orientation one bone and sample at a time'''
    T_mtx = np.identity(4, dtype=np.float32) # 4x4 homogenous translation matrix
    R_mtx = np.identity(4, dtype=np.float32) # 4x4 homogenous rotation matrix
    xyz_inter_vecs = np.zeros((3,3), dtype=np.float32) # 0:AXIS-vec, 1:vec-forming-PLANE-with-0, 2:z-AXIS-orthogonal-to-0&1
    xyz_axis_uvecs = np.zeros((3,3), dtype=np.float32) # 0:x-axis, 1:y-axis, 2:z-axis

    pivot_kpt, xy_axis_kpt, yx_plane_kpt, free_kpt = quadruplet_kpts[:,:3]

    # Deduce translation vector from pivot joint
    translation_vec = -pivot_kpt  # frame origin will be translated to pivot joint
    T_mtx[:3, 3] = translation_vec

    # Get limb vectors from pairs of keypoint positions
    xy_axis_vec = xy_axis_kpt - pivot_kpt
    yx_plane_vec = yx_plane_kpt - pivot_kpt
    xyz_inter_vecs[xy_idx] = xy_axis_vec
    xyz_inter_vecs[yx_idx] = yx_plane_vec

    # Define new x-axis, y-axis, and z-axis
    xy_axis_uvec = (xy_axis_vec / np.linalg.norm(xy_axis_vec)) * xy_axis_dir
    xyz_axis_uvecs[xy_idx] = xy_axis_uvec
    #assert(is_unit_vector(xy_axis_uvec)), '|xy_axis_uvec|:{}'.format(np.linalg.norm(xy_axis_uvec))
    z_axis_vec = np.cross(xyz_inter_vecs[z_axis_a_idx], xyz_inter_vecs[z_axis_b_idx])
    xyz_inter_vecs[z_idx] = z_axis_vec
    z_axis_uvec = z_axis_vec / np.linalg.norm(z_axis_vec)
    xyz_axis_uvecs[z_idx] = z_axis_uvec
    #assert(is_unit_vector(z_axis_uvec)), '|z_axis_uvec|:{}'.format(np.linalg.norm(z_axis_uvec))
    yx_axis_vec = np.cross(xyz_inter_vecs[yx_axis_a_idx], xyz_inter_vecs[yx_axis_b_idx])
    yx_axis_uvec = yx_axis_vec / np.linalg.norm(yx_axis_vec)
    xyz_axis_uvecs[yx_idx] = yx_axis_uvec
    #assert(is_unit_vector(y_axis_uvec)), '|y_axis_uvec|:{}'.format(np.linalg.norm(y_axis_uvec))

    # Derive frame rotation matrix from unit vec axis
    #assert(is_rotation_matrix(R)), 'R not orthogonal:\n{}\n{}'.format(R, np.dot(R.T, R))
    R_mtx[:3,:3] = xyz_axis_uvecs # xyz_axis_uvecs <-> R
    Tfm_mtx = np.matmul(R_mtx, T_mtx)

    # Using frame rotation matrix, transform position of non-origin translated
    # keypoints (in camera coordinated) to be represented in the joint frame
    kpts_wrt_pvt_frm_hom = np.dot(Tfm_mtx, quadruplet_kpts.T).T
    kpts_wrt_pvt_frm = kpts_wrt_pvt_frm_hom[:,:3] / kpts_wrt_pvt_frm_hom[:,[3]]

    # Compute axis and free limb vectors
    axis_limb_vec = kpts_wrt_pvt_frm[xy_axis_kpt_idx] - kpts_wrt_pvt_frm[pivot_kpt_idx]
    axis_limb_uvec = axis_limb_vec / np.linalg.norm(axis_limb_vec) # to be logged*
    free_limb_vec = kpts_wrt_pvt_frm[free_kpt_idx] - kpts_wrt_pvt_frm[pivot_kpt_idx]
    free_limb_uvec = free_limb_vec / np.linalg.norm(free_limb_vec) # to be logged**

    # Note the dot product of the free limb unit vector and x,y,z unit vectors
    # this may be thought of as the cosine of the euler angles
    free_cosines = np.dot(np.identity(3, dtype=np.float32), free_limb_uvec)

    # Assemble the rotation matrix that rotates axis_limb onto free_limb
    # and consequently derive the euler angles from it
    R = rotation_matrix(axis_limb_uvec, free_limb_uvec)
    euler_angles = rotation_matrix_to_xyz_euler_angles(R) # to be logged**

    return Tfm_mtx, free_limb_uvec, free_cosines, euler_angles


def numpy_bone_orientation_v1(quadruplet_kpts, n_samples, n_frames, z_axis_a_idx, z_axis_b_idx,
                              yx_axis_a_idx, yx_axis_b_idx, xy_axis_dir, xy_idx, yx_idx, z_idx=2,
                              pivot_kpt_idx=0, xy_axis_kpt_idx=1, yx_plane_kpt_idx=2, free_kpt_idx=3):
    '''Computes (vectorized operations) orientation per bone per batch'''
    # quadruplet_kpts --> (?,f,n,4)
    pivot_kpts = quadruplet_kpts[:,:,pivot_kpt_idx,:3]
    xy_axis_kpts = quadruplet_kpts[:,:,xy_axis_kpt_idx,:3]
    yx_plane_kpts = quadruplet_kpts[:,:,yx_plane_kpt_idx,:3]

    quadruplet_kpts_hom = np.ones((n_samples,n_frames,4,4), dtype=np.float32)
    quadruplet_kpts_hom[:,:,:,:3] = quadruplet_kpts

    eps_4 = 1e-4
    eps_6 = 1e-6
    neg_one = -1.
    pos_one = 1.
    zero = 0
    identity_3x3 = np.identity(3, dtype=np.float32)
    identity_4x4 = np.eye(4, dtype=np.float32)
    identity_3x3_matrices = np.repeat([identity_3x3.flatten()], n_samples*n_frames, axis=0)
    identity_3x3_matrices = np.reshape(identity_3x3_matrices, (n_samples,n_frames,3,3))
    assert(np.all(identity_3x3_matrices==identity_3x3)), "identity_3:\n{}".format(identity_3x3_matrices)
    translate_matrices = np.repeat([identity_4x4.flatten()], n_samples*n_frames, axis=0)
    translate_matrices = np.reshape(translate_matrices, (n_samples,n_frames,4,4))
    assert(np.all(translate_matrices==identity_4x4)), "identity_4:\n{}".format(translate_matrices[0,0])
    rotation_matrices = np.repeat([identity_4x4.flatten()], n_samples*n_frames, axis=0)
    rotation_matrices = np.reshape(rotation_matrices, (n_samples,n_frames,4,4))
    assert(np.all(rotation_matrices==identity_4x4)), "identity_4:\n{}".format(rotation_matrices)
    vector_matrices = np.zeros((n_samples,n_frames,3,3), dtype=np.float32)
    xyz_comparable_vecs = np.zeros((n_samples,n_frames,3,3), dtype=np.float32)
    xyz_axis_unit_vecs = np.zeros((n_samples,n_frames,3,3), dtype=np.float32)

    # Deduce translation vector from pivot joint
    translate_matrices[:,:,:3,3] = -pivot_kpts  # frame origin will be translated to pivot joint
    assert (are_translation_matrices(translate_matrices)), "May not be translation matrices\n{}".format(translate_matrices)

    # Get limb vectors from pairs of keypoint positions
    xy_axis_vecs = xy_axis_kpts - pivot_kpts # (?,f,3)
    yx_plane_vecs = yx_plane_kpts - pivot_kpts
    xyz_comparable_vecs[:,:,xy_idx] = xy_axis_vecs
    xyz_comparable_vecs[:,:,yx_idx] = yx_plane_vecs

    # Define new x-axis, y-axis, and z-axis
    xy_axis_uvecs = (xy_axis_vecs / np.linalg.norm(xy_axis_vecs, axis=2, keepdims=True)) * xy_axis_dir # (?,f,3)
    assert (not np.any(np.isnan(xy_axis_uvecs))), 'xy_axis_uvecs:\n{}'.format(xy_axis_uvecs)
    xyz_axis_unit_vecs[:,:,xy_idx] = xy_axis_uvecs
    #assert(is_unit_vector(xy_axis_uvec)), '|xy_axis_uvec|:{}'.format(np.linalg.norm(xy_axis_uvec))
    z_axis_vecs = numpy_cross(xyz_comparable_vecs[:,:,z_axis_a_idx], xyz_comparable_vecs[:,:,z_axis_b_idx])
    xyz_comparable_vecs[:,:,z_idx] = z_axis_vecs
    z_axis_uvecs = z_axis_vecs / np.linalg.norm(z_axis_vecs, axis=2, keepdims=True)
    assert (not np.any(np.isnan(z_axis_uvecs))), 'z_axis_uvecs:\n{}'.format(z_axis_uvecs)
    xyz_axis_unit_vecs[:,:,z_idx] = z_axis_uvecs
    #assert(is_unit_vector(z_axis_uvec)), '|z_axis_uvec|:{}'.format(np.linalg.norm(z_axis_uvec))
    yx_axis_vecs = numpy_cross(xyz_comparable_vecs[:,:,yx_axis_a_idx], xyz_comparable_vecs[:,:,yx_axis_b_idx])
    yx_axis_uvecs = yx_axis_vecs / np.linalg.norm(yx_axis_vecs, axis=2, keepdims=True)
    assert (not np.any(np.isnan(yx_axis_uvecs))), 'yx_axis_uvecs:\n{}'.format(yx_axis_uvecs)
    xyz_axis_unit_vecs[:,:,yx_idx] = yx_axis_uvecs
    #assert(is_unit_vector(y_axis_uvec)), '|y_axis_uvec|:{}'.format(np.linalg.norm(y_axis_uvec))

    # Derive frame rotation matrix from unit vec axis
    #assert(is_rotation_matrix(R)), 'R not orthogonal:\n{}\n{}'.format(R, np.dot(R.T, R))
    rotation_matrices[:,:,:3,:3] = xyz_axis_unit_vecs
    transform_matrices = numpy_matmul(rotation_matrices, translate_matrices, n=4)

    # Using frame rotation matrix, transform position of non-origin translated
    # keypoints (in camera coordinated) to be represented in the joint frame
    kpts_wrt_pvt_frms_hom = numpy_matdot(transform_matrices, quadruplet_kpts_hom)
    kpts_wrt_pvt_frms = kpts_wrt_pvt_frms_hom[:,:,:,:3] / kpts_wrt_pvt_frms_hom[:,:,:,[3]] #*** nan trouble
    assert (not np.any(np.isnan(kpts_wrt_pvt_frms))), 'kpts_wrt_pvt_frms:\n{}'.format(kpts_wrt_pvt_frms)

    # Compute axis and free limb vectors
    axis_limb_vecs = kpts_wrt_pvt_frms[:,:,xy_axis_kpt_idx] - kpts_wrt_pvt_frms[:,:,pivot_kpt_idx] # (?,f,3)
    axis_limb_uvecs = axis_limb_vecs / np.linalg.norm(axis_limb_vecs, axis=2, keepdims=True) #*** nan trouble
    assert (not np.any(np.isnan(axis_limb_uvecs))), 'axis_limb_uvecs:\n{}'.format(axis_limb_uvecs)
    free_limb_vecs = kpts_wrt_pvt_frms[:,:,free_kpt_idx] - kpts_wrt_pvt_frms[:,:,pivot_kpt_idx]
    free_limb_uvecs = free_limb_vecs / np.linalg.norm(free_limb_vecs, axis=2, keepdims=True) #*** nan trouble
    assert (not np.any(np.isnan(free_limb_uvecs))), 'free_limb_uvecs:\n{}'.format(free_limb_uvecs)

    # Compile the rotation matrix that rotates axis_limb onto free_limb
    cosines = numpy_vecdot(axis_limb_uvecs, free_limb_uvecs, vsize=3) # cosine of angle --> (?,f)
    # implies the angle is +/-180" (a flip) and hence a case where the rotation matrix is not 1-1
    # for our application clipping the angle at +-179" may be a sufficient work-around this case
    cosines = np.where(cosines==neg_one, cosines+eps_4, cosines) # (?,f) : if c==-1 --> c += 1e-4 (ideally 0.00015)
    cosines = np.expand_dims(np.expand_dims(cosines, axis=2), axis=3) # (?,f)->(?,f,1)->(?,f,1,1)
    const = pos_one / (pos_one + cosines) # (?,f) #*** nan trouble
    assert (not np.any(np.isnan(const))), 'const:\n{}'.format(const)
    vector = numpy_cross(axis_limb_uvecs, free_limb_uvecs) # (?,f,3)
    vector_matrices[:,:,0,1] = -vector[:,:,2]
    vector_matrices[:,:,0,2] =  vector[:,:,1]
    vector_matrices[:,:,1,0] =  vector[:,:,2]
    vector_matrices[:,:,1,2] = -vector[:,:,0]
    vector_matrices[:,:,2,0] = -vector[:,:,1]
    vector_matrices[:,:,2,1] =  vector[:,:,0]
    assert (not np.any(np.isnan(vector_matrices))), 'vector_matrices:\n{}'.format(vector_matrices)
    vector_x_vector_matrices = numpy_matmul(vector_matrices, vector_matrices)
    assert (not np.any(np.isnan(vector_x_vector_matrices))), 'vector_x_vector_matrices:\n{}'.format(vector_x_vector_matrices)
    vec_rotate_matrices = identity_3x3_matrices + vector_matrices + vector_x_vector_matrices*const # (?,f,3,3)
    assert (not np.any(np.isnan(vec_rotate_matrices))), 'vec_rotate_matrices:\n{}'.format(vec_rotate_matrices)
    sy_tensor = np.sqrt(vec_rotate_matrices[:,:,0,0]*vec_rotate_matrices[:,:,0,0] +
                           vec_rotate_matrices[:,:,1,0]*vec_rotate_matrices[:,:,1,0])
    assert (not np.any(np.isnan(sy_tensor))), 'sy_tensor:\n{}'.format(sy_tensor)
    is_singular = sy_tensor<eps_6

    singular_x = numpy_atan2(-vec_rotate_matrices[:,:,1,2], vec_rotate_matrices[:,:,1,1])
    assert (not np.any(np.isnan(singular_x))), 'singular_x:\n{}'.format(singular_x)
    not_singular_x = numpy_atan2(vec_rotate_matrices[:,:,2,1], vec_rotate_matrices[:,:,2,2])
    assert (not np.any(np.isnan(not_singular_x))), 'not_singular_x:\n{}'.format(not_singular_x)
    not_singular_z = numpy_atan2(vec_rotate_matrices[:,:,1,0], vec_rotate_matrices[:,:,0,0])
    assert (not np.any(np.isnan(not_singular_z))), 'not_singular_z:\n{}'.format(not_singular_z)

    # Compute euler angles from rotation matrices
    y_euler_angle = numpy_atan2(-vec_rotate_matrices[:,:,2,0], sy_tensor)
    assert (not np.any(np.isnan(y_euler_angle))), 'y_euler_angle:\n{}'.format(y_euler_angle)
    x_euler_angle = np.where(is_singular, singular_x, not_singular_x)
    assert (not np.any(np.isnan(x_euler_angle))), 'x_euler_angle:\n{}'.format(x_euler_angle)
    z_euler_angle = np.where(is_singular, zero, not_singular_z)
    assert (not np.any(np.isnan(z_euler_angle))), 'z_euler_angle:\n{}'.format(z_euler_angle)

    return transform_matrices, free_limb_uvecs, [x_euler_angle, y_euler_angle, z_euler_angle]


def numpy_bone_orientation_v2(quadruplet_kpts, n_samples, n_frames, z_axis_a_idx, z_axis_b_idx,
                              yx_axis_a_idx, yx_axis_b_idx, xy_axis_dir, xy_idx, yx_idx, z_idx=2,
                              pivot_kpt_idx=0, xy_axis_kpt_idx=1, yx_plane_kpt_idx=2, free_kpt_idx=3):
    '''Computes (vectorized operations) orientation per bone per batch,
        written to mimic semantic that supports backpropagation in pytorch
    '''
    # 0-dimension constants
    eps_4 = 1e-4
    eps_6 = 1e-6
    neg_one = -1.
    pos_one = 1.
    zero = 0
    # n-dimension single-value constants
    zero_tsr_1 = np.zeros((n_samples,n_frames,1), dtype=np.float32) # (?,f,1)
    zero_tsr_4 = np.zeros((n_samples,n_frames,4), dtype=np.float32) # (?,f,4)
    zero_tsr_3x1 = np.zeros((n_samples,n_frames,3,1), dtype=np.float32) # (?,f,3,1)
    one_tsr_4x1 = np.ones((n_samples,n_frames,4,1), dtype=np.float32) # (?,f,4,1)
    one_tsr_1 = np.ones((n_samples,n_frames,1), dtype=np.float32) # (?,f,1)
    rotate_tsr_1x4 = np.stack((zero_tsr_1, zero_tsr_1, zero_tsr_1, one_tsr_1), axis=3) # (?,f,1)*4->(?,f,"1,4)
    # n-dimension nxn matrices constants
    identity_3x3 = np.identity(3, dtype=np.float32)
    identity_3x3_matrices = np.repeat([identity_3x3.flatten()], n_samples*n_frames, axis=0)
    identity_3x3_matrices = np.reshape(identity_3x3_matrices, (n_samples,n_frames,3,3))
    assert(np.all(identity_3x3_matrices==identity_3x3)), "identity_3:\n{}".format(identity_3x3_matrices)
    identity_4x4 = np.eye(4, dtype=np.float32)
    identity_4x4_matrices = np.repeat([identity_4x4.flatten()], n_samples*n_frames, axis=0)
    identity_4x4_matrices = np.reshape(identity_4x4_matrices, (n_samples,n_frames,4,4))
    assert(np.all(identity_4x4_matrices==identity_4x4)), "identity_4:\n{}".format(identity_4x4_matrices[0,0])

    # quadruplet_kpts --> (?,f,n,4)
    quadruplet_kpts_hom = np.concatenate((quadruplet_kpts, one_tsr_4x1), axis=3) # (?,f,4,3)+(?,f,4,1)->(?,f,4,4)
    pivot_kpts = quadruplet_kpts[:,:,pivot_kpt_idx]
    xy_axis_kpts = quadruplet_kpts[:,:,xy_axis_kpt_idx]
    yx_plane_kpts = quadruplet_kpts[:,:,yx_plane_kpt_idx]

    # Deduce translation vector from pivot joint
    trans_tsr_4 = np.concatenate((-pivot_kpts, zero_tsr_1), axis=2) # (?,f,3)+(?,f,1)->(?,f,4)
    trans_tsr_4x4 = np.stack((zero_tsr_4, zero_tsr_4, zero_tsr_4, trans_tsr_4), axis=3) # (?,f,4)*4->(?,f,"4,4)
    translate_matrices = identity_4x4_matrices + trans_tsr_4x4 # (?,f,4,4)
    assert (are_translation_matrices(translate_matrices)), "May not be translation matrices\n{}".format(translate_matrices)

    # Get limb vectors from pairs of keypoint positions
    xy_axis_vecs = xy_axis_kpts - pivot_kpts # (?,f,3)
    yx_plane_vecs = yx_plane_kpts - pivot_kpts
    if xy_idx==0:
        xy_comparable_vecs = np.stack([xy_axis_vecs, yx_plane_vecs], axis=2)
    else: xy_comparable_vecs = np.stack([yx_plane_vecs, xy_axis_vecs], axis=2)

    # Define new x-axis, y-axis, and z-axis
    xy_axis_uvecs = (xy_axis_vecs / np.linalg.norm(xy_axis_vecs, axis=2, keepdims=True)) * xy_axis_dir # (?,f,3)
    assert (not np.any(np.isnan(xy_axis_uvecs))), 'xy_axis_uvecs:\n{}'.format(xy_axis_uvecs)
    z_axis_vecs = numpy_cross(xy_comparable_vecs[:,:,z_axis_a_idx], xy_comparable_vecs[:,:,z_axis_b_idx])
    z_axis_1_vecs = np.expand_dims(z_axis_vecs, axis=2) # (?,f,3)->(?,f,1,3)
    xyz_comparable_vecs = np.concatenate((xy_comparable_vecs, z_axis_1_vecs), axis=2) # (?,f,2,3)+(?,f,1,3)->(?,f,3,3)
    z_axis_uvecs = z_axis_vecs / np.linalg.norm(z_axis_vecs, axis=2, keepdims=True)
    assert (not np.any(np.isnan(z_axis_uvecs))), 'z_axis_uvecs:\n{}'.format(z_axis_uvecs)
    yx_axis_vecs = numpy_cross(xyz_comparable_vecs[:,:,yx_axis_a_idx], xyz_comparable_vecs[:,:,yx_axis_b_idx])
    yx_axis_uvecs = yx_axis_vecs / np.linalg.norm(yx_axis_vecs, axis=2, keepdims=True)
    assert (not np.any(np.isnan(yx_axis_uvecs))), 'yx_axis_uvecs:\n{}'.format(yx_axis_uvecs)

    # Derive frame rotation matrix from unit vec axis
    if xy_idx==0:
        xyz_axis_unit_vecs = np.stack((xy_axis_uvecs, yx_axis_uvecs, z_axis_uvecs), axis=2) # (?,f,3)*3->(?,f,"3,3)
    else: xyz_axis_unit_vecs = np.stack((yx_axis_uvecs, xy_axis_uvecs, z_axis_uvecs), axis=2) # (?,f,3)*3->(?,f,"3,3)
    #assert(is_rotation_matrix(R)), 'R not orthogonal:\n{}\n{}'.format(R, np.dot(R.T, R))
    rotate_tsr_3x4 = np.concatenate((xyz_axis_unit_vecs, zero_tsr_3x1), axis=3) # (?,f,3,3)+(?,f,3,1)->(?,f,3,4)
    rotation_matrices = np.concatenate((rotate_tsr_3x4, rotate_tsr_1x4), axis=2) # (?,f,3,4)+(?,f,1,4)->(?,f,4,4)
    transform_matrices = numpy_matmul(rotation_matrices, translate_matrices, n=4) # (?,f,4,4)

    # Using frame rotation matrix, transform position of non-origin translated
    # keypoints (in camera coordinated) to be represented in the joint frame
    kpts_wrt_pvt_frms_hom = numpy_matdot(transform_matrices, quadruplet_kpts_hom)
    kpts_wrt_pvt_frms = kpts_wrt_pvt_frms_hom[:,:,:,:3] / kpts_wrt_pvt_frms_hom[:,:,:,[3]] #*** nan trouble
    assert (not np.any(np.isnan(kpts_wrt_pvt_frms))), 'kpts_wrt_pvt_frms:\n{}'.format(kpts_wrt_pvt_frms)

    # Compute axis and free limb vectors
    axis_limb_vecs = kpts_wrt_pvt_frms[:,:,xy_axis_kpt_idx] - kpts_wrt_pvt_frms[:,:,pivot_kpt_idx] # (?,f,3)
    axis_limb_uvecs = axis_limb_vecs / np.linalg.norm(axis_limb_vecs, axis=2, keepdims=True) #*** nan trouble
    assert (not np.any(np.isnan(axis_limb_uvecs))), 'axis_limb_uvecs:\n{}'.format(axis_limb_uvecs)
    free_limb_vecs = kpts_wrt_pvt_frms[:,:,free_kpt_idx] - kpts_wrt_pvt_frms[:,:,pivot_kpt_idx]
    free_limb_uvecs = free_limb_vecs / np.linalg.norm(free_limb_vecs, axis=2, keepdims=True) #*** nan trouble
    assert (not np.any(np.isnan(free_limb_uvecs))), 'free_limb_uvecs:\n{}'.format(free_limb_uvecs)

    # Compile the rotation matrix that rotates axis_limb onto free_limb
    cosines = numpy_vecdot(axis_limb_uvecs, free_limb_uvecs, vsize=3) # cosine of angle --> (?,f)
    # implies the angle is +/-180" (a flip) and hence a case where the rotation matrix is not 1-1
    # for our application clipping the angle at +-179" may be a sufficient work-around this case
    cosines = np.where(cosines==neg_one, cosines+eps_4, cosines) # (?,f) : if c==-1 --> c += 1e-4 (ideally 0.00015)
    cosines = np.expand_dims(np.expand_dims(cosines, axis=2), axis=3) # (?,f)->(?,f,1)->(?,f,1,1)
    const = pos_one / (pos_one + cosines) # (?,f) #*** nan trouble
    assert (not np.any(np.isnan(const))), 'const:\n{}'.format(const)
    vector = numpy_cross(axis_limb_uvecs, free_limb_uvecs) # (?,f,3)
    vector_row0_1x3 = np.stack((zero_tsr_1, -vector[:,:,[2]], vector[:,:,[1]]), axis=3) # (?,f,1)*3->(?,f,1,"3)
    vector_row1_1x3 = np.stack((vector[:,:,[2]], zero_tsr_1, -vector[:,:,[0]]), axis=3) # (?,f,1)*3->(?,f,1,"3)
    vector_row2_1x3 = np.stack((-vector[:,:,[1]], vector[:,:,[0]], zero_tsr_1), axis=3) # (?,f,1)*3->(?,f,1,"3)
    vector_matrices = np.concatenate((vector_row0_1x3, vector_row1_1x3, vector_row2_1x3), axis=2) # (?,f,3,3)
    assert (not np.any(np.isnan(vector_matrices))), 'vector_matrices:\n{}'.format(vector_matrices)
    vector_x_vector_matrices = numpy_matmul(vector_matrices, vector_matrices)
    assert (not np.any(np.isnan(vector_x_vector_matrices))), 'vector_x_vector_matrices:\n{}'.format(vector_x_vector_matrices)
    vec_rotate_matrices = identity_3x3_matrices + vector_matrices + vector_x_vector_matrices*const # (?,f,3,3)
    assert (not np.any(np.isnan(vec_rotate_matrices))), 'vec_rotate_matrices:\n{}'.format(vec_rotate_matrices)
    sy_tensor = np.sqrt(vec_rotate_matrices[:,:,0,0]*vec_rotate_matrices[:,:,0,0] +
                        vec_rotate_matrices[:,:,1,0]*vec_rotate_matrices[:,:,1,0])
    assert (not np.any(np.isnan(sy_tensor))), 'sy_tensor:\n{}'.format(sy_tensor)

    # Compute euler angles from rotation matrices
    is_singular = sy_tensor<eps_6
    singular_x = numpy_atan2(-vec_rotate_matrices[:,:,1,2], vec_rotate_matrices[:,:,1,1])
    assert (not np.any(np.isnan(singular_x))), 'singular_x:\n{}'.format(singular_x)
    not_singular_x = numpy_atan2(vec_rotate_matrices[:,:,2,1], vec_rotate_matrices[:,:,2,2])
    assert (not np.any(np.isnan(not_singular_x))), 'not_singular_x:\n{}'.format(not_singular_x)
    not_singular_z = numpy_atan2(vec_rotate_matrices[:,:,1,0], vec_rotate_matrices[:,:,0,0])
    assert (not np.any(np.isnan(not_singular_z))), 'not_singular_z:\n{}'.format(not_singular_z)
    y_euler_angle = numpy_atan2(-vec_rotate_matrices[:,:,2,0], sy_tensor)
    assert (not np.any(np.isnan(y_euler_angle))), 'y_euler_angle:\n{}'.format(y_euler_angle)
    x_euler_angle = np.where(is_singular, singular_x, not_singular_x)
    assert (not np.any(np.isnan(x_euler_angle))), 'x_euler_angle:\n{}'.format(x_euler_angle)
    z_euler_angle = np.where(is_singular, zero, not_singular_z)
    assert (not np.any(np.isnan(z_euler_angle))), 'z_euler_angle:\n{}'.format(z_euler_angle)

    return transform_matrices, free_limb_uvecs, (x_euler_angle, y_euler_angle, z_euler_angle)


def numpy_bone_orientation_v3(quadruplet_kpts, n_samples, n_frames, xy_axis_dirs, z_axis_a_idxs, z_axis_b_idxs,
                              yx_axis_a_idxs, yx_axis_b_idxs, xy_idxs, yx_idxs, z_idx=2, n_joints=12,
                              pivot_kpt_idx=0, xy_axis_kpt_idx=1, yx_plane_kpt_idx=2, free_kpt_idx=3, test_mode=True):
    '''Computes (vectorized operations) bone orientation per batch
       quadruplet_kpts->(?,f,12,4,3), xy_axis_dir->(1,1,12,1)
       z_axis_a&b_idxs->(12,), yx_axis_a&b_idxs->(12,), xy&yx_idxs->(12,)
    '''
    pivot_kpts = quadruplet_kpts[:,:,:,pivot_kpt_idx,:3] # (?,f,j,3)
    xy_axis_kpts = quadruplet_kpts[:,:,:,xy_axis_kpt_idx,:3] # (?,f,j,3)
    yx_plane_kpts = quadruplet_kpts[:,:,:,yx_plane_kpt_idx,:3] # (?,f,j,3)

    quadruplet_kpts_hom = np.ones((n_samples,n_frames,n_joints,4,4), dtype=np.float32)
    quadruplet_kpts_hom[:,:,:,:,:3] = quadruplet_kpts # (?,f,j,4,4)

    jnt_idxs = np.arange(n_joints)
    identity_3x3 = np.eye(3, dtype=np.float32)
    identity_4x4 = np.eye(4, dtype=np.float32)
    identity_3x3_matrices = np.repeat([identity_3x3.flatten()], n_samples*n_frames*n_joints, axis=0)
    identity_3x3_matrices = np.reshape(identity_3x3_matrices, (n_samples,n_frames,n_joints,3,3))
    assert (np.all(identity_3x3_matrices==identity_3x3)), "identity_3:\n{}".format(identity_3x3_matrices)
    translate_matrices = np.repeat([identity_4x4.flatten()], n_samples*n_frames*n_joints, axis=0)
    translate_matrices = np.reshape(translate_matrices, (n_samples,n_frames,n_joints,4,4))
    assert (np.all(translate_matrices==identity_4x4)), "identity_4:\n{}".format(translate_matrices[0,0])
    rotation_matrices = np.repeat([identity_4x4.flatten()], n_samples*n_frames*n_joints, axis=0)
    rotation_matrices = np.reshape(rotation_matrices, (n_samples,n_frames,n_joints,4,4))
    assert (np.all(rotation_matrices==identity_4x4)), "identity_4:\n{}".format(rotation_matrices)
    xyz_comparable_vecs = np.zeros((n_samples,n_frames,n_joints,3,3), dtype=np.float32)
    xyz_axis_unit_vecs = np.zeros((n_samples,n_frames,n_joints,3,3), dtype=np.float32)

    # Deduce translation vector from pivot joint
    translate_matrices[:,:,:,:3,3] = -pivot_kpts  # frame origin will be translated to pivot joint
    assert (are_translation_matrices(translate_matrices)), "May contain non-translation matrix\n{}".format(translate_matrices)

    # Get limb vectors from pairs of keypoint positions
    xy_axis_vecs = xy_axis_kpts - pivot_kpts # (?,f,j,3)
    yx_plane_vecs = yx_plane_kpts - pivot_kpts # (?,f,j,3)
    xyz_comparable_vecs[:,:,jnt_idxs,xy_idxs] = xy_axis_vecs
    xyz_comparable_vecs[:,:,jnt_idxs,yx_idxs] = yx_plane_vecs

    # Define new x-axis, y-axis, and z-axis
    xy_axis_uvecs = (xy_axis_vecs / np.linalg.norm(xy_axis_vecs, axis=3, keepdims=True)) * xy_axis_dirs # (?,f,j,3)
    assert (not np.any(np.isnan(xy_axis_uvecs))), 'xy_axis_uvecs:\n{}'.format(xy_axis_uvecs)
    xyz_axis_unit_vecs[:,:,jnt_idxs,xy_idxs] = xy_axis_uvecs
    assert (are_unit_vectors(xy_axis_uvecs)), '|xy_axis_uvecs|:{}'.format(np.linalg.norm(xy_axis_uvecs))
    z_axis_vecs = numpy_cross(xyz_comparable_vecs[:,:,jnt_idxs,z_axis_a_idxs],
                              xyz_comparable_vecs[:,:,jnt_idxs,z_axis_b_idxs]) # (?,f,j,3)
    xyz_comparable_vecs[:,:,:,z_idx] = z_axis_vecs
    z_axis_uvecs = z_axis_vecs / np.linalg.norm(z_axis_vecs, axis=3, keepdims=True) # (?,f,j,3)
    assert (not np.any(np.isnan(z_axis_uvecs))), 'z_axis_uvecs:\n{}'.format(z_axis_uvecs)
    xyz_axis_unit_vecs[:,:,:,z_idx] = z_axis_uvecs
    assert (are_unit_vectors(z_axis_uvecs)), '|z_axis_uvecs|:{}'.format(np.linalg.norm(z_axis_uvecs))
    yx_axis_vecs = numpy_cross(xyz_comparable_vecs[:,:,jnt_idxs,yx_axis_a_idxs],
                               xyz_comparable_vecs[:,:,jnt_idxs,yx_axis_b_idxs]) # (?,f,j,3)
    yx_axis_uvecs = yx_axis_vecs / np.linalg.norm(yx_axis_vecs, axis=3, keepdims=True) # (?,f,j,3)
    assert (not np.any(np.isnan(yx_axis_uvecs))), 'yx_axis_uvecs:\n{}'.format(yx_axis_uvecs)
    assert (are_unit_vectors(yx_axis_uvecs)), '|yx_axis_uvecs|:{}'.format(np.linalg.norm(yx_axis_uvecs))
    xyz_axis_unit_vecs[:,:,jnt_idxs,yx_idxs] = yx_axis_uvecs

    # Derive frame rotation matrix from unit vec axis
    rotation_matrices[:,:,:,:3,:3] = xyz_axis_unit_vecs # (?,f,j,4,4)
    #assert(is_rotation_matrix(R)), 'R not orthogonal:\n{}\n{}'.format(R, np.dot(R.T, R))
    transform_matrices = numpy_matmul(rotation_matrices, translate_matrices, n=4)

    # Using frame rotation matrix, transform position of non-origin translated
    # keypoints (in camera coordinated) to be represented in the joint frame
    kpts_wrt_pvt_frms_hom = numpy_matdot(transform_matrices, quadruplet_kpts_hom)
    kpts_wrt_pvt_frms = kpts_wrt_pvt_frms_hom[:,:,:,:,:3] / kpts_wrt_pvt_frms_hom[:,:,:,:,[3]] #*** nan trouble
    assert (not np.any(np.isnan(kpts_wrt_pvt_frms))), 'kpts_wrt_pvt_frms:\n{}'.format(kpts_wrt_pvt_frms)

    # Compute free and axis limb vectors
    free_limb_vecs = kpts_wrt_pvt_frms[:,:,:,free_kpt_idx] - kpts_wrt_pvt_frms[:,:,:,pivot_kpt_idx] # (?,f,j,3)
    free_limb_uvecs = free_limb_vecs / np.linalg.norm(free_limb_vecs, axis=3, keepdims=True) #*** nan trouble
    assert (not np.any(np.isnan(free_limb_uvecs))), 'free_limb_uvecs:\n{}'.format(free_limb_uvecs)
    axis_limb_vecs = kpts_wrt_pvt_frms[:,:,:,xy_axis_kpt_idx] - kpts_wrt_pvt_frms[:,:,:,pivot_kpt_idx] # (?,f,j,3)
    axis_limb_uvecs = axis_limb_vecs / np.linalg.norm(axis_limb_vecs, axis=3, keepdims=True) #*** nan trouble
    assert (not np.any(np.isnan(axis_limb_uvecs))), 'axis_limb_uvecs:\n{}'.format(axis_limb_uvecs)

    # Note the dot product of the free limb unit vector and x,y,z unit vectors (aka: cosine of the euler angles)
    free_limb_cosines = numpy_matdot(identity_3x3_matrices, np.expand_dims(free_limb_uvecs, axis=3)) # (?.f,j,1,3)

    zero_t = 0
    eps_4_t = 1e-4
    eps_6_t = 1e-6
    pos_one_t = 1.
    neg_one_t = -1.
    vector_matrices = np.zeros((n_samples,n_frames,n_joints,3,3), dtype=np.float32)

    # Compile the rotation matrix that rotates axis_limb onto free_limb
    cosines = numpy_vecdot(axis_limb_uvecs, free_limb_uvecs, vsize=3) # cosine of angle --> (?,f,j)
    # implies the angle is +/-180" (a flip) and hence a case where the rotation matrix is not 1-1
    # for our application clipping the angle at +-179" may be a sufficient work-around this case
    cosines = np.where(cosines==neg_one_t, cosines+eps_4_t, cosines) # (?,f,j) : if c==-1 --> c += 1e-4 (ideally 0.00015)
    cosines = np.expand_dims(np.expand_dims(cosines, axis=3), axis=4) # (?,f,j)->(?,f,j,1)->(?,f,j,1,1)
    const = pos_one_t / (pos_one_t + cosines) # (?,f,j,1,1) #*** nan trouble
    assert (not np.any(np.isnan(const))), 'const:\n{}'.format(const)
    vector = numpy_cross(axis_limb_uvecs, free_limb_uvecs) # (?,f,j,3)
    vector_matrices[:,:,:,0,1] = -vector[:,:,:,2]
    vector_matrices[:,:,:,0,2] =  vector[:,:,:,1]
    vector_matrices[:,:,:,1,0] =  vector[:,:,:,2]
    vector_matrices[:,:,:,1,2] = -vector[:,:,:,0]
    vector_matrices[:,:,:,2,0] = -vector[:,:,:,1]
    vector_matrices[:,:,:,2,1] =  vector[:,:,:,0]
    assert (not np.any(np.isnan(vector_matrices))), 'vector_matrices:\n{}'.format(vector_matrices)
    vector_x_vector_matrices = numpy_matmul(vector_matrices, vector_matrices) # (?,f,j,3,3)
    assert (not np.any(np.isnan(vector_x_vector_matrices))), 'vector_x_vector_matrices:\n{}'.format(vector_x_vector_matrices)
    vec_rotate_matrices = identity_3x3_matrices + vector_matrices + vector_x_vector_matrices*const # (?,f,j,3,3)
    assert (not np.any(np.isnan(vec_rotate_matrices))), 'vec_rotate_matrices:\n{}'.format(vec_rotate_matrices)
    sy_tensor = np.sqrt(vec_rotate_matrices[:,:,:,0,0]*vec_rotate_matrices[:,:,:,0,0] +
                        vec_rotate_matrices[:,:,:,1,0]*vec_rotate_matrices[:,:,:,1,0]) # (?,f,j)
    assert (not np.any(np.isnan(sy_tensor))), 'sy_tensor:\n{}'.format(sy_tensor)
    is_singular = sy_tensor<eps_6_t # (?,f,j)

    singular_x = numpy_atan2(-vec_rotate_matrices[:,:,:,1,2], vec_rotate_matrices[:,:,:,1,1]) # (?,f,j)
    assert (not np.any(np.isnan(singular_x))), 'singular_x:\n{}'.format(singular_x)
    not_singular_x = numpy_atan2(vec_rotate_matrices[:,:,:,2,1], vec_rotate_matrices[:,:,:,2,2]) # (?,f,j)
    assert (not np.any(np.isnan(not_singular_x))), 'not_singular_x:\n{}'.format(not_singular_x)
    not_singular_z = numpy_atan2(vec_rotate_matrices[:,:,:,1,0], vec_rotate_matrices[:,:,:,0,0]) # (?,f,j)
    assert (not np.any(np.isnan(not_singular_z))), 'not_singular_z:\n{}'.format(not_singular_z)

    # Compute euler angles from rotation matrices
    y_euler_angles = numpy_atan2(-vec_rotate_matrices[:,:,:,2,0], sy_tensor) # (?,f,j)
    assert (not np.any(np.isnan(y_euler_angles))), 'y_euler_angle:\n{}'.format(y_euler_angles)
    x_euler_angles = np.where(is_singular, singular_x, not_singular_x) # (?,f,j)
    assert (not np.any(np.isnan(x_euler_angles))), 'x_euler_angle:\n{}'.format(x_euler_angles)
    z_euler_angles = np.where(is_singular, zero_t, not_singular_z) # (?,f,j)
    assert (not np.any(np.isnan(z_euler_angles))), 'z_euler_angle:\n{}'.format(z_euler_angles)

    return transform_matrices, free_limb_uvecs, free_limb_cosines, [x_euler_angles, y_euler_angles, z_euler_angles]


def save_jmc_priors(fb_orientation_vecs, jme_ordered_tags, n_fb_jnts, augment, group_joints,
                    supr_subset_tag, frm_rate_tag, rbo_type, jmc_joint_config, from_torch=True):
    op_tag = '_tch' if from_torch else '_npy'
    aug_tag = 'wxaug' if augment else 'nouag'
    grp_tag = 'grpjnt' if group_joints else 'perjnt'
    file_path = os.path.join('priors', supr_subset_tag, 'properties', 'joint_orients{}_{}_{}_{}_{}{}.pickle'.
                             format(frm_rate_tag, rbo_type, aug_tag, n_fb_jnts, grp_tag, op_tag))
    jme_prior_src = {'joint_align_config':jmc_joint_config, 'keypoint_indexes':KPT_2_IDX,
                     'joint_order':jme_ordered_tags, 'group':group_joints, 'augment':augment}
    for joint_id, joint_orient_vecs in fb_orientation_vecs.items():
        jme_prior_src[joint_id] = np.concatenate(joint_orient_vecs, axis=0)
    pickle_write(jme_prior_src, file_path)


def save_bone_prior_properties(proportions_metadata, bpe_ordered_tags, augment,
                               supr_subset_tag, frm_rate_tag, n_ratios, from_torch=True):
    properties_dir = os.path.join('priors', supr_subset_tag, 'properties')
    op_tag = '_tch' if from_torch else '_npy'
    aug_tag = 'wxaug' if augment else 'nouag'
    if len(proportions_metadata['bone_lengths'])>0:
        bone_len = np.concatenate(proportions_metadata['bone_lengths'])
        file_path = os.path.join(properties_dir, 'bone_lengths{}_16_m.npy'.format(frm_rate_tag))
        np.save(file_path, bone_len)
        print('avg bone lengths: {}'.format(np.mean(bone_len, axis=0, keepdims=False)))
    if len(proportions_metadata['bone_symms'])>0:
        bone_symm = np.concatenate(proportions_metadata['bone_symms'])
        file_path = os.path.join(properties_dir, 'bone_symms{}_6_m.npy'.format(frm_rate_tag))
        np.save(file_path, bone_symm)
        print('avg bone symmetry fractions: {}'.format(np.mean(bone_symm, axis=0, keepdims=False)))
    if len(proportions_metadata['bone_ratios'])>0:
        bone_ratio = np.concatenate(proportions_metadata['bone_ratios'])
        assert (n_ratios==bone_ratio.shape[-1]), '{} vs. {}'.format(n_ratios, bone_ratio.shape[-1])
        bpe_prior_src = {'bone_ratios':bone_ratio,
                         'keypoint_indexes':KPT_2_IDX, 'bone_kpt_pairs':BONE_KPT_PAIRS,
                         'ratio_order':bpe_ordered_tags, 'augment':augment,
                         'rgt_sym_bones':RGT_SYM_BONES, 'lft_sym_bones':LFT_SYM_BONES}
        file_path = os.path.join(properties_dir, 'bone_ratios_{}_{}{}_m{}.pickle'.
                                 format(aug_tag, n_ratios, frm_rate_tag, op_tag))
        pickle_write(bpe_prior_src, file_path)
        #print('avg bone proportion ratio:\n{}'.format(np.mean(bone_ratio, axis=0, keepdims=False)))


def np_orthographic_projection(poses_3d):
    assert (poses_3d.shape[1:] == (1, 17, 3)), 'poses_3d.shape:{}'.format(poses_3d.shape) # (?,f,17,3)
    #x_lft = np.amin(poses_3d[:,:,:,0], axis=-1, keepdims=True) # (?,f,17) -> (?,f,1) #*!
    #x_rgt = np.amax(poses_3d[:,:,:,0], axis=-1, keepdims=True) # (?,f,17) -> (?,f,1) #*!
    #y_top = np.amin(poses_3d[:,:,:,1], axis=-1, keepdims=True) # (?,f,17) -> (?,f,1) #*!
    #y_bot = np.amax(poses_3d[:,:,:,1], axis=-1, keepdims=True) # (?,f,17) -> (?,f,1) #*!
    #z_nea = np.amin(poses_3d[:,:,:,2], axis=-1, keepdims=True) # (?,f,17) -> (?,f,1) #*!
    #z_far = np.amax(poses_3d[:,:,:,2], axis=-1, keepdims=True) # (?,f,17) -> (?,f,1) #*!
    projection_mtx = np.zeros(poses_3d.shape[:-1]+(4,4), dtype=np.float32) # (?,f,17,4,4)
    projection_mtx[:,:,:,0,0] =  1 # 2 / (x_rgt - x_lft) #*!
    projection_mtx[:,:,:,1,1] =  1 # 2 / (y_top - y_bot) #*!
    #projection_mtx[:,:,:,2,2] =  -2 / (z_far - z_nea) #*!
    projection_mtx[:,:,:,3,3] =  1
    #projection_mtx[:,:,:,0,3] = -(x_rgt + x_lft) / (x_rgt - x_lft) #*!
    #projection_mtx[:,:,:,1,3] = -(y_top + y_bot) / (y_top - y_bot) #*!
    #projection_mtx[:,:,:,2,3] = -(z_far + z_nea) / (z_far - z_nea) #*!
    homg_ones = np.ones(poses_3d.shape[:-1]+(1,), dtype=np.float32) # (?,f,17,1)
    homg_3d_poses = np.concatenate([poses_3d, homg_ones], axis=-1) # (?,f,17,4)
    homg_3d_poses = np.expand_dims(homg_3d_poses, axis=4) # (?,f,17,4,1)
    assert (homg_3d_poses.shape[1:] == (1, 17, 4, 1)), 'homg_3d_poses.shape:{}'.format(homg_3d_poses.shape) # (?,f,17,3) # (?,f,17,4,1)
    proj_2d_poses = np.matmul(projection_mtx, homg_3d_poses)
    assert (proj_2d_poses.shape[1:] == (1, 17, 4, 1)), 'proj_2d_poses.shape:{}'.format(proj_2d_poses.shape) # (?,f,17,3) # (?,f,17,4,1)
    assert (np.all(np.isclose(proj_2d_poses[:,:,:,2,0], 0, atol=1e-05))) #*!
    assert (np.all(np.isclose(proj_2d_poses[:,:,:,3,0], 1, atol=1e-05)))
    return proj_2d_poses[:,:,:,:2,0] # (?,f,17,2)

def np_scale_normalize(predicted, target):
    assert (predicted.shape == target.shape) # (?,f,p,3)
    # Added by Lawrence on 05/22/22. Recommended to get best scale factor
    predicted -= predicted[:,:,[0]]
    target -= target[:,:,[0]]

    norm_predicted = np.mean(np.sum(predicted**2, axis=3, keepdims=True), axis=2, keepdims=True)
    norm_target = np.mean(np.sum(target*predicted, axis=3, keepdims=True), axis=2, keepdims=True)
    scale = norm_target / norm_predicted
    return scale * predicted


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
    return qv_mult_v2(q2, vpose)


