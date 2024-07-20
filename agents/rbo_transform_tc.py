# -*- coding: utf-8 -*-
# @Time    : 4/28/2021 10:53 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : rbo_transform_tc.py
# @Software: pose.reg

import torch
import torch.nn as nn
import numpy as np

from agents.helper import processor


def to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()

def torch_t(num, dtype=torch.float32, gradient=False):
    return torch.tensor(num, dtype=dtype, device=torch.device(processor), requires_grad=gradient)

def guard_div1(tensor_1, tensor_2):
    '''
    Protecting division is necessary to guard against division by 0.
    Particularly, when used to convert a vector to its unit-vector by dividing by the magnitude
    - since there are instances that may generate admisable null-vectors <0,0,0>,
    - computing the magnitude of such null vector will lead to division by 0 and consequently nan
    Args:
        tensor_1: numerator tensor
        tensor_2: denominator tensor
    Returns: tensor_1/tensor_2 with infinite and nan values replaced with 0
    '''
    #assert(tensor_1.dtype==torch.float32), 'tensor_1 dtype: {}'.format(tensor_1.dtype)
    #assert(tensor_2.dtype==torch.float32), 'tensor_2 dtype: {}'.format(tensor_2.dtype)
    division = torch.div(tensor_1, tensor_2)
    return torch.nan_to_num(division, nan=0.0) # safe_div

def guard_div2(tensor_1, tensor_2, eps=1e-08):
    return torch.div(tensor_1, tensor_2.clamp(min=eps))

def guard_div3(tensor_1, tensor_2, eps=1e-05):
    return torch.div(tensor_1, (tensor_2 + eps))

def guard_sqrt(tsr, eps=1e-08):
    return torch.sqrt(tsr.clamp(min=eps)) # sqrt_clamped_tsr

def torch_vecdot(a_vec_tensors, b_vec_tensors, keepdim=True, dim=-1):
    # dot product of vectors in dim=2
    # equivalent to np.dot(a_vec_tensors, b_vec_tensors)
    return torch.sum(a_vec_tensors * b_vec_tensors, dim=dim, keepdim=keepdim)

def torch_matdot(mtx_tensor, vec_tensors, row_ax=-2, col_ax=-1):
    # dot product between matrices in dim=2&3 and vectors in dim=3
    # equivalent to np.dot(mtx_tensor, vec_tensors.T).T
    # mtx_tensor-->(?,f,4,4)
    # vec_tensor-->(?,f,n,4)
    return torch.transpose(torch.matmul(mtx_tensor, torch.transpose(vec_tensors, col_ax, row_ax)), col_ax, row_ax)

def are_valid_numbers(tensor):
    is_infinity = torch.isinf(tensor)
    some_infinite_values = torch.any(is_infinity)
    if some_infinite_values:
        is_infinity = torch.flatten(is_infinity)
        infinite_tsr = torch.flatten(tensor)[is_infinity]
        print('\n[Assert Warning] {:,} (or {:.2%}) tensor are infinity (inf)\n'.format(
            infinite_tsr.shape[0], infinite_tsr.shape[0]/is_infinity.shape[0]))

    is_undefined = torch.isnan(tensor)
    some_undefined_values = torch.any(is_undefined)
    if some_undefined_values:
        is_undefined = torch.flatten(is_undefined)
        undefined_tsr = torch.flatten(tensor)[is_undefined]
        print('\n[Assert Warning] {:,} (or {:.2%}) tensor are undefined (nan)\n'.format(
            undefined_tsr.shape[0], undefined_tsr.shape[0]/is_undefined.shape[0]))

    return not (some_infinite_values or some_undefined_values)

def are_unit_vectors(vector_t, atol=1e-08):
    # Checks if a vector is a valid unit vector
    magnitude = torch.linalg.norm(vector_t, dim=-1) # should be 1
    are_unit_vecs = torch.isclose(magnitude, torch_t(1.), atol=atol)
    all_are_unit_vecs = torch.all(are_unit_vecs)
    if not all_are_unit_vecs:
        are_not_ones = torch.flatten(torch.logical_not(are_unit_vecs))
        not_ones_tsr = torch.flatten(magnitude)[are_not_ones]
        print('\n[Assert Warning] {:,} (or {:.2%}) magnitudes are not exactly 1, some values differ by more than {}\n{}\n'.
              format(not_ones_tsr.shape[0], not_ones_tsr.shape[0]/are_not_ones.shape[0], atol, not_ones_tsr))
    return all_are_unit_vecs

def are_non_zero_length_vectors(vector_t, low_bound):
    magnitude = torch.linalg.norm(vector_t, dim=-1) # should be 1
    are_non_zero = magnitude>low_bound
    all_are_non_zero = torch.all(are_non_zero)
    if not all_are_non_zero:
        are_zeros = torch.flatten(torch.logical_not(are_non_zero))
        zeros_vec = vector_t.view(-1,3)[are_zeros,:]
        zeros_len = magnitude.view(-1,1)[are_zeros,:]
        zeros_tsr = torch.cat([zeros_len, zeros_vec], dim=-1)
        print('\n[Assert Warning] {:,} (or {:.2%}) vectors are length 0 or less\n{}\n'.
              format(zeros_len.shape[0], zeros_len.shape[0]/are_zeros.shape[0], zeros_tsr))
    return all_are_non_zero

def angles_are_within_bounds(thetas, low_bound, up_bound):
    are_within_bounds = torch.logical_and(low_bound<=thetas, thetas<=up_bound)
    all_are_within_bounds = torch.all(are_within_bounds)
    if not all_are_within_bounds:
        are_not_in_bounds = torch.flatten(torch.logical_not(are_within_bounds))
        out_of_bounds_angles = torch.flatten(thetas)[are_not_in_bounds]
        print('\n[Assert Warning] {:,} (or {:.2%}) angles are out of bounds - ie. some angles are <{} or >{}\n{}\n'.
              format(out_of_bounds_angles.shape[0], out_of_bounds_angles.shape[0]
                     / are_not_in_bounds.shape[0], low_bound, up_bound, out_of_bounds_angles))
    return all_are_within_bounds

def are_parallel_to_axis_uvecs(rotation_axis_uvec, parallel_axis_uvec, atol=1e-08):
    # if not null, then both unit vectors should be parallel to each other
    if parallel_axis_uvec is None: return True
    uvecs_diff = torch.abs(rotation_axis_uvec) - torch.abs(parallel_axis_uvec) # (?,f,j,3)
    uvecs_dist = torch.linalg.norm(uvecs_diff) # (?,f,j)
    are_similar_uvecs = torch.isclose(uvecs_dist, torch_t(0.), atol=atol) # (?,f,j)
    rot_vec_magnitude = torch.linalg.norm(rotation_axis_uvec, dim=-1) # (?,f,j)
    are_null_rot_vecs = torch.isclose(rot_vec_magnitude, torch_t(0.), atol=atol) # (?,f,j)
    are_parallel_uvec = torch.logical_or(are_similar_uvecs, are_null_rot_vecs) # (?,f,j)
    all_are_parallel_uvec = torch.all(are_parallel_uvec)
    if not all_are_parallel_uvec:
        are_not_parallel_uvec = torch.flatten(torch.logical_not(are_parallel_uvec))
        not_parallel_rot_uvec = rotation_axis_uvec.view(-1,3)[are_not_parallel_uvec,:]
        not_parallel_par_uvec = parallel_axis_uvec.view(-1,3)[are_not_parallel_uvec,:]
        not_parallel_uvec = torch.stack((not_parallel_rot_uvec, not_parallel_par_uvec), dim=-2) # (?*,2,3)
        print('\n[Assert Warning] {:,} (or {:.2%}) rotation-axis and anticipated axis uvecs are not parallel\n{}\n'.
              format(not_parallel_uvec.shape[0], not_parallel_uvec.shape[0]/are_not_parallel_uvec.shape[0], not_parallel_uvec))
    return all_are_parallel_uvec

def are_valid_angle_axis(rotation_axis_uvec, cos_theta, atol=1e-08):
    # Rotation axis must be a unit vector or a null vector <0,0,0>
    # rotation axis can be a null vector (with corresponding cos_theta=1 or theta=0)
    # which will result in no rotation
    # rotation_axis_uvec:(?,f,j,3) - cos_theta:(?,f,j,1)
    magnitude = torch.linalg.norm(rotation_axis_uvec, dim=-1) # (?,f,j) should be 1
    are_unit_vecs = torch.isclose(magnitude, torch_t(1.), atol=atol) # (?,f,j)
    are_null_vecs = torch.isclose(magnitude, torch_t(0.), atol=atol) # (?,f,j)
    cos_theta_are_ones = torch.isclose(cos_theta[...,0], torch_t(1.), atol=atol) # (?,f,j)
    are_ok_nulls = torch.logical_and(are_null_vecs, cos_theta_are_ones) # (?,f,j)
    are_ok_angle_axis = torch.logical_or(are_unit_vecs, are_ok_nulls) # (?,f,j)
    all_are_ok_angle_axis = torch.all(are_ok_angle_axis)
    if not all_are_ok_angle_axis:
        are_not_ok_angle_axis = torch.flatten(torch.logical_not(are_ok_angle_axis)) # (?*f*j,)
        not_ok_axis = rotation_axis_uvec.view(-1,3)[are_not_ok_angle_axis, :] # (?*f*j,3)
        not_ok_veclen = torch.flatten(magnitude)[are_not_ok_angle_axis] # (?*,)
        not_ok_cosine = torch.flatten(cos_theta)[are_not_ok_angle_axis] # (?*,)
        not_ok_angle = torch.rad2deg(torch.arccos(not_ok_cosine)) # (?*,)
        not_ok_meta = torch.stack((not_ok_veclen, not_ok_cosine, not_ok_angle), dim=-1) # (?*,3)
        not_ok_meta = torch.stack((not_ok_axis, not_ok_meta), dim=-2) # (?*,2,3)
        print('\n[Assert Warning] {:,} (or {:.2%}) angle-axis are invalid\n{}\n'.
              format(not_ok_meta.shape[0], not_ok_meta.shape[0]/are_not_ok_angle_axis.shape[0], not_ok_meta))
    return all_are_ok_angle_axis

def contains_null_vectors(vec_tsr, tsr_name, atol=1e-07):
    magnitude = torch.linalg.norm(vec_tsr, dim=-1) # (?,f,j) should be 1
    are_null_vecs = torch.isclose(magnitude, torch_t(0.), atol=atol) # (?,f,j)
    has_null_vecs = torch.any(are_null_vecs)
    if has_null_vecs:
        zero_vec_lens = torch.flatten(magnitude[are_null_vecs])
        all_vec_lens = torch.flatten(magnitude)
        print('\n[Check Warning] {:,} (or {:.2%}) of {} vectors are null\n{}\n'.
              format(zero_vec_lens.shape[0], zero_vec_lens.shape[0]/all_vec_lens.shape[0], tsr_name, zero_vec_lens))
    return has_null_vecs


def are_translation_matrices(T, is_transposed=False):
    # Checks if the last two axis in multi-dimensional array T looks like a translation matrix
    # 4x4 translation matrix: |1  0  0  tx|  -  4x4 transposed translation matrix: | 1  0  0  0|
    #                         |0  1  0  ty|                                        | 0  1  0  0|
    #                         |0  0  1  tz|                                        | 0  0  1  0|
    #                         |0  0  0   1|                                        |tx ty tz  1|
    #assert(T.ndim==5), 'T should be ?xfxjx4x4, but ndim=={}'.format(T.ndim)
    assert(T.shape[3:]==(4,4)), 'Translation should be 4x4 matrix, not {}'.format(T.shape[3:])
    diagonals = T[:,:,:,[0,1,2,3],[0,1,2,3]]
    diagonals_are_1s = torch.isclose(diagonals, torch_t(1.))
    if is_transposed:
        zero_terms = T[:,:,:,[0,0,0,1,1,1,2,2,2],[1,2,3,0,2,3,0,1,3]] # for transposed translation matrix
    else: zero_terms = T[:,:,:,[0,0,1,1,2,2,3,3,3],[1,2,0,2,0,1,0,1,2]] # for translation matrix
    zero_terms_are_0s = torch.isclose(zero_terms, torch_t(0.))
    are_translations = torch.all(diagonals_are_1s) and torch.all(zero_terms_are_0s)
    if not are_translations:
        print ('\n[Assert Warning]Not translation matrix:\n{}\n'.
               format(torch.cat((diagonals_are_1s, zero_terms_are_0s), dim=-1)))
    return are_translations

def are_rotation_matrices(R, atol=1e-05):
    # Checks if the matrices of R:(?,f,j,4,4) at last two axis are orthogonal rotation matrices.
    #assert(R.ndim==5), 'R should be ?xfxjx4x4, but ndim=={}'.format(R.ndim)
    assert(R.shape[3:]==(3,3)), 'Rotation should be 4x4 matrix, not {}'.format(R.shape[3:])
    Rt = torch.transpose(R, 4, 3)
    shouldBeIdentity = torch.matmul(Rt, R)
    identity_4x4 = torch.eye(3, dtype=torch.float32, device=torch.device(processor)) # (3,3)
    are_identity = torch.isclose(shouldBeIdentity, identity_4x4, atol=atol) # was atol=1e-04
    all_are_identity = torch.all(are_identity)
    if not all_are_identity:
        are_identity = torch.all(torch.all(are_identity, dim=-1), dim=-1)
        are_not_identity = torch.flatten(torch.logical_not(are_identity))
        not_identity_tsr = shouldBeIdentity.view((-1,3,3))[are_not_identity,:,:]
        print('\n[Assert Warning] {:,} (or {:.2%}) of R^T and R matmul are not identity. Some values differ by more than {}'
              '\n\tTherefore the rotation matrices are not orthogonal and improper\n{}\n'.
              format(not_identity_tsr.shape[0], not_identity_tsr.shape[0]
                     / are_not_identity.shape[0], atol, not_identity_tsr))
    return all_are_identity

def are_proper_rotation_matrices(R, nr_fb_idxs, atol=1e-05):
    # Checks if the matrices of R:(?,f,j*,3,3) at last two axis are proper rotation matrices.
    # When group_jmc is True, reflections are intended for Left body parts: (LHip,LThigh,LLeg,LShoulder,LBicep,LForearm)
    # so exclude left body parts when (group_jmc=True) and test on right and central body parts
    assert(R.shape[3:]==(3,3)), 'Rotation should be 3x3 matrix, not {}'.format(R.shape[3:])
    shouldBePositiveOne = torch.linalg.det(R[:,:,nr_fb_idxs]) # (?,f,j')
    are_positive_ones = torch.isclose(shouldBePositiveOne, torch_t(1.0), atol=atol) # (?,f,j')
    all_are_positive_ones = torch.all(are_positive_ones) # (1.)
    if not all_are_positive_ones:
        are_not_positive_ones = torch.flatten(torch.logical_not(are_positive_ones)) # (?*f*j',)
        not_positive_ones_tsr = shouldBePositiveOne.view(-1)[are_not_positive_ones] # (?')
        print('\n[Assert Warning] {:,} (or {:.2%}) of det(R) are not 1 (up to +/-{}).\n\tAssuming the rotation matrices '
              'have been confirmed orthogonal, then they must be improper, especially if det(R) = -1\n{}\n'.
              format(not_positive_ones_tsr.shape[0], not_positive_ones_tsr.shape[0]
                     / are_not_positive_ones.shape[0], atol, not_positive_ones_tsr))
    return all_are_positive_ones

def pivot_kpts_is_at_origin(pivot_aligned_kpts):
    pivot_at_origin = torch.all(torch.isclose(pivot_aligned_kpts, torch_t(0.)))
    assert (pivot_at_origin), 'Not all pivot keypoints are at (0,0,0)'
    return pivot_at_origin

def axis_kpts_lie_on_axis(axis_aligned_kpts, axis_uvecs, axis_2x2_matrix=None, atol1=1e-05, atol2=1e-05, test_type=0):
    '''
    Test if the axis-keypoint (after relative bone orientation alignment)
    is on the corresponding X or Y axis line
    Args:
        axis_aligned_kpts: (?,f,j,3) axis keypoint coordinates after alignments
        axis_uvecs: (?,f,j,3) corresponding 1st quadrant axis unit vector
        axis_2x2_matrix: (?,f,j,2,2) matrix for determinant test
        xy_axis_endpoints: (?,f,j,3) last index is (N,0) or (0,N) for X or Y axis
    Returns: True or False
    '''

    # test axis-kpt is on x-axis or y-axis by
    # (Option 1) checking the determinant of column stacked matrix: [ |x/y-axis(2d)| |(x,y) of axis-kpt| ] is 0
    # - this holds provided that the z-components of axis-kpts are 0
    if test_type==0: # determinant test
        # test z-components of axis-kpts are 0
        z_comp_are_0s = torch.isclose(axis_aligned_kpts[:,:,:,2], torch_t(0.), atol=atol1) # rmtx:1e-05, quat:7.5e-05
        all_z_comp_are_03 = torch.all(z_comp_are_0s)
        if not all_z_comp_are_03:
            z_comp_are_not_0s = torch.flatten(torch.logical_not(z_comp_are_0s))
            non_zero_tsr = torch.flatten(axis_aligned_kpts[:,:,:,2])[z_comp_are_not_0s]
            print('\n[Assert Warning] {:,} (or {:.2%}) Axis keypoints Z-component are non-zero'
                  '\n\tand therefore are not on the XY plane:\n{}\n'.
                  format(non_zero_tsr.shape[0], non_zero_tsr.shape[0]/z_comp_are_not_0s.shape[0], non_zero_tsr))

        # determinant test
        axis_2x2_matrix[:,:,:,:,1] = axis_aligned_kpts[:,:,:,:2]
        #axis_2x2_matrix = torch.stack((axis_uvecs[:,:,:,:2], axis_aligned_kpts[:,:,:,:2]), dim=4) # (?,f,j,2,2)
        determinants = torch.linalg.det(axis_2x2_matrix) # (?,f,j)
        are_on_x_or_y_axis = torch.isclose(determinants, torch_t(0.), atol=atol2) # rmtx:1e-07, quat:9.5e-06
        all_are_on_x_or_y_axis = torch.all(are_on_x_or_y_axis)
        if not all_are_on_x_or_y_axis:
            are_not_on_x_or_y_axis = torch.flatten(torch.logical_not(are_on_x_or_y_axis))
            not_on_x_or_y_axis_tsr = torch.flatten(determinants)[are_not_on_x_or_y_axis]
            print('\n[Assert Warning] {:,} (or {:.2%}) determinants of X and Y components are non-zero'
                  '\n\tTherefore the Axis keypoints are not on the respective X or Y axis:\n{}\n'.
                  format(not_on_x_or_y_axis_tsr.shape[0], not_on_x_or_y_axis_tsr.shape[0]
                         / are_not_on_x_or_y_axis.shape[0], not_on_x_or_y_axis_tsr))

        return all_z_comp_are_03 and all_are_on_x_or_y_axis

    # (Option 2) alternatively, the cross product of x/y-axis(3d) and axis-kpt should be null vector
    # - provided that neither are null vectors <0,0,0>
    if test_type==1: # cross-product test
        axis_kpts_x_uvecs = torch.cross(axis_aligned_kpts, axis_uvecs, dim=-1) # (?,f,j,3)
        magnitude = torch.linalg.norm(axis_kpts_x_uvecs, dim=-1) # (?,f,j) should be 1
        are_null_vecs = torch.isclose(magnitude, torch_t(0.), atol=atol1) # rmtx:1e-05, quat:7.5e-05
        all_are_null_vecs = torch.all(are_null_vecs)
        if not all_are_null_vecs:
            are_not_null_vecs = torch.flatten(torch.logical_not(are_null_vecs))
            not_null_vecs = axis_kpts_x_uvecs.view(-1,3)[are_not_null_vecs,:] # (?*f*j,3)->(?*,3)
            not_null_vecs_len = magnitude.view(-1,1)[are_not_null_vecs,:] # (?*f*j,1)->(?*,1)
            not_null_meta = torch.cat([not_null_vecs, not_null_vecs_len], dim=-1) # (?*,4)
            print('\n[Assert Warning] {:,} (or {:.2%}) cross products of axis vectors are not null zero'
                  '\n\tTherefore the Axis keypoints are not on the respective X or Y axis: [x,y,z,len]\n{}\n'.
                  format(not_null_meta.shape[0], not_null_meta.shape[0]/are_not_null_vecs.shape[0], not_null_meta))
        return all_are_null_vecs

def kpts_are_on_axis_hemisphere(plane_aligned_kpts, quadrant_2nd_axis_uvec):
    '''
    Test and confirm that the plane-kpt (after rotation alignment) is on the correct hemisphere
    - ie. the sign of the x or y component of the plane-kpt matches the sign of the corresponding 2nd quadrant unit-vector
    - eg. if 2nd quadrant axis is the y-axis, then sign(2nd-quadrant-axis[1]) = sign(plane-kpt[1])
    -     if 2nd quadrant axis is the x-axis, then sign(2nd-quadrant-axis[0]) = sign(plane-kpt[0])
    Args:
        plane_aligned_kpts: (?,f,j,3) plane keypoint coordinates after alignments
        quadrant_2nd_axis_uvec: (?,f,j,3) unit vector of 2nd quadrant axis
    Returns: True or False
    '''
    plane_aligned_kpts = plane_aligned_kpts.view(-1,3) # (?*f*j,3)
    quadrant_2nd_axis_uvec = quadrant_2nd_axis_uvec.view(-1,3) # (?*f*j,3)
    component_idxs = torch.argmax(torch.abs(quadrant_2nd_axis_uvec), dim=-1) # (?*f*j,)
    dim1_idxs = np.arange(component_idxs.shape[0])
    plane_kpt_comp_sign = torch.sign(plane_aligned_kpts[dim1_idxs, component_idxs]) # (?*f*j,)
    q2nd_axis_comp_sign = torch.sign(quadrant_2nd_axis_uvec[dim1_idxs, component_idxs]) # (?*f*j,)
    are_on_axis_hemi = plane_kpt_comp_sign==q2nd_axis_comp_sign # (?*f*j,)
    all_are_on_axis_hemi = torch.all(are_on_axis_hemi)
    if not all_are_on_axis_hemi:
        are_not_on_axis_hemi = torch.logical_not(are_on_axis_hemi) # (?*f*j,)
        not_on_axis_hemi_pk = plane_aligned_kpts[are_not_on_axis_hemi,:] # (?*,3)
        not_on_axis_hemi_q2 = quadrant_2nd_axis_uvec[are_not_on_axis_hemi,:] # (?*,3)
        not_on_axis_hemi_tsr = torch.stack((not_on_axis_hemi_pk, not_on_axis_hemi_q2), dim=-2) # (?*,2,3)
        print('\n[Assert Warning] {:,} (or {:.2%}) plane-kpts are not on correct 2nd-axis hemisphere:\n{}\n'.
              format(not_on_axis_hemi_tsr.shape[0], not_on_axis_hemi_tsr.shape[0]
                     / are_not_on_axis_hemi.shape[0], not_on_axis_hemi_tsr))
    return all_are_on_axis_hemi

def plane_kpts_line_on_xyplane(plane_aligned_kpts, planar_4x4_matrix):
    '''
    Test if the plane-keypoint (after relative bone orientation alignment)
    is on the XY plane
    Args:
        plane_aligned_kpts: (?,f,j,3) plane keypoint coordinates after alignments
        planar_4x4_matrix: (?,f,j,4,4) matrix for determinant test
    Returns: True or False
    '''
    planar_4x4_matrix[:,:,:,:3,3] = plane_aligned_kpts
    determinants = torch.linalg.det(planar_4x4_matrix)
    kpts_are_on_xyplane = torch.isclose(determinants, torch_t(0.))
    all_kpts_are_on_xyplane = torch.all(kpts_are_on_xyplane)
    if not all_kpts_are_on_xyplane:
        kpts_are_not_on_xyplane = torch.flatten(torch.logical_not(kpts_are_on_xyplane))
        kpts_not_on_xyplane_tsr = torch.flatten(determinants)[kpts_are_not_on_xyplane]
        print('\n[Assert Warning] {:,} (or {:.2%}) determinants of origin:(0,0,0), x-axis:(1,0,0), y-axis:(0,1,0)'
              '\n        and kpt are non-zero. Therefore the Axis keypoints are not on the respective X or Y axis:\n{}\n'.
              format(kpts_not_on_xyplane_tsr.shape[0], kpts_not_on_xyplane_tsr.shape[0]
                     / kpts_are_not_on_xyplane.shape[0], kpts_not_on_xyplane_tsr))
    return all_kpts_are_on_xyplane

def plane_pair_z_depth_are_same(plane_kpts1_z, plane_kpts2_z, atol=5e-05):
    z_comp_are_equal = torch.isclose(plane_kpts1_z, plane_kpts2_z, atol=atol) # (?,f,t)
    all_z_comp_are_equal = torch.all(z_comp_are_equal)
    if not all_z_comp_are_equal:
        z_comp_are_not_equal = torch.flatten(torch.logical_not(z_comp_are_equal)) # (?*f*t)
        not_equal_z_comp_tsr = torch.stack((torch.flatten(plane_kpts1_z)[z_comp_are_not_equal], # (?*f*t,2)
                                            torch.flatten(plane_kpts2_z)[z_comp_are_not_equal]), dim=-1)
        print('\n[Assert Warning] {:,} (or {:.2%}) plane keypoint pairs do not have the same z-depth component'
              '\n        after alignment and are therefore not equally displaced from the XY-plane:\n{}\n'.
              format(not_equal_z_comp_tsr.shape[0], not_equal_z_comp_tsr.shape[0]
                     / z_comp_are_not_equal.shape[0], not_equal_z_comp_tsr))
    return all_z_comp_are_equal

def likelihoods_are_non_negative(likelihood_tsr, log_spread=None):
    if log_spread is not None:
        assert(log_spread>=0), 'log_spread:{} must be >=0'.format(log_spread)
        log_inputs_are_positive = likelihood_tsr>=-log_spread
    else: log_inputs_are_positive = likelihood_tsr>=0
    all_log_inputs_are_positive = torch.all(log_inputs_are_positive)
    if not all_log_inputs_are_positive:
        log_inputs_are_negative = torch.flatten(torch.logical_not(log_inputs_are_positive))
        negative_log_inputs_tsr = torch.flatten(likelihood_tsr)[log_inputs_are_negative]
        print('\n[Assert Warning] {:,} (or {:.2%}) likelihoods/log-inputs are negative'
              '\n        and would produce undefined values due to log(x<0):\n{}\n'.
              format(negative_log_inputs_tsr.shape[0], negative_log_inputs_tsr.shape[0]
                     / log_inputs_are_negative.shape[0], negative_log_inputs_tsr))
    return all_log_inputs_are_positive


class FreeBoneOrientation(nn.Module):

    def __init__(self, batch_size, xy_yx_axis_dirs=None, z_axis_ab_idxs=None, yx_axis_ab_idxs=None, xy_yx_idxs=None,
                 quad_uvec_axes1=None, quad_uvec_axes2=None, plane_proj_mult=None, hflip_multiplier=None, nr_fb_idxs=None,
                 n_frm=1, n_fbs=16, pivot_kpt_idx=0, xy_axis_kpt_idx=1, yx_plane_kpt_idx=2, free_kpt_idx=-1, z_idx=2,
                 rot_tfm_mode=0, ret_mode=0, quintuple=False, invert_pose=False, validate_ops=False, **kwargs):
        super(FreeBoneOrientation, self).__init__(**kwargs)

        # for rotation matrix
        if rot_tfm_mode==0:
            self.rotation_alignment_func = self.matrix_rotation
            self.xy_axis_dirs = xy_yx_axis_dirs[0] # xy
            self.yx_axis_dirs = xy_yx_axis_dirs[1] # yx
            self.z_axis_a_idxs = z_axis_ab_idxs[0] # a
            self.z_axis_b_idxs = z_axis_ab_idxs[1] # b
            self.yx_axis_a_idxs = yx_axis_ab_idxs[0] # a
            self.yx_axis_b_idxs = yx_axis_ab_idxs[1] # b
            self.xy_idxs = xy_yx_idxs[0] # xy
            self.yx_idxs = xy_yx_idxs[1] # yx
            self.z_idx = z_idx

        # for quaternion rotation
        else: # rot_tfm_mode==1
            self.rotation_alignment_func = self.quaternion_rotation
            self.quad_uvec_axes1 = quad_uvec_axes1 # (1,1,j,3)
            self.quad_uvec_axes2 = quad_uvec_axes2 # (1,1,j,3)
            self.plane_proj_mult = plane_proj_mult # (1,1,j,3)
            self.hflip_multiplier = hflip_multiplier # (1,1,j,3)
            self.two_tsr = torch_t(2.)
            self.one_tsr = torch_t(1.)
            self.half_tsr = torch_t(0.5)
            self.zero_tsr = torch_t(0.0)
            self.epsilon_tsr = torch_t(1e-08)

        self.n_fbs = n_fbs
        self.pivot_kpt_idx = pivot_kpt_idx
        self.axis_kpt_idx = xy_axis_kpt_idx
        self.plane_kpt_idx = yx_plane_kpt_idx
        self.free_kpt_idx = free_kpt_idx
        self.ret_mode = ret_mode
        self.rot_tfm_mode = rot_tfm_mode
        self.validate_ops = validate_ops
        self.invert_pose = invert_pose
        self.nr_fb_idxs = nr_fb_idxs # (j*,)
        self.qb = 4 #3

        self.oat_atol1 = 1e-05 if rot_tfm_mode==0 else 7.5e-05 # tolerance for on-axis-test
        self.oat_atol2 = 1e-06 if rot_tfm_mode==0 else 9.5e-06 # tolerance for on-axis-test

        self.invert_pose_mult = torch.reshape(torch_t([-1,-1,1]), (1,1,1,1,3))
        self.jnt_idxs = np.arange(self.n_fbs) # (j,) do NOT remove, very necessary!
        self.quintuple_kpts = quintuple
        self.n_koi = 5 if self.quintuple_kpts and self.validate_ops else 4 # koi: keypoints of interest
        self.pap_kpts_idxs = [self.pivot_kpt_idx,self.axis_kpt_idx, self.plane_kpt_idx]
        if self.quintuple_kpts: self.pap_kpts_idxs.append(self.plane_kpt_idx+1)
        self.build_for_batch(batch_size, n_frm)

    def build_for_batch(self, batch_size, n_frames):
        # input_shape:(?,f,j,4,3)
        self.n_bsz = batch_size
        self.n_frm = n_frames

        if self.rot_tfm_mode==0: # for rotation matrix
            if self.validate_ops:
                # self.ones_tsr_bxfxjxkx1 = torch.ones((self.n_bsz,self.n_frm,self.n_fbs,self.n_koi-1,1),
                #                                      dtype=torch.float32, device=torch.device(processor)) # (?,f,j,3or4,1)
                self.axis_2x2_matrix = torch.zeros((self.n_bsz,self.n_frm,self.n_fbs,2,2),
                                                   dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4,4)
                self.axis_2x2_matrix[:,:,self.jnt_idxs,0,self.xy_idxs] = self.xy_axis_dirs[0,0,:,0] #torch_t(1.)
                self.axis_test_uvecs = torch.zeros((self.n_bsz,self.n_frm,self.n_fbs,3),
                                                   dtype=torch.float32, device=torch.device(processor)) # (?,f,j,3)
                self.axis_test_uvecs[:,:,self.jnt_idxs,self.xy_idxs] = self.xy_axis_dirs[0,0,:,0]

        else: # rot_tfm_mode==1 for quaternion
            self.quadrant_1st_axis_uvec = torch.tile(self.quad_uvec_axes1, dims=(self.n_bsz,self.n_frm,1,1))  # (?,f,j,3)
            self.quadrant_2nd_axis_uvec = torch.tile(self.quad_uvec_axes2, dims=(self.n_bsz,self.n_frm,1,1))  # (?,f,j,3)
            if self.validate_ops:
                self.axis_2x2_matrix = torch.zeros((self.n_bsz,self.n_frm,self.n_fbs,2,2),
                                                   dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4,4)
                self.axis_2x2_matrix[:,:,:,0,:] = self.quadrant_1st_axis_uvec[:,:,:,:2]
                self.axis_test_uvecs = self.quadrant_1st_axis_uvec

        # Needed for tests and assertions
        if self.validate_ops:
            self.planar_4x4_matrix = torch.zeros((self.n_bsz,self.n_frm,self.n_fbs,4,4),
                                                dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4,4)
            self.planar_4x4_matrix[:,:,[0,1,3,3,3,3],[1,2,0,1,2,3]] = torch_t(1.)
            self.pi_tsr = torch_t(np.pi)

    def forward(self, input_tensor, **kwargs):
        if input_tensor.shape[:2]!=(self.n_bsz, self.n_frm):
            self.build_for_batch(input_tensor.shape[0], input_tensor.shape[1])
        if self.invert_pose: input_tensor = input_tensor * self.invert_pose_mult

        # Step 0: Translate quadruplet/quintuple keypoints such that rotation pivot is at origin
        pivot_kpts = input_tensor[:,:,:,[self.pivot_kpt_idx],:3] # (?,f,j,1,3)
        if self.quintuple_kpts:
            rgt_pln2pvt_vec = input_tensor[:,:,:self.qb,0] - input_tensor[:,:,:self.qb,2] # (?,f,t,3) 2->0 : R->O
            plane_pair_vec = input_tensor[:,:,:self.qb,3] - input_tensor[:,:,:self.qb,2] # (?,f,t,3) 2->3 : R->L
            plane_pair_uvec = guard_div1(plane_pair_vec, torch.linalg.norm(plane_pair_vec, dim=-1, keepdim=True)) # (?,f,t,1)
            rgt_pln2pvt_proj = torch_vecdot(rgt_pln2pvt_vec, plane_pair_uvec) * plane_pair_uvec # (?,f,t,3)
            plane_displace_vec = rgt_pln2pvt_vec - rgt_pln2pvt_proj # (?,f,t,3)
        else: plane_displace_vec = None

        if self.quintuple_kpts and not self.validate_ops:
            quadruplet_kpts = input_tensor[:,:,:,[0,1,2,-1],:3] - pivot_kpts # (?,f,j,4,3)
        else: quadruplet_kpts = input_tensor - pivot_kpts # (?,f,j,4or5,3)

        tfm_meta_1, tfm_meta_2, aligned_pap_kpts, aligned_fb_vecs = \
            self.rotation_alignment_func(quadruplet_kpts, plane_displace_vec)

        if self.validate_ops:
            assert(are_valid_numbers(aligned_pap_kpts))
            assert(pivot_kpts_is_at_origin(aligned_pap_kpts[:,:,:,0]))
            assert(axis_kpts_lie_on_axis(aligned_pap_kpts[:,:,:,1], self.axis_test_uvecs,
                                         self.axis_2x2_matrix, atol1=self.oat_atol1, atol2=self.oat_atol2))
            if self.quintuple_kpts:
                assert(plane_kpts_line_on_xyplane(aligned_pap_kpts[:,:,self.qb:,2], self.planar_4x4_matrix[:,:,self.qb:]))
                assert(plane_pair_z_depth_are_same(aligned_pap_kpts[:,:,:self.qb,2,2], aligned_pap_kpts[:,:,:self.qb,3,2]))
            else: assert(plane_kpts_line_on_xyplane(aligned_pap_kpts[:,:,:,2], self.planar_4x4_matrix))

        if self.ret_mode==1: return aligned_fb_vecs # (?,f,j,3)

        # Compute free limb vectors
        free_bone_uvecs = guard_div1(aligned_fb_vecs, torch.linalg.norm(aligned_fb_vecs, dim=3, keepdim=True)) # (?,f,j,3)
        if self.ret_mode==0: return free_bone_uvecs # (?,f,j,3)

        # when self.ret_mode==-1 or -2
        return to_numpy(pivot_kpts), to_numpy(tfm_meta_1), to_numpy(tfm_meta_2), to_numpy(free_bone_uvecs)

    def matrix_rotation(self, quadruplet_kpts, plane_displace_vecs):
        # Get limb vectors from pairs of keypoint positions
        xy_axis_vecs = quadruplet_kpts[:,:,:,self.axis_kpt_idx] # (?,f,j,4,3)->(?,f,j,3)
        if self.quintuple_kpts:
            yx_plane_vecs = torch.cat((quadruplet_kpts[:,:,:self.qb,self.plane_kpt_idx] + plane_displace_vecs,
                                       quadruplet_kpts[:,:,self.qb:,self.plane_kpt_idx]), dim=2)
        else: yx_plane_vecs = quadruplet_kpts[:,:,:,self.plane_kpt_idx] # (?,f,j,4,3)->(?,f,j,3)
        xyz_comparable_vecs = self.broadcast_indexed_list_assign(xy_axis_vecs, yx_plane_vecs) # (?,f,j,2,3)

        # Define new x-axis, y-axis, and z-axis
        xy_axis_uvecs = guard_div1(xy_axis_vecs, torch.linalg.norm(xy_axis_vecs, dim=3, keepdim=True)) * self.xy_axis_dirs # (?,f,j,3)

        z_axis_vecs = torch.cross(xyz_comparable_vecs[:,:,self.jnt_idxs,self.z_axis_a_idxs], # (?,f,j,3)
                                  xyz_comparable_vecs[:,:,self.jnt_idxs,self.z_axis_b_idxs], dim=-1) # (?,f,j,3) -> (?,f,j,3)
        xyz_comparable_vecs = self.broadcast_indexed_list_assign(xy_axis_vecs, yx_plane_vecs, z_axis_vecs) # (?,f,j,3,3)
        z_axis_uvecs = guard_div1(z_axis_vecs, torch.linalg.norm(z_axis_vecs, dim=3, keepdim=True)) # (?,f,j,3)

        yx_axis_vecs = torch.cross(xyz_comparable_vecs[:,:,self.jnt_idxs,self.yx_axis_a_idxs], # (?,f,j,3)
                                   xyz_comparable_vecs[:,:,self.jnt_idxs,self.yx_axis_b_idxs], dim=-1) # (?,f,j,3) -> (?,f,j,3)
        yx_axis_uvecs = guard_div1(yx_axis_vecs, torch.linalg.norm(yx_axis_vecs, dim=3, keepdim=True)) * self.yx_axis_dirs # (?,f,j,3)

        # Derive frame rotation matrix from unit vec axis
        rotation_mtxs_3x3 = self.broadcast_indexed_list_assign(xy_axis_uvecs, yx_axis_uvecs, z_axis_uvecs) # (?,f,j,3,3)

        if self.validate_ops:
            assert(are_valid_numbers(xy_axis_uvecs) and are_unit_vectors(xy_axis_uvecs))
            assert(are_valid_numbers(z_axis_uvecs) and are_unit_vectors(z_axis_uvecs))
            assert(are_valid_numbers(yx_axis_uvecs) and are_unit_vectors(yx_axis_uvecs))
            # assert(are_rotation_matrices(rotation_mtxs_4x4)), 'Contains a non-orthogonal rotation matrix'
            assert(are_rotation_matrices(rotation_mtxs_3x3)), 'Contains a non-orthogonal rotation matrix'
            assert(are_proper_rotation_matrices(rotation_mtxs_3x3, self.nr_fb_idxs)), 'Contains an improper (reflection) matrix'
            # apply rotation transformation to other quadruplet/quintuple keypoints (pivot, axis, and plane kpts)
            # pap: pivot-axis-plane keypoints
            aligned_pap_kpts = torch_matdot(rotation_mtxs_3x3, quadruplet_kpts[:,:,:,self.pap_kpts_idxs]) # (?,f,j,3,3)
        else: aligned_pap_kpts = None

        # Using frame rotation matrix, transform position of non-origin translated
        # keypoints (in camera coordinated) to be represented in the joint frame
        aligned_fb_vec = torch_matdot(rotation_mtxs_3x3, quadruplet_kpts[:,:,:,[self.free_kpt_idx]]) # (?,f,j,1,3)

        return rotation_mtxs_3x3, rotation_mtxs_3x3, aligned_pap_kpts, aligned_fb_vec[:,:,:,0,:]

    def broadcast_indexed_list_assign(self, xy_vecs_bxfxjx3, yx_vecs_bxfxjx3, z_vecs_bxfxjx3=None):
        '''
        xy_vecs_bxfxjx3:(?,f,j,3), yx_vecs_bxfxjx3:(?,f,j,3), z_vecs_bxfxjx3:(?,f,j,3) or None
        Equivalent to:
        jnt_idxs = np.arange(self.n_fbs)
        xyz_vector_bxfxjx3x3 = tf.zeros((self.n_bsz,self.n_frm,self.n_fbs,3,3), dtype=tf.dtypes.float32) # (?,j,3,3)
        xyz_vector_bxfxjx3x3[:,:,jnt_idxs,self.xy_idxs] = xy_vecs_bxfxjx3
        xyz_vector_bxfxjx3x3[:,:,jnt_idxs,self.yx_idxs] = yx_vecs_bxfxjx3
        xyz_vector_bxfxjx3x3[:,:,jnt_idxs,self.z_idx] = z_vecs_bxfxjx3
        Returns: tensor of size (?,f,j,k,3), where k is 2 or 3
        '''
        if self.validate_ops:
            assert(xy_vecs_bxfxjx3.shape==yx_vecs_bxfxjx3.shape) , 'xy_vecs_bxfxjx3:{}'.format(xy_vecs_bxfxjx3.shape)
            assert(z_vecs_bxfxjx3 is None or yx_vecs_bxfxjx3.shape==z_vecs_bxfxjx3.shape), 'yx_vecs_bxfxjx3:{}'.format(yx_vecs_bxfxjx3.shape)

        xyz_vector_bxfxkx3_list = [] # j*(?,f,k,3)
        k_axs = 2 if z_vecs_bxfxjx3 is None else 3 # k==2->{xy,yx}, k==3->{xy,yx,z}

        for j_idx in range(self.n_fbs):
            xy_idx = self.xy_idxs[j_idx]
            yx_idx = self.yx_idxs[j_idx]
            if self.validate_ops:
                jnt_xyz_idxs = np.array([xy_idx,yx_idx,self.z_idx]) # xy, yx, & z idx are unique and each, one of 0 1 & 2
                assert(np.intersect1d(np.arange(0, 3), jnt_xyz_idxs).shape[0]==3), 'jnt_xyz_idxs:{}'.format(jnt_xyz_idxs)

            jnt_xyz_vector_bxfx3_list = [] # k*(?,f,3)
            for current_idx in range(k_axs):
                if xy_idx==current_idx: jnt_xyz_vector_bxfx3_list.append(xy_vecs_bxfxjx3[:,:,j_idx]) # (?,f,3)
                elif yx_idx==current_idx: jnt_xyz_vector_bxfx3_list.append(yx_vecs_bxfxjx3[:,:,j_idx]) # (?,f,3)
                else: # self.z_idx==current_idx
                    #assert(self.z_idx==current_idx), '{} vs. {}'.format(self.z_idx, current_idx)
                    #assert(z_vecs_bxfxjx3 is not None), "z_vecs_bxfxjx3 can't be None if this statement is reached"
                    jnt_xyz_vector_bxfx3_list.append(z_vecs_bxfxjx3[:,:,j_idx]) # (?,f,3)
            xyz_vector_bxfxkx3_list.append(torch.stack(jnt_xyz_vector_bxfx3_list, dim=2)) # k*(?,f,3)-->(?,f,k,3)
        xyz_vector_bxfxjxkx3 = torch.stack(xyz_vector_bxfxkx3_list, dim=2) # j*(?,f,k,3)-->(?,f,j,k,3)
        return xyz_vector_bxfxjxkx3


    def quaternion_rotation(self, quadruplet_kpts, plane_displace_vecs, pln_idx=0, fb_idx=1):
        # Step 1: Rotate to align Axis-kpt (or Axis-Bone) with the pre-configured X or Y axis
        xy_axis_vecs = quadruplet_kpts[:,:,:,self.axis_kpt_idx] # (?,f,j,4,3)->(?,f,j,3)
        quat_vec1 = self.rotate_vecb2veca_quaternion(xy_axis_vecs, self.quadrant_1st_axis_uvec) # (?,f,j,4)
        pln_fb_kpts_1of2_align = self.quaternion_rotate_vecs(
            quadruplet_kpts[:,:,:,[self.plane_kpt_idx, self.free_kpt_idx]], quat_vec1, k=2) # (?,f,j,2,3)

        # Step 2: Rotate to align Plane-kpt (or Plane-Bone) with the pre-configured XY-plane quadrant
        # todo: for better precision, instead of multiplying, crgeate zero tensor and insert none zero components into tensor
        if self.quintuple_kpts:
            rot_pln_disp_vecs = self.quaternion_rotate_vecs(plane_displace_vecs, quat_vec1[:,:,:self.qb]) # (?,f,t,3)
            yx_plane_vecs = torch.cat((pln_fb_kpts_1of2_align[:,:,:self.qb,pln_idx] + rot_pln_disp_vecs,
                                       pln_fb_kpts_1of2_align[:,:,self.qb:,pln_idx]), dim=2) # (?,f,j,3)
        else: yx_plane_vecs = pln_fb_kpts_1of2_align[:,:,:,pln_idx] # (?,f,j,2,3)->(?,f,j,3)
        proj_yx_plane_vecs = yx_plane_vecs * self.plane_proj_mult # (?,f,j,3)
        quat_vec2 = self.rotate_vecb2veca_quaternion(proj_yx_plane_vecs, self.quadrant_2nd_axis_uvec,
                                                     parallel_axis_uvec=self.quadrant_1st_axis_uvec) # (?,f,j,4)
        fb_vec_2of2_align = self.quaternion_rotate_vecs(pln_fb_kpts_1of2_align[:,:,:,fb_idx], quat_vec2) # (?,f,j,3)

        if self.validate_ops:
            pap_kpts_1of2_align = self.quaternion_rotate_vecs(
                quadruplet_kpts[:,:,:,self.pap_kpts_idxs], quat_vec1, k=self.n_koi-1) # (?,f,j,3or4,3)
            pap_kpts_2of2_align = self.quaternion_rotate_vecs(pap_kpts_1of2_align, quat_vec2, k=self.n_koi-1) # (?,f,j,3or4,3)
            assert(kpts_are_on_axis_hemisphere(pap_kpts_2of2_align[:,:,:,2], self.quadrant_2nd_axis_uvec))
        else: pap_kpts_2of2_align = None

        if self.hflip_multiplier is not None:
            fb_vec_2of2_align = fb_vec_2of2_align * self.hflip_multiplier # horizontal-flip for grouping symmetric fb-joints

        return quat_vec1, quat_vec2, pap_kpts_2of2_align, fb_vec_2of2_align

    def rotate_vecb2veca_quaternion(self, vec_b, vec_a, parallel_axis_uvec=None, eps=1e-07):
        '''
        Derives the angle-axis quaternion needed to rotate vec_b to align with vec_a
        Args:
            vec_b: (?,f,j,3) - vector to rotate
            vec_a: (?,f,j,3) - vector to align (typically a unit vector)
            cos_theta: (?,f,j,1) - cosine of smaller angle (in radians) between vec_b and vec_a
            parallel_to_axis: (?,f,j,3) or None, a unit vector that should be parallel to the rotation axis
        Returns: (?,f,j,4) quaternion vector
        '''

        vecb_x_veca = torch.cross(vec_b, vec_a, dim=-1) # (?,f,j,3)
        # axis of rotation is the unit vector of vec_b cross vec_a  [ self.torch_veclen ]
        rot_axis_uvec = guard_div2(vecb_x_veca, torch.linalg.norm(vecb_x_veca, dim=-1, keepdim=True))
        cos_theta = guard_div2(torch_vecdot(vec_b, vec_a), torch.linalg.norm(vec_b, dim=-1, keepdim=True)) # (?,f,j,1)

        # # The derivative of torch.arccos is numerical unstable at -1 or 1,
        # # producing infinite values. Hence we clip values to open interval (-1,1)
        # # https://discuss.pytorch.org/t/numerically-stable-acos-dot-product-derivative/12851/2
        # thetas = torch.arccos(cos_theta.clamp(-1.0 + eps, 1.0 - eps)) # (?,f,j,1)
        # half_theta = thetas * self.half_tsr # equivalent to theta/2 (?,f,j,1)
        # cos_half_theta = torch.cos(half_theta) # q_w:(?,f,j,1)
        # sin_half_theta = torch.sin(half_theta) # (?,f,j,3)

        # Alternative method to compute cos(theta/2) and sin(theta/2)
        # cos(x/2) = sqrt((1+cos(x))/2) and sin(x/z) = sqrt((1-cos(x))/2) for 0<=x<=pi
        cos_half_theta = guard_sqrt(guard_div2(self.one_tsr + cos_theta, self.two_tsr)) # q_w:(?,f,j,1)
        # todo: handle borderline case when cos_theta=-1 (ie. theta=0) >> no rotation
        #assert (torch.all(torch.isclose(cos_half_theta2, cos_half_theta1, atol=1e-07)))
        sin_half_theta = guard_sqrt(guard_div2(self.one_tsr - cos_theta, self.two_tsr))
        # todo: handle borderline case when sin_theta=1 (ie. theta=pi or 180) >> rotate twice about axis by 90"
        #assert (torch.all(torch.isclose(sin_half_theta2, sin_half_theta1, atol=1e-07)))

        if self.validate_ops:
            assert (are_non_zero_length_vectors(vec_a, low_bound=1e-05))
            assert (torch.all(torch_t(-1.)<=cos_theta) and torch.all(cos_theta<=torch_t(1.))) # implies 0<=theta<=180
            #assert(angles_are_within_bounds(thetas, low_bound=self.zero_tsr, up_bound=self.pi_tsr))
            # note, rotation axis can be a null <0,0,0> vector which results in no rotations
            assert(are_valid_numbers(rot_axis_uvec))
            assert(are_valid_angle_axis(rot_axis_uvec, cos_theta))
            assert(are_parallel_to_axis_uvecs(rot_axis_uvec, parallel_axis_uvec))

        q_xyz = rot_axis_uvec * sin_half_theta # (?,f,j,3)
        return torch.cat([cos_half_theta, q_xyz], dim=-1) # (?,f,j,4)

    def quaternion_rotate_vecs(self, vectors_3d, quaternion_vecs, k=None):
        '''
        Rotate vectors_3d according to the quaternion_vecs
        Args:
            vectors_3d: (?,f,j,k<=4,3) 3D pose vectors of quad-kpts
            quaternion_vecs: (?,f,j,4) corresponding quaternion vectors
            k: if not None, repeat k dimension (1<=k<=4)
        Returns: (?,f,j,k,3) or (?,f,j,3) if one_to_many=False >>
                 transformed 3D pose vectors of quad-kpts after rotation
        '''
        if k is not None:
            quaternion_vecs = torch.tile(torch.unsqueeze(quaternion_vecs, dim=-2), (1,1,1,k,1)) # (?,f,j,k,4)
        uv = torch.cross(quaternion_vecs[..., 1:], vectors_3d, dim=-1) # (?,f,j,k,3) or (?,f,j,3)
        uuv = torch.cross(quaternion_vecs[..., 1:], uv, dim=-1) # (?,f,j,k,3) or (?,f,j,3)
        return vectors_3d + self.two_tsr * (quaternion_vecs[..., :1] * uv + uuv) # (?,f,j,k,3) or (?,f,j,3)


# class FreeBoneConstruct(nn.Module):
#
#     def __init__(self, qset_kpt_idxs, batch_size, xy_yx_axis_dirs=None, z_axis_ab_idxs=None, yx_axis_ab_idxs=None, xy_yx_idxs=None,
#                  quad_uvec_axes1=None, quad_uvec_axes2=None, plane_proj_mult=None, hflip_multiplier=None, nr_fb_idxs=None,
#                  n_frm=1, n_fbs=16, pivot_kpt_idx=0, xy_axis_kpt_idx=1, yx_plane_kpt_idx=2, free_kpt_idx=-1, z_idx=2,
#                  rot_tfm_mode=0, ret_mode=0, quintuple=False, invert_pose=False, validate_ops=False, **kwargs):
#         super(FreeBoneConstruct, self).__init__(**kwargs)
#
#         self.qset_kpt_idxs = qset_kpt_idxs
#         self.xy_axis_dirs = xy_yx_axis_dirs[0] # xy
#         self.yx_axis_dirs = xy_yx_axis_dirs[1] # yx
#         self.z_axis_a_idxs = z_axis_ab_idxs[0] # a
#         self.z_axis_b_idxs = z_axis_ab_idxs[1] # b
#         self.yx_axis_a_idxs = yx_axis_ab_idxs[0] # a
#         self.yx_axis_b_idxs = yx_axis_ab_idxs[1] # b
#         self.xy_idxs = xy_yx_idxs[0] # xy
#         self.yx_idxs = xy_yx_idxs[1] # yx
#         self.z_idx = z_idx
#
#         self.n_fbs = n_fbs
#         # self.pivot_kpt_idx = pivot_kpt_idx
#         # self.axis_kpt_idx = xy_axis_kpt_idx
#         # self.plane_kpt_idx = yx_plane_kpt_idx
#         # self.free_kpt_idx = free_kpt_idx
#         # self.ret_mode = ret_mode
#         self.validate_ops = validate_ops
#         self.invert_pose = invert_pose
#         self.nr_fb_idxs = nr_fb_idxs # (j*,)
#         self.qb = 4 #3
#
#         self.oat_atol1 = 1e-05 # tolerance for on-axis-test
#         self.oat_atol2 = 1e-06 # tolerance for on-axis-test
#
#         self.invert_pose_mult = torch.reshape(torch_t([-1,-1,1]), (1,1,1,1,3))
#         # self.jnt_idxs = np.arange(self.n_fbs) # (j,) do NOT remove, very necessary!
#         self.n_koi = 4
#         self.pap_kpts_idxs = [self.pivot_kpt_idx,self.axis_kpt_idx, self.plane_kpt_idx]
#         self.build_for_batch(batch_size, n_frm)
#
#     def build_for_batch(self, batch_size, n_frames):
#         # input_shape:(?,f,j,4,3)
#         self.n_bsz = batch_size
#         self.n_frm = n_frames
#
#         if self.validate_ops:
#             self.axis_2x2_matrix = torch.zeros((self.n_bsz,self.n_frm,self.n_fbs,2,2),
#                                                dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4,4)
#             self.axis_2x2_matrix[:,:,self.jnt_idxs,0,self.xy_idxs] = self.xy_axis_dirs[0,0,:,0] #torch_t(1.)
#             self.axis_test_uvecs = torch.zeros((self.n_bsz,self.n_frm,self.n_fbs,3),
#                                                dtype=torch.float32, device=torch.device(processor)) # (?,f,j,3)
#             self.axis_test_uvecs[:,:,self.jnt_idxs,self.xy_idxs] = self.xy_axis_dirs[0,0,:,0]
#             # Needed for tests and assertions
#             self.planar_4x4_matrix = torch.zeros((self.n_bsz,self.n_frm,self.n_fbs,4,4),
#                                                  dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4,4)
#             self.planar_4x4_matrix[:,:,[0,1,3,3,3,3],[1,2,0,1,2,3]] = torch_t(1.)
#             self.pi_tsr = torch_t(np.pi)
#
#     def forward(self, input_tensor, fbj_idx, fb_vecs, **kwargs):
#         if input_tensor.shape[:2]!=(self.n_bsz, self.n_frm):
#             self.build_for_batch(input_tensor.shape[0], input_tensor.shape[1])
#         if self.invert_pose: input_tensor = input_tensor * self.invert_pose_mult
#
#         pivot_kpt_idx, axis_kpt_idx, plane_kpt_idx = self.qset_kpt_idxs[fbj_idx][:3]
#         free_kpt_idx = self.qset_kpt_idxs[fbj_idx][-1]
#
#         # Step 0: Translate quadruplet/quintuple keypoints such that rotation pivot is at origin
#         pivot_kpts = input_tensor[:,:,[pivot_kpt_idx],:3] # (?,f,1,3)
#
#         # quadruplet_kpts = input_tensor - pivot_kpts # (?,f,j,4or5,3)
#
#         fb_aligned_pose_kpts = self.matrix_rotation(input_tensor - pivot_kpts, fbj_idx, axis_kpt_idx, plane_kpt_idx)
#
#         if self.validate_ops:
#             assert(are_valid_numbers(fb_aligned_pose_kpts))
#             assert(pivot_kpts_is_at_origin(fb_aligned_pose_kpts[:,:,pivot_kpts]))
#             assert(axis_kpts_lie_on_axis(fb_aligned_pose_kpts[:,:,axis_kpt_idx], self.axis_test_uvecs,
#                                          self.axis_2x2_matrix, atol1=self.oat_atol1, atol2=self.oat_atol2))
#             assert(plane_kpts_line_on_xyplane(fb_aligned_pose_kpts[:,:,plane_kpt_idx], self.planar_4x4_matrix))
#
#         # Insert free-bone vector after alignment
#         fb_aligned_pose_kpts[:,:,free_kpt_idx] = fb_vecs
#         return fb_aligned_pose_kpts
#
#     def matrix_rotation(self, pose_kpts, fbj_idx, axis_kpt_idx, plane_kpt_idx):
#
#         # Get limb vectors from pairs of keypoint positions
#         xy_axis_vecs = pose_kpts[:,:,:,axis_kpt_idx] # (?,f,j,3)->(?,f,3)
#         yx_plane_vecs = pose_kpts[:,:,:,plane_kpt_idx] # (?,f,j,3)->(?,f,3)
#         xyz_comparable_vecs = self.broadcast_indexed_list_assign(fbj_idx, xy_axis_vecs, yx_plane_vecs) # (?,f,2,3)
#
#         # Define new x-axis, y-axis, and z-axis
#         xy_axis_uvecs = guard_div1(xy_axis_vecs, torch.linalg.norm(xy_axis_vecs, dim=-1, keepdim=True)) * self.xy_axis_dirs[:,:,fbj_idx] # (?,f,3)
#
#         z_axis_vecs = torch.cross(xyz_comparable_vecs[:,:,self.z_axis_a_idxs[fbj_idx]], # (?,f,3)
#                                   xyz_comparable_vecs[:,:,self.z_axis_b_idxs[fbj_idx]], dim=-1) # (?,f,3) -> (?,f,3)
#         xyz_comparable_vecs = self.broadcast_indexed_list_assign(fbj_idx, xy_axis_vecs, yx_plane_vecs, z_axis_vecs) # (?,f,3,3)
#         z_axis_uvecs = guard_div1(z_axis_vecs, torch.linalg.norm(z_axis_vecs, dim=-1, keepdim=True)) # (?,f,j,3)
#
#         yx_axis_vecs = torch.cross(xyz_comparable_vecs[:,:,self.yx_axis_a_idxs[fbj_idx]], # (?,f,3)
#                                    xyz_comparable_vecs[:,:,self.yx_axis_b_idxs[fbj_idx]], dim=-1) # (?,f,3) -> (?,f,3)
#         yx_axis_uvecs = guard_div1(yx_axis_vecs, torch.linalg.norm(yx_axis_vecs, dim=-1, keepdim=True)) * self.yx_axis_dirs[:,:,fbj_idx] # (?,f,3)
#
#         # Derive frame rotation matrix from unit vec axis
#         rotation_mtxs_3x3 = self.broadcast_indexed_list_assign(fbj_idx, xy_axis_uvecs, yx_axis_uvecs, z_axis_uvecs) # (?,f,3,3)
#
#         if self.validate_ops:
#             assert(are_valid_numbers(xy_axis_uvecs) and are_unit_vectors(xy_axis_uvecs))
#             assert(are_valid_numbers(z_axis_uvecs) and are_unit_vectors(z_axis_uvecs))
#             assert(are_valid_numbers(yx_axis_uvecs) and are_unit_vectors(yx_axis_uvecs))
#             # assert(are_rotation_matrices(rotation_mtxs_4x4)), 'Contains a non-orthogonal rotation matrix'
#             assert(are_rotation_matrices(rotation_mtxs_3x3)), 'Contains a non-orthogonal rotation matrix'
#             assert(are_proper_rotation_matrices(rotation_mtxs_3x3, self.nr_fb_idxs)), 'Contains an improper (reflection) matrix'
#
#         # apply rotation transformation to all keypoints
#         fb_aligned_pose_kpts = torch_matdot(rotation_mtxs_3x3, pose_kpts) # (?,f,j,3)
#         return fb_aligned_pose_kpts
#
#     def broadcast_indexed_list_assign(self, fbj_idx, xy_vecs_bxfx3, yx_vecs_bxfx3, z_vecs_bxfx3=None):
#         '''
#         xy_vecs_bxfx3:(?,f,3), yx_vecs_bxfx3:(?,f,3), z_vecs_bxfx3:(?,f,3) or None
#         Equivalent to:
#         xyz_vector_bxfx3x3 = tf.zeros((self.n_bsz,self.n_frm,3,3), dtype=tf.dtypes.float32) # (?,f,3,3)
#         xyz_vector_bxfx3x3[:,:,self.xy_idxs] = xy_vecs_bxfx3
#         xyz_vector_bxfx3x3[:,:,self.yx_idxs] = yx_vecs_bxfx3
#         xyz_vector_bxfx3x3[:,:,self.z_idx] = z_vecs_bxfx3
#         Returns: tensor of size (?,f,k,3), where k is 2 or 3
#         '''
#         if self.validate_ops:
#             assert(xy_vecs_bxfx3.shape==yx_vecs_bxfx3.shape) , 'xy_vecs_bxfx3:{}'.format(xy_vecs_bxfx3.shape)
#             assert(z_vecs_bxfx3 is None or yx_vecs_bxfx3.shape==z_vecs_bxfx3.shape), 'yx_vecs_bxfx3:{}'.format(yx_vecs_bxfx3.shape)
#
#         k_axs = 2 if z_vecs_bxfx3 is None else 3 # k==2->{xy,yx}, k==3->{xy,yx,z}
#
#         xy_idx = self.xy_idxs[fbj_idx]
#         yx_idx = self.yx_idxs[fbj_idx]
#         if self.validate_ops:
#             jnt_xyz_idxs = np.array([xy_idx,yx_idx,self.z_idx]) # xy, yx, & z idx are unique and each, one of 0 1 & 2
#             assert(np.intersect1d(np.arange(0, 3), jnt_xyz_idxs).shape[0]==3), 'jnt_xyz_idxs:{}'.format(jnt_xyz_idxs)
#
#         jnt_xyz_vector_bxfx3_list = [] # k*(?,f,3)
#         for current_idx in range(k_axs):
#             if xy_idx==current_idx: jnt_xyz_vector_bxfx3_list.append(xy_vecs_bxfx3) # (?,f,3)
#             elif yx_idx==current_idx: jnt_xyz_vector_bxfx3_list.append(yx_vecs_bxfx3) # (?,f,3)
#             else: jnt_xyz_vector_bxfx3_list.append(z_vecs_bxfx3) # (?,f,3)
#
#         xyz_vector_bxfxkx3 = torch.stack(jnt_xyz_vector_bxfx3_list, dim=2) # k*(?,f,3)-->(?,f,k,3)
#         return xyz_vector_bxfxkx3


def bpc_likelihood_func(bone_ratio_prop, variance, exponent_coefs, ratio_mean):
    # bone_ratio_prop->(?,f,r), variance->(1,1,r), exponent_coefs->(1,1,r), ratio_mean->(1,1,r)
    mean_centered = bone_ratio_prop - ratio_mean # (?,f,r)
    exponent = torch_t(-1/2) * guard_div1(torch.square(mean_centered), variance) # (?,f,r)
    likelihoods = exponent_coefs * torch.exp(exponent) # (?,f,r)
    #assert(likelihoods_are_non_negative(likelihoods))
    return likelihoods # (?,f,r)

def jmc_likelihood_func(joint_freelimb_uvec, uvec_means, inv_covariance, exponent_coefs):
    # covariance_matrices->(1,1,j,k,k), uvec_means->(1,1,j,k,1)
    x = torch.unsqueeze(joint_freelimb_uvec, dim=4) # (?,f,j,k)->(?,f,j,k,1)
    mean_centered = x - uvec_means # (?,f,j,k,1)
    mean_centered_T = torch.transpose(mean_centered, 4, 3) # (?,f,j,1,k)
    exponent = torch_t(-1/2) * torch.matmul(torch.matmul(mean_centered_T, inv_covariance), mean_centered)
    likelihoods = exponent_coefs * torch.exp(exponent) # (?,f,j,1,1)
    #assert(likelihoods_are_non_negative(likelihoods))
    return likelihoods[:,:,:,0,0] # (?,f,j,1,1)-->(?,f,j)


def log_likelihood_loss_func(likelihoods, function_modes, logli_mean, logli_std, logli_min, logli_span,
                             nilm_wgts, logli_spread, log_of_spread, log_lihood_eps,
                             move_up_const, ret_move_const=False):
    logli_func_mode, norm_logli_func_mode, reverse_logli_func_mode, shift_logli_func_mode = function_modes

    # compute log-likelihood
    if logli_func_mode==1: # shift and spread the range of log-likelihood
        log_likelihoods = torch.log(likelihoods + logli_spread) - log_of_spread # same as log(x+1) when spread==1
    elif logli_func_mode==2: # compute log of inverse log-likelihood
        log_likelihoods = torch.log(logli_spread / (likelihoods + logli_spread))
    else: # compute vanilla log of likelihood
        log_likelihoods = torch.log(likelihoods) # may produce nans when likelihood_values<1

    # normalize log-likelihood
    if norm_logli_func_mode==1: # normalized mean-&-std log-likelihood (nmsl)
        log_likelihoods = (log_likelihoods - logli_mean) / logli_std
    elif norm_logli_func_mode==2: # normalized min-max-scale log-likelihood (nmml)
        log_likelihoods = (log_likelihoods - logli_min) / logli_span
    elif norm_logli_func_mode==3: # normalized inverse of log-likelihood mean weighted log-likelihood (nilm)
        log_likelihoods = log_likelihoods * nilm_wgts

    # reverse log-likelihood
    if reverse_logli_func_mode==1: # negative log-likelihood
        log_likelihoods = -log_likelihoods
    elif reverse_logli_func_mode==2: # inverse log-likelihood
        log_likelihoods = log_lihood_eps / (log_likelihoods + log_lihood_eps)

    # log-likelihood vertical shift
    if shift_logli_func_mode==1: # log-likelihood plus some constant to move into (+)quadrant
        log_likelihoods = move_up_const + log_likelihoods

    if ret_move_const: return -log_likelihoods
    return log_likelihoods #torch.mean(log_likelihoods)



def orth_project_and_scaledown_align(poses_3d, pred_3dto2d_scale, target_rootkpts, root_kpt_idx, mode=-1):
    orth_proj_poses_2d = pred_3dto2d_scale * poses_3d[:,:,:,:2] # (?,f,j,2)
    if mode==-1:
        # Option-1: move projected-3dto2d-pose target-2d-pose location.
        # Tolerant to 3D pose not rooted at origin (Lesser error).
        translate_pose_vec = target_rootkpts - orth_proj_poses_2d[:,:,[root_kpt_idx]] # (?,f,1,2)
        trans_orth_proj_poses_2d = orth_proj_poses_2d + translate_pose_vec # (?,f,j,2)
    elif mode==-2:
        # Option-2: move projected-3dto2d-pose target-2d-pose location.
        # Strict to 3D pose not rooted at origin. Expects 3D poses' root-kpt is at origin
        trans_orth_proj_poses_2d = orth_proj_poses_2d + target_rootkpts # (?,f,j,2)
    elif mode==-3:
        # Option-3: for when target-2d-pose is moved to origin
        # Tolerant to 3D pose not rooted at origin (Lesser error).
        trans_orth_proj_poses_2d = orth_proj_poses_2d - orth_proj_poses_2d[:,:,[root_kpt_idx]] # (?,f,1,2)
    else: # mode==-4
        #assert(mode==-4), 'mode:{}'.format(mode)
        # Option-4: for when target-2d-pose is moved to origin
        # Strict to 3D pose not rooted at origin. Expects 3D poses' root-kpt is at origin
        trans_orth_proj_poses_2d = orth_proj_poses_2d
    #assert(torch.all(torch.isclose(trans_orth_proj_poses_2d[:,:,[root_kpt_idx]], target_rootkpts, atol=1e-03)))
    return trans_orth_proj_poses_2d

def scaleup_align(poses_2d, pred_2dto3d_scale, root_kpt_idx):
    rooted_poses_2d = poses_2d - poses_2d[:,:,[root_kpt_idx]] # (?,f,j,2)
    return pred_2dto3d_scale * rooted_poses_2d # (?,f,j,2)

def tc_scale_normalize(predicted, target, root_kpt_idx, pad, ret_scl=True):
    # scaled to match target 2d poses
    #assert(predicted.shape==target.shape), '{} vs {}'.format(predicted.shape, target.shape) # (?,f,p,3)

    # select middle 2d-pose(s) corresponding to 3d-pose
    if pad > 0:
        target = target[:, pad:-pad, :, :2].contiguous() # (?,f,j,2)
    else: target = target[:, :, :, :2].contiguous() # (?,f,j,2)

    # Added by Lawrence on 05/22/22. Recommended to get best scale factor
    rooted_pred_2d = predicted - predicted[:,:,[root_kpt_idx]]
    rooted_targ_2d = target - target[:,:,[root_kpt_idx]]

    norm_predicted = torch.mean(torch.sum(rooted_pred_2d**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(rooted_targ_2d*rooted_pred_2d, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted # (?,1,1,1)
    if ret_scl: return scale
    return scale * predicted

def np_scale_normalize(predicted, target):
    assert (predicted.shape==target.shape), '{} vs {}'.format(predicted.shape, target.shape) # (?,f,p,3)
    # Added by Lawrence on 05/22/22. Recommended to get best scale factor
    predicted -= predicted[:,:,[0]]
    target -= target[:,:,[0]]

    norm_predicted = np.mean(np.sum(predicted**2, axis=3, keepdims=True), axis=2, keepdims=True)
    norm_target = np.mean(np.sum(target*predicted, axis=3, keepdims=True), axis=2, keepdims=True)
    scale = norm_target / norm_predicted # (?,1,1,1)
    print('scale.shape >> {}'.format(scale.shape))
    return scale * predicted