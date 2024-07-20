# -*- coding: utf-8 -*-
# @Time    : 4/28/2021 10:53 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : rbo_transform_tc.py
# @Software: videopose3d

import torch
import torch.nn as nn
import numpy as np

from agents.helper import processor
# processor = 'cpu'#"cpu"
# if torch.cuda.is_available():
#     processor = 'cuda'#"cuda:0"


def to_numpy(torch_tensor):
    return torch_tensor.detach().cpu().clone().numpy()

def guard_div(tensor_1, tensor_2, eps=1e-08):
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
    # #assert(tensor_1.dtype==torch.float32), 'tensor_1 dtype: {}'.format(tensor_1.dtype)
    # #assert(tensor_2.dtype==torch.float32), 'tensor_2 dtype: {}'.format(tensor_2.dtype)
    # division = torch.div(tensor_1, tensor_2)
    # safe_div = torch.nan_to_num(division, nan=0.0)
    safe_div = torch.div(tensor_1, tensor_2.clamp(min=eps))
    return safe_div

def guard_sqrt(tsr, eps=1e-08):
    sqrt_clamped_tsr = torch.sqrt(tsr.clamp(min=eps))
    return sqrt_clamped_tsr

def torch_t(num, dtype=torch.float32, gradient=False):
    return torch.tensor(num, dtype=dtype, device=torch.device(processor), requires_grad=gradient)

def torch_square(tensor):
    return tensor*tensor

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
    # Checks if the matrices of R:(?,f,j,4,4) at last two axis are valid rotation matrices.
    #assert(R.ndim==5), 'R should be ?xfxjx4x4, but ndim=={}'.format(R.ndim)
    assert(R.shape[3:]==(4,4)), 'Rotation should be 4x4 matrix, not {}'.format(R.shape[3:])
    Rt = torch.transpose(R, 4, 3)
    shouldBeIdentity = torch.matmul(Rt, R)
    identity_4x4 = torch.eye(4, dtype=torch.float32, device=torch.device(processor))
    are_identity = torch.isclose(shouldBeIdentity, identity_4x4, atol=atol) # was atol=1e-04
    all_are_identity = torch.all(are_identity)
    if not all_are_identity:
        are_identity = torch.all(torch.all(are_identity, dim=-1), dim=-1)
        are_not_identity = torch.flatten(torch.logical_not(are_identity))
        not_identity_tsr = shouldBeIdentity.view((-1,4,4))[are_not_identity,:,:]
        print('\n[Assert Warning] {:,} (or {:.2%}) of R^T and R matmul are not identity. Some values differ by more than {}'
              '\n\tTherefore the rotation matrices are not orthogonal and improper\n{}\n'.
              format(not_identity_tsr.shape[0], not_identity_tsr.shape[0]
                     / are_not_identity.shape[0], atol, not_identity_tsr))
    return all_are_identity

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


def torch_atan2(numerator_tensor, denominator_tensor):
    '''
    arctan2 function is a piece-wise arctan function and is only differentiable
    when (n)numerator!=0 or (d)denominator>0 see https://en.wikipedia.org/wiki/Atan2
    gradient when d>0 or n!=0
      partial_derivative_wrt_d = -n / (d^2 + n^2)
      partial_derivative_wrt_n =  d / (d^2 + n^2)
    gradient when d==0 and n!=0
      partial_derivative_wrt_d = partial_derivative_wrt_n = 0
    gradient when d==0 and n==0
      partial_derivative_wrt_d = partial_derivative_wrt_n = undefined (unless hard-reset to 0)
    '''
    # TODO: test assertions
    #d_gt_zero = denominator_tensor > torch_t(1e-3) # d greater than 0
    #n_not_zero = torch.logical_not(torch.abs(numerator_tensor) < torch_t(1e-3))
    #assert#(torch.all(torch.logical_or(d_gt_zero, n_not_zero)))
    atan2_nd = torch.atan2(numerator_tensor, denominator_tensor)
    #atan2_nd = torch.where(torch.logical_not(torch.logical_or(d_gt_zero, n_not_zero)), torch_t(0), atan2_nd)
    return atan2_nd


def torch_cross(a_vec_tensors, b_vec_tensors, axis=-1):
    # cross product of vectors in dim=2
    # equivalent to np.cross(a_vec_tensors, b_vec_tensors)
    # a_vec_tensor-->(?,f,3)
    # b_vec_tensor-->(?,f,3)
    cross_prod = torch.cross(a_vec_tensors, b_vec_tensors, dim=axis) # dim=2
    # i_components = a_vec_tensors[:,:,1]*b_vec_tensors[:,:,2] - a_vec_tensors[:,:,2]*b_vec_tensors[:,:,1] # (?,f)
    # j_components = a_vec_tensors[:,:,2]*b_vec_tensors[:,:,0] - a_vec_tensors[:,:,0]*b_vec_tensors[:,:,2] # (?,f)
    # k_components = a_vec_tensors[:,:,0]*b_vec_tensors[:,:,1] - a_vec_tensors[:,:,1]*b_vec_tensors[:,:,0] # (?,f)
    # cross_prod = torch.stack((i_components, j_components, k_components), dim=2) # (?,f,3)
    return cross_prod

def torch_vecdot(a_vec_tensors, b_vec_tensors, keepdim=False, ndims=3, vsize=3):
    # dot product of vectors in dim=2
    # equivalent to np.dot(a_vec_tensors, b_vec_tensors)
    dot_prod_1 = torch.sum(a_vec_tensors * b_vec_tensors, dim=-1, keepdim=keepdim)
    # if ndims==3:
    #     # a_vec_tensor-->(?,f,3)
    #     # b_vec_tensor-->(?,f,3)
    #     dot_prod_2 = a_vec_tensors[:,:,0] * b_vec_tensors[:,:,0]
    #     for v_comp_idx in range(1, vsize):
    #         dot_prod_2 += a_vec_tensors[:,:,v_comp_idx] * b_vec_tensors[:,:,v_comp_idx]
    # else: # ndims==4:
    #     # a_vec_tensor-->(?,f,m,3)
    #     # b_vec_tensor-->(?,f,n,3)
    #     dot_prod_2 = a_vec_tensors[:,:,:,0] * b_vec_tensors[:,:,:,0] + \
    #                     a_vec_tensors[:,:,:,1] * b_vec_tensors[:,:,:,1] + \
    #                     a_vec_tensors[:,:,:,2] * b_vec_tensors[:,:,:,2] + \
    #                     a_vec_tensors[:,:,:,3] * b_vec_tensors[:,:,:,3]
    # assert (torch.equal(dot_prod_1, dot_prod_2)), 'dot_prod_1:\n{}\ndot_prod_2:\n{}'.format(dot_prod_1, dot_prod_2)
    return dot_prod_1 # (?,f) or (?,f,n)

def torch_matdot(mtx_tensor, vec_tensors, row_ax=-2, col_ax=-1):
    # dot product between matrices in dim=2&3 and vectors in dim=3
    # equivalent to np.dot(mtx_tensor, vec_tensors.T).T
    # mtx_tensor-->(?,f,4,4)
    # vec_tensor-->(?,f,n,4)
    mat_dot = torch.transpose(torch.matmul(mtx_tensor, torch.transpose(vec_tensors, col_ax, row_ax)), col_ax, row_ax)
    # row0_dot_prod = torch_vecdot(mtx_tensor[:,:,[0],:], vec_tensors, ndims=4, vsize=4) # (?,f,n)
    # row1_dot_prod = torch_vecdot(mtx_tensor[:,:,[1],:], vec_tensors, ndims=4, vsize=4) # (?,f,n)
    # row2_dot_prod = torch_vecdot(mtx_tensor[:,:,[2],:], vec_tensors, ndims=4, vsize=4) # (?,f,n)
    # row3_dot_prod = torch_vecdot(mtx_tensor[:,:,[3],:], vec_tensors, ndims=4, vsize=4) # (?,f,n)
    # mat_dot = torch.stack((row0_dot_prod, row1_dot_prod, row2_dot_prod, row3_dot_prod), dim=3) # (?,f,n,4)
    return mat_dot

def torch_matmul(a_mtx_tensor, b_mtx_tensor, n=3):
    # matrix multiplication between two matrices
    # equivalent to np.matmul(a_mtx_tensor, b_mtx_tensor)
    # a_mtx_tensor-->(?,f,n,n)
    # b_mtx_tensor-->(?,f,n,n)
    mat_mul = torch.matmul(a_mtx_tensor, b_mtx_tensor)
    # row_list, col_list = [], []
    # for i in range(n):
    #     for j in range(n):
    #         mij_dot_prod = torch_vecdot(a_mtx_tensor[:,:,i,:], b_mtx_tensor[:,:,:,j], vsize=n) # (?,f)
    #         col_list.append(mij_dot_prod)
    #
    #     col_tensor = torch.stack(col_list, dim=2) # (?,f,n)
    #     row_list.append(col_tensor)
    #     col_list.clear()
    #
    # mat_mul = torch.stack(row_list, dim=2) # (?,f,n,n)
    # del row_list, col_list
    return mat_mul


class FreeBoneOrientation(nn.Module):

    def __init__(self, batch_size, xy_yx_axis_dirs=None, z_axis_ab_idxs=None, yx_axis_ab_idxs=None, xy_yx_idxs=None,
                 quad_uvec_axes1=None, quad_uvec_axes2=None, plane_proj_mult=None, hflip_multiplier=None, n_frames=1,
                 n_fb_joints=16, pivot_kpt_idx=0, xy_axis_kpt_idx=1, yx_plane_kpt_idx=2, free_kpt_idx=3, z_idx=2,
                 rot_tfm_mode=0, ret_mode=0, invert_pose=False, validate_ops=False, **kwargs):
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
            self.hflip_multiplier = hflip_multiplier # (1,1,j,1,3)
            #self.xy_idxs = xy_yx_idxs[0] # xy
            self.two_tsr = torch_t(2.)
            self.one_tsr = torch_t(1.)
            self.half_tsr = torch_t(0.5)
            self.zero_tsr = torch_t(0.0)
            self.epsilon_tsr = torch_t(1e-08)

        self.n_frames = n_frames
        self.n_fb_joints = n_fb_joints
        self.pivot_kpt_idx = pivot_kpt_idx
        self.xy_axis_kpt_idx = xy_axis_kpt_idx
        self.yx_plane_kpt_idx = yx_plane_kpt_idx
        self.free_kpt_idx = free_kpt_idx
        #self.scale_up = torch_t(scale)
        #self.scale_down = torch_t(1./scale)
        self.ret_mode = ret_mode
        self.rot_tfm_mode = rot_tfm_mode
        self.validate_ops = validate_ops
        self.invert_pose = invert_pose

        self.oat_atol1 = 1e-05 if rot_tfm_mode==0 else 7.5e-05 # tolerance for on-axis-test
        self.oat_atol2 = 1e-07 if rot_tfm_mode==0 else 9.5e-06 # tolerance for on-axis-test

        self.invert_pose_mult = torch.reshape(torch_t([-1,-1,1]), (1,1,1,1,3))
        self.jnt_idxs = np.arange(self.n_fb_joints) # (j,) do NOT remove, very necessary!
        self.build_for_batch(batch_size)

    def build_for_batch(self, batch_size):
        # input_shape:(?,f,j,4,3)
        self.batch_size = batch_size
        # todo: time optimization. Instead of recreating constant tensors everytime batch_size changes,
        #  create constant tensors once with initial self.batch_size,
        #  then use 'bs' (batch size for each step) to slice tensor: ie self.zero_tsr_bxfx3[:bs]
        if self.rot_tfm_mode==0: # for rotation matrix
            self.zero_tsr_bxfx3 = torch.zeros((self.batch_size,self.n_frames,3),
                                              dtype=torch.float32, device=torch.device(processor)) # (?,f,3)
            self.zero_tsr_bxfxjx1 = torch.zeros((self.batch_size,self.n_frames,self.n_fb_joints,1),
                                                dtype=torch.float32, device=torch.device(processor)) # (?,f,j,1)
            self.zero_tsr_bxfxjx4 = torch.zeros((self.batch_size,self.n_frames,self.n_fb_joints,4),
                                                dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4)
            self.zero_tsr_bxfxjx3x1 = torch.zeros((self.batch_size,self.n_frames,self.n_fb_joints,3,1),
                                                  dtype=torch.float32, device=torch.device(processor)) # (?,f,j,3,1)
            self.ones_tsr_bxfxjx4x1 = torch.ones((self.batch_size,self.n_frames,self.n_fb_joints,4,1),
                                                 dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4,1)
            self.homg_tsr_bxfxjx1x4 = torch.tile(torch_t([[[[0,0,0,1]]]], dtype=torch.float32),
                                                 dims=(self.batch_size,self.n_frames,self.n_fb_joints,1,1)) # (?,f,j,1,4)

            identity_4x4 = torch.eye(4, dtype=torch.float32, device=torch.device(processor)) # (4,4)
            self.identity_bxfxjx4x4 = \
                torch.broadcast_to(identity_4x4, [self.batch_size,self.n_frames,self.n_fb_joints,4,4]) # (4,4)->(?,f,j,4,4)
            assert (torch.all(torch.eq(self.identity_bxfxjx4x4, identity_4x4))), "{}".format(self.identity_bxfxjx4x4)

            if self.validate_ops:
                self.axis_2x2_matrix = torch.zeros((self.batch_size,self.n_frames,self.n_fb_joints,2,2),
                                                   dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4,4)
                self.axis_2x2_matrix[:,:,self.jnt_idxs,0,self.xy_idxs] = self.xy_axis_dirs[0,0,:,0] #torch_t(1.)
                self.axis_test_uvecs = torch.zeros((self.batch_size,self.n_frames,self.n_fb_joints,3),
                                                   dtype=torch.float32, device=torch.device(processor)) # (?,f,j,3)
                self.axis_test_uvecs[:,:,self.jnt_idxs,self.xy_idxs] = self.xy_axis_dirs[0,0,:,0]

        else: # rot_tfm_mode==1 for quaternion
            self.quadrant_1st_axis_uvec = torch.tile(self.quad_uvec_axes1, dims=(self.batch_size,self.n_frames,1,1))  # (?,f,j,3)
            self.quadrant_2nd_axis_uvec = torch.tile(self.quad_uvec_axes2, dims=(self.batch_size,self.n_frames,1,1))  # (?,f,j,3)
            if self.validate_ops:
                self.axis_2x2_matrix = torch.zeros((self.batch_size,self.n_frames,self.n_fb_joints,2,2),
                                                   dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4,4)
                self.axis_2x2_matrix[:,:,:,0,:] = self.quadrant_1st_axis_uvec[:,:,:,:2]
                self.axis_test_uvecs = self.quadrant_1st_axis_uvec

        # Needed for tests and assertions
        if self.validate_ops:
            self.planar_4x4_matrix = torch.zeros((self.batch_size,self.n_frames,self.n_fb_joints,4,4),
                                                dtype=torch.float32, device=torch.device(processor)) # (?,f,j,4,4)
            self.planar_4x4_matrix[:,:,[0,1,3,3,3,3],[1,2,0,1,2,3]] = torch_t(1.)
            #self.zero_tsr = torch_t(0.)
            self.pi_tsr = torch_t(np.pi)

    def forward(self, input_tensor, **kwargs):
        if input_tensor.shape[0]!=self.batch_size: self.build_for_batch(input_tensor.shape[0])
        if self.invert_pose: input_tensor = input_tensor * self.invert_pose_mult

        # Step 0: Translate quadruplet keypoints such that pivot is on origin
        quadruplet_kpts = input_tensor - input_tensor[:,:,:,[self.pivot_kpt_idx],:3]
        ##quadruplet_kpts = quadruplet_kpts / 1000

        tfm_meta_1, tfm_meta_2, rot_quadruplet_kpts = self.rotation_alignment_func(quadruplet_kpts)

        # Compute free limb vectors
        free_bone_vecs = rot_quadruplet_kpts[:,:,:,self.free_kpt_idx] # (?,f,j,3)
        free_bone_uvecs = guard_div(free_bone_vecs, torch.linalg.norm(free_bone_vecs, dim=3, keepdim=True)) # (?,f,j,3)

        if self.validate_ops:
            assert(are_valid_numbers(rot_quadruplet_kpts) and are_valid_numbers(free_bone_uvecs))
            assert(pivot_kpts_is_at_origin(rot_quadruplet_kpts[:,:,:,self.pivot_kpt_idx]))
            assert(axis_kpts_lie_on_axis(rot_quadruplet_kpts[:,:,:,self.xy_axis_kpt_idx], self.axis_test_uvecs,
                                         self.axis_2x2_matrix, atol1=self.oat_atol1, atol2=self.oat_atol2))
            assert(plane_kpts_line_on_xyplane(rot_quadruplet_kpts[:,:,:,self.yx_plane_kpt_idx], self.planar_4x4_matrix))

        if self.ret_mode==-1:
            return to_numpy(input_tensor[:,:,:,[self.pivot_kpt_idx],:3]), to_numpy(tfm_meta_1), \
                   to_numpy(tfm_meta_2), to_numpy(free_bone_uvecs)
        elif self.ret_mode==1:
            return free_bone_uvecs, free_bone_vecs # (?,f,j,3)
        return free_bone_uvecs # (?,f,j,3) # for ret_mode == 0

    def quaternion_rotation(self, quadruplet_kpts):
        # Step 1: Rotate to align Axis-kpt (or Axis-Bone) with the pre-configured X or Y axis
        xy_axis_vecs = quadruplet_kpts[:,:,:,self.xy_axis_kpt_idx,:3] # (?,f,j,4,3)->(?,f,j,3)
        # cos_thetas_1 = guard_div(torch_vecdot(xy_axis_vecs, self.quadrant_1st_axis_uvec, keepdim=True),
        #                          torch.linalg.norm(xy_axis_vecs, dim=-1, keepdim=True)) # (?,f,j,1)
        quat_vec1 = self.rotate_vecb2veca_quaternion(xy_axis_vecs, self.quadrant_1st_axis_uvec) # (?,f,j,4)
        quad_kpts_1of2_align = self.quaternion_rotate_vecs(quadruplet_kpts, quat_vec1) # (?,f,j,4,3)

        # Step 2: Rotate to align Plane-kpt (or Plane-Bone) with the pre-configured XY-plane quadrant
        # todo: for better precision, instead of multiplying, create zero tensor and insert none zero components into tensor
        proj_yx_plane_vecs = quad_kpts_1of2_align[:,:,:,self.yx_plane_kpt_idx,:3] * self.plane_proj_mult # (?,f,j,3)
        # cos_thetas_2 = guard_div(torch_vecdot(proj_yx_plane_vecs, self.quadrant_2nd_axis_uvec, keepdim=True),
        #                          torch.linalg.norm(proj_yx_plane_vecs, dim=-1, keepdim=True)) # (?,f,j,1)
        quat_vec2 = self.rotate_vecb2veca_quaternion(proj_yx_plane_vecs, self.quadrant_2nd_axis_uvec,
                                                     parallel_axis_uvec=self.quadrant_1st_axis_uvec) # (?,f,j,4)
        quad_kpts_2of2_align = self.quaternion_rotate_vecs(quad_kpts_1of2_align, quat_vec2) # (?,f,j,4,3)

        if self.validate_ops:
            assert(kpts_are_on_axis_hemisphere(quad_kpts_2of2_align[:,:,:,self.yx_plane_kpt_idx], self.quadrant_2nd_axis_uvec))

        quad_kpts_2of2_align = quad_kpts_2of2_align * self.hflip_multiplier # horizontal-flip for grouping symmetric fb-joints

        return quat_vec1, quat_vec2, quad_kpts_2of2_align


    def matrix_rotation(self, quadruplet_kpts):
        quadruplet_kpts_hom = torch.cat([quadruplet_kpts, self.ones_tsr_bxfxjx4x1], dim=4) # (?,f,j,4,3)+(?,f,j,4,1)->(?,f,j,4,4)

        # Get limb vectors from pairs of keypoint positions
        xy_axis_vecs = quadruplet_kpts[:,:,:,self.xy_axis_kpt_idx,:3] # (?,f,j,4,3)->(?,f,j,3)
        yx_plane_vecs = quadruplet_kpts[:,:,:,self.yx_plane_kpt_idx,:3] # (?,f,j,4,3)->(?,f,j,3)
        xyz_comparable_vecs = self.broadcast_indexed_list_assign(xy_axis_vecs, yx_plane_vecs) # (?,f,j,3,3)

        # Define new x-axis, y-axis, and z-axis
        xy_axis_uvecs = guard_div(xy_axis_vecs, torch.linalg.norm(xy_axis_vecs, dim=3, keepdim=True)) * self.xy_axis_dirs # (?,f,j,3)

        z_axis_vecs = torch.cross(xyz_comparable_vecs[:,:,self.jnt_idxs,self.z_axis_a_idxs], # (?,f,j,3)
                                  xyz_comparable_vecs[:,:,self.jnt_idxs,self.z_axis_b_idxs], dim=-1) # (?,f,j,3) -> (?,f,j,3)
        xyz_comparable_vecs = self.broadcast_indexed_list_assign(xy_axis_vecs, yx_plane_vecs, z_axis_vecs) # (?,f,j,3,3)
        z_axis_uvecs = guard_div(z_axis_vecs, torch.linalg.norm(z_axis_vecs, dim=3, keepdim=True)) # (?,f,j,3)

        yx_axis_vecs = torch.cross(xyz_comparable_vecs[:,:,self.jnt_idxs,self.yx_axis_a_idxs], # (?,f,j,3)
                                   xyz_comparable_vecs[:,:,self.jnt_idxs,self.yx_axis_b_idxs], dim=-1) # (?,f,j,3) -> (?,f,j,3)
        yx_axis_uvecs = guard_div(yx_axis_vecs, torch.linalg.norm(yx_axis_vecs, dim=3, keepdim=True)) * self.yx_axis_dirs # (?,f,j,3)

        # Derive frame rotation matrix from unit vec axis
        xyz_axis_unit_vecs = self.broadcast_indexed_list_assign(xy_axis_uvecs, yx_axis_uvecs, z_axis_uvecs) # (?,f,j,3,3)
        rotation_matrices = torch.cat([xyz_axis_unit_vecs, self.zero_tsr_bxfxjx3x1], dim=4) # (?,f,j,3,4)
        rotation_matrices = torch.cat([rotation_matrices, self.homg_tsr_bxfxjx1x4], dim=3) # (?,f,j,4,4)

        if self.validate_ops:
            assert(are_valid_numbers(xy_axis_uvecs) and are_unit_vectors(xy_axis_uvecs))
            assert(are_valid_numbers(z_axis_uvecs) and are_unit_vectors(z_axis_uvecs))
            assert(are_valid_numbers(yx_axis_uvecs) and are_unit_vectors(yx_axis_uvecs))
            assert(are_rotation_matrices(rotation_matrices)), 'Contain a non-rotation (non-orthogonal) matrix'

        # Using frame rotation matrix, transform position of non-origin translated
        # keypoints (in camera coordinated) to be represented in the joint frame
        kpts_wrt_pvt_frms_hom = torch_matdot(rotation_matrices, quadruplet_kpts_hom)
        kpts_wrt_pvt_frms = guard_div(kpts_wrt_pvt_frms_hom[:,:,:,:,:3], kpts_wrt_pvt_frms_hom[:,:,:,:,[3]]) # (?,f,j,4,3)

        return xyz_axis_unit_vecs, rotation_matrices, kpts_wrt_pvt_frms

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
        rot_axis_uvec = guard_div(vecb_x_veca, torch.linalg.norm(vecb_x_veca, dim=-1, keepdim=True))
        cos_theta = guard_div(torch_vecdot(vec_b, vec_a, keepdim=True),
                              torch.linalg.norm(vec_b, dim=-1, keepdim=True)) # (?,f,j,1)

        # # The derivative of torch.arccos is numerical unstable at -1 or 1,
        # # producing infinite values. Hence we clip values to open interval (-1,1)
        # # https://discuss.pytorch.org/t/numerically-stable-acos-dot-product-derivative/12851/2
        # thetas = torch.arccos(cos_theta.clamp(-1.0 + eps, 1.0 - eps)) # (?,f,j,1)
        # half_theta = thetas * self.half_tsr # equivalent to theta/2 (?,f,j,1)
        # cos_half_theta = torch.cos(half_theta) # q_w:(?,f,j,1)
        # sin_half_theta = torch.sin(half_theta) # (?,f,j,3)

        # Alternative method to compute cos(theta/2) and sin(theta/2)
        # cos(x/2) = sqrt((1+cos(x))/2) and sin(x/z) = sqrt((1-cos(x))/2) for 0<=x<=pi
        cos_half_theta = guard_sqrt(guard_div(self.one_tsr + cos_theta, self.two_tsr)) # q_w:(?,f,j,1)
        # todo: handle borderline case when cos_theta=-1 (ie. theta=0) >> no rotation
        #assert (are_valid_numbers(cos_half_theta2))
        #assert (torch.all(torch.isclose(cos_half_theta2, cos_half_theta1, atol=1e-07)))
        sin_half_theta = guard_sqrt(guard_div(self.one_tsr - cos_theta, self.two_tsr))
        # todo: handle borderline case when sin_theta=1 (ie. theta=pi or 180) >> rotate twice about axis by 90"
        #assert (are_valid_numbers(sin_half_theta2))
        #assert (torch.all(torch.isclose(sin_half_theta2, sin_half_theta1, atol=1e-07)))

        if self.validate_ops:
            assert (are_non_zero_length_vectors(vec_a, low_bound=1e-05))
            assert (torch.all(torch_t(-1.)<=cos_theta) and torch.all(cos_theta<=torch_t(1.))) # implies 0<=theta<=180
            #assert(angles_are_within_bounds(thetas, low_bound=self.zero_tsr, up_bound=self.pi_tsr))
            # note, rotation axis can be a null <0,0,0> vector which results in no rotations
            assert(are_valid_numbers(rot_axis_uvec))
            assert(are_valid_angle_axis(rot_axis_uvec, cos_theta))
            assert(are_parallel_to_axis_uvecs(rot_axis_uvec, parallel_axis_uvec))

        # if parallel_axis_uvec is not None:
        #     rot_axis_uvec = torch.round(rot_axis_uvec) # todo: why round?
        q_xyz = rot_axis_uvec * sin_half_theta # (?,f,j,3)
        return torch.cat([cos_half_theta, q_xyz], dim=-1) # (?,f,j,4)

    def quaternion_rotate_vecs(self, vectors_3d, quaternion_vecs):
        '''
        Rotate vectors_3d according to the quaternion_vecs
        Args:
            vectors_3d: (?,f,j,4,3) 3D pose vectors of quad-kpts
            quaternion_vecs: (?,f,j,4) corresponding quaternion vectors
        Returns: (?,f,j,4,3) transformed 3D pose vectors of quad-kpts after rotation
        '''
        quaternion_vecs = torch.tile(torch.unsqueeze(quaternion_vecs, dim=-2), (1,1,1,4,1)) # (?,f,j,4,4)
        uv = torch.cross(quaternion_vecs[..., 1:], vectors_3d, dim=-1) # (?,f,j,4,3)
        uuv = torch.cross(quaternion_vecs[..., 1:], uv, dim=-1) # (?,f,j,4,3)
        return vectors_3d + self.two_tsr * (quaternion_vecs[..., :1] * uv + uuv) # (?,f,j,4,3)

    # Test version until 06/11/2022
    # def forward(self, input_tensor, **kwargs):
    #     if input_tensor.shape[0]!=self.batch_size: self.build_for_batch(input_tensor.shape[0])
    #
    #     pivot_kpts = input_tensor[:,:,:,self.pivot_kpt_idx,:3] # (?,f,j,4,3)->(?,f,j,3)
    #     xy_axis_kpts = input_tensor[:,:,:,self.xy_axis_kpt_idx,:3] # (?,f,j,4,3)->(?,f,j,3)
    #     yx_plane_kpts = input_tensor[:,:,:,self.yx_plane_kpt_idx,:3] # (?,f,j,4,3)->(?,f,j,3)
    #     quadruplet_kpts_hom = torch.cat([input_tensor, self.ones_tsr_bxfxjx4x1], dim=4) # (?,f,j,4,3)+(?,f,j,4,1)->(?,f,j,4,4)
    #
    #     # Deduce translation vector from pivot joint. Frame origin will be translated to pivot joint
    #     neg_pivot_kpts_4 = torch.cat([-pivot_kpts, self.zero_tsr_bxfxjx1], dim=3) # (?,f,j,4)
    #     neg_pivot_kpts_4x4 = torch.stack(
    #         (self.zero_tsr_bxfxjx4,self.zero_tsr_bxfxjx4,self.zero_tsr_bxfxjx4,neg_pivot_kpts_4), dim=4) # (?,f,j,4,4)
    #     translate_matrices = self.identity_bxfxjx4x4 + neg_pivot_kpts_4x4 # (?,f,j,4,4)
    #     #assert(are_translation_matrices(translate_matrices)), "Contain non-translation matrix\n{}".format(translate_matrices)
    #
    #     # Get limb vectors from pairs of keypoint positions
    #     xy_axis_vecs = xy_axis_kpts - pivot_kpts # (?,f,j,3)
    #     yx_plane_vecs = yx_plane_kpts - pivot_kpts # (?,f,j,3)
    #     xyz_comparable_vecs = self.broadcast_indexed_list_assign(xy_axis_vecs, yx_plane_vecs) # (?,f,j,3,3)
    #
    #     # Define new x-axis, y-axis, and z-axis
    #     xy_axis_uvecs = guard_div(xy_axis_vecs, torch.linalg.norm(xy_axis_vecs, dim=3, keepdim=True)) * self.xy_axis_dirs # (?,f,j,3)
    #     #assert(are_valid_numbers(xy_axis_uvecs)), 'xy_axis_uvecs:\n{}'.format(xy_axis_uvecs)
    #     #assert(are_unit_vectors(xy_axis_uvecs)), '|xy_axis_uvecs|:{}'.format(torch.linalg.norm(xy_axis_uvecs, dim=-1))
    #
    #     z_axis_vecs = torch.cross(xyz_comparable_vecs[:,:,self.jnt_idxs,self.z_axis_a_idxs], # (?,f,j,3)
    #                               xyz_comparable_vecs[:,:,self.jnt_idxs,self.z_axis_b_idxs]) # (?,f,j,3) -> (?,f,j,3)
    #     xyz_comparable_vecs = self.broadcast_indexed_list_assign(xy_axis_vecs, yx_plane_vecs, z_axis_vecs) # (?,f,j,3,3)
    #     z_axis_uvecs = guard_div(z_axis_vecs, torch.linalg.norm(z_axis_vecs, dim=3, keepdim=True)) # (?,f,j,3)
    #     #assert(are_valid_numbers(z_axis_uvecs)), 'z_axis_uvecs:\n{}'.format(z_axis_uvecs)
    #     #assert(are_unit_vectors(z_axis_uvecs)), '|z_axis_uvecs|:{}'.format(torch.linalg.norm(z_axis_uvecs, dim=-1))
    #
    #     yx_axis_vecs = torch.cross(xyz_comparable_vecs[:,:,self.jnt_idxs,self.yx_axis_a_idxs], # (?,f,j,3)
    #                                xyz_comparable_vecs[:,:,self.jnt_idxs,self.yx_axis_b_idxs]) # (?,f,j,3) -> (?,f,j,3)
    #     yx_axis_uvecs = guard_div(yx_axis_vecs, torch.linalg.norm(yx_axis_vecs, dim=3, keepdim=True)) * self.yx_axis_dirs # (?,f,j,3)
    #     #assert(are_valid_numbers(yx_axis_uvecs)), 'yx_axis_uvecs:\n{}'.format(yx_axis_uvecs)
    #     #assert(are_unit_vectors(yx_axis_uvecs)), '|yx_axis_uvecs|:{}'.format(torch.linalg.norm(yx_axis_uvecs, dim=-1))
    #
    #     # Derive frame rotation matrix from unit vec axis
    #     xyz_axis_unit_vecs = self.broadcast_indexed_list_assign(xy_axis_uvecs, yx_axis_uvecs, z_axis_uvecs) # (?,f,j,3,3)
    #     rotation_matrices = torch.cat([xyz_axis_unit_vecs, self.zero_tsr_bxfxjx3x1], dim=4) # (?,f,j,3,4)
    #     rotation_matrices = torch.cat([rotation_matrices, self.homg_tsr_bxfxjx1x4], dim=3) # (?,f,j,4,4)
    #     #assert(are_rotation_matrices(rotation_matrices)), 'Contain a non-rotation (non-orthogonal) matrix'
    #     transform_matrices = torch.matmul(rotation_matrices, translate_matrices) # (?,f,j,4,4)
    #
    #     # Using frame rotation matrix, transform position of non-origin translated
    #     # keypoints (in camera coordinated) to be represented in the joint frame
    #     kpts_wrt_pvt_frms_hom = torch_matdot(transform_matrices, quadruplet_kpts_hom)
    #     kpts_wrt_pvt_frms = guard_div(kpts_wrt_pvt_frms_hom[:,:,:,:,:3], kpts_wrt_pvt_frms_hom[:,:,:,:,[3]]) # (?,f,j,4,3)
    #     #assert(are_valid_numbers(kpts_wrt_pvt_frms)), 'kpts_wrt_pvt_frms:\n{}'.format(kpts_wrt_pvt_frms)
    #     assert (pivot_kpts_is_at_origin(kpts_wrt_pvt_frms[:,:,:,self.pivot_kpt_idx]))
    #     assert (axis_kpts_lie_on_axis(kpts_wrt_pvt_frms[:,:,:,self.xy_axis_kpt_idx], self.xy_axis_endpoints))
    #     assert (plane_kpts_line_on_xyplane(kpts_wrt_pvt_frms[:,:,:,self.yx_plane_kpt_idx], self.planar_4x4_matrix))
    #
    #     # Compute free limb vectors
    #     free_bone_vecs = kpts_wrt_pvt_frms[:,:,:,self.free_kpt_idx] - kpts_wrt_pvt_frms[:,:,:,self.pivot_kpt_idx] # (?,f,j,3)
    #     free_bone_uvecs = guard_div(free_bone_vecs, torch.linalg.norm(free_bone_vecs, dim=3, keepdim=True)) # (?,f,j,3)
    #     #assert(are_valid_numbers(free_bone_uvecs)), 'free_bone_uvecs:\n{}'.format(free_bone_uvecs)
    #
    #     if self.ret_mode==-1:
    #         return to_numpy(pivot_kpts), to_numpy(xyz_axis_unit_vecs), \
    #                to_numpy(transform_matrices), to_numpy(free_bone_uvecs)
    #     elif self.ret_mode==1:
    #         return free_bone_uvecs, free_bone_vecs # (?,f,j,3)
    #     return free_bone_uvecs, None # (?,f,j,3) # ret_mode == 0

    def broadcast_indexed_list_assign(self, xy_vecs_bxfxjx3, yx_vecs_bxfxjx3, z_vecs_bxfxjx3=None):
        '''
        xy_vecs_bxfxjx3:(?,j,3), yx_vecs_bxfxjx3:(?,j,3), z_vecs_bxfxjx3:(?,j,3) or None
        Equivalent to:
        jnt_idxs = np.arange(self.n_fb_joints)
        xyz_vector_bxfxjx3x3 = tf.zeros((self.batch_size,self.n_fb_joints,3,3), dtype=tf.dtypes.float32) # (?,j,3,3)
        xyz_vector_bxfxjx3x3[:,jnt_idxs,self.xy_idxs] = xy_vecs_bxfxjx3
        xyz_vector_bxfxjx3x3[:,jnt_idxs,self.yx_idxs] = yx_vecs_bxfxjx3
        xyz_vector_bxfxjx3x3[:,jnt_idxs,self.z_idx] = z_vecs_bxfxjx3
        '''
        if self.validate_ops:
            assert(xy_vecs_bxfxjx3.shape==yx_vecs_bxfxjx3.shape) , 'xy_vecs_bxfxjx3:{}'.format(xy_vecs_bxfxjx3.shape)
            assert(z_vecs_bxfxjx3 is None or yx_vecs_bxfxjx3.shape==z_vecs_bxfxjx3.shape), 'yx_vecs_bxfxjx3:{}'.format(yx_vecs_bxfxjx3.shape)

        xyz_vector_bxfx3x3_list = [] # j*(?,f,3,3)
        for j_idx in range(self.n_fb_joints):
            xy_idx = self.xy_idxs[j_idx]
            yx_idx = self.yx_idxs[j_idx]
            if self.validate_ops:
                jnt_xyz_idxs = np.array([xy_idx,yx_idx,self.z_idx])
                assert(np.intersect1d(np.arange(0, 3), jnt_xyz_idxs).shape[0]==3), 'jnt_xyz_idxs:{}'.format(jnt_xyz_idxs)

            jnt_xyz_vector_bxfx3_list = [] # 3*(?,f,3)
            while len(jnt_xyz_vector_bxfx3_list)<3:
                current_idx = len(jnt_xyz_vector_bxfx3_list)
                if xy_idx==current_idx: jnt_xyz_vector_bxfx3_list.append(xy_vecs_bxfxjx3[:,:,j_idx]) # (?,f,3)
                elif yx_idx==current_idx: jnt_xyz_vector_bxfx3_list.append(yx_vecs_bxfxjx3[:,:,j_idx]) # (?,f,3)
                else:
                    if z_vecs_bxfxjx3 is None: jnt_xyz_vector_bxfx3_list.append(self.zero_tsr_bxfx3)
                    elif self.z_idx==current_idx: jnt_xyz_vector_bxfx3_list.append(z_vecs_bxfxjx3[:,:,j_idx]) # (?,f,3)
            xyz_vector_bxfx3x3_list.append(torch.stack(jnt_xyz_vector_bxfx3_list, dim=2)) # 3'*(?,f,3)-->(?,f,3',3)
        xyz_vector_bxfxjx3x3 = torch.stack(xyz_vector_bxfx3x3_list, dim=2) # j*(?,f,3,3)-->(?,f,j,3,3)
        return xyz_vector_bxfxjx3x3



def bpc_likelihood_func(bone_ratio_prop, variance, exponent_coefs, ratio_mean):
    # bone_ratio_prop->(?,f,r), variance->(1,1,r), exponent_coefs->(1,1,r), ratio_mean->(1,1,r)
    mean_centered = bone_ratio_prop - ratio_mean # (?,f,r)
    exponent = torch_t(-1/2) * guard_div(torch.square(mean_centered), variance) # (?,f,r)
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


def log_likelihood_func(likelihoods, scale, epsilon):
    log_likelihoods = torch.log((likelihoods * scale) + epsilon)
    return -log_likelihoods

def sigm_log_likelihood_func(likelihoods, scale, epsilon):
    log_likelihoods = torch.log((likelihoods * scale) + epsilon)
    actv_likelihoods = torch.sigmoid(-log_likelihoods)
    return actv_likelihoods

def tanh_likelihood_func(likelihoods, scale):
    scaled_likelihoods = likelihoods * scale
    actv_likelihoods = torch.tanh(-scaled_likelihoods)
    return actv_likelihoods


def log_likelihood_loss_func(likelihoods, function_modes, logli_mean, logli_std, logli_min, logli_span,
                             nilm_wgts, logli_spread, log_of_spread, log_lihood_eps,
                             move_up_const, ret_move_const=False):
    logli_func_mode, norm_logli_func_mode, reverse_logli_func_mode, shift_logli_func_mode = function_modes

    # compute log-likelihood
    if logli_func_mode==1: # shift and spread the range of log-likelihood
        log_likelihoods = torch.log(likelihoods + logli_spread) - log_of_spread # same as log(x+1) when spread==1
    elif logli_func_mode==2: # compute log of inverse log-likelihood
        # logli_spread within [1-e5, 1] prevents nan values from division by 0
        log_likelihoods = torch.log(logli_spread / (likelihoods + logli_spread))
    else: # compute vanilla log of likelihood
        log_likelihoods = torch.log(likelihoods) # may produce nans when likelihood_values<1

    # normalize log-likelihood
    if norm_logli_func_mode==1: # normalized mean-&-std log-likelihood (nmsl)
        log_likelihoods = (log_likelihoods - logli_mean) / logli_std
    elif norm_logli_func_mode==2: # normalized min-max-scale log-likelihood (nmml)
        log_likelihoods = (log_likelihoods - logli_min) / logli_span
        # todo: clip [0,1] for values exceeding 1. and assert outcome is within range
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


def uvec_euclidean_dist_jmc(joint_freelimb_uvec, cluster_centroids, cluster_radius):
    # cluster_centroids->(1,1,j,c,3), cluster_radius->(1,1,j,c,1)
    x = torch.unsqueeze(joint_freelimb_uvec, dim=3) # (?,f,j,3)->(?,f,j,1,3)
    x2centroids = x - cluster_centroids # (?,f,j,c,3)
    euc_dist_2_ctrs = torch.linalg.norm(x2centroids, dim=4, keepdim=True) # (?,f,j,c,1)
    oor_dist_errs = euc_dist_2_ctrs - cluster_radius # (?,f,j,c,1) out-of-range error
    min_errs, min_idxs = torch.min(oor_dist_errs, dim=3) # (?,f,j,1)
    actv_min_errs = torch.relu(min_errs) # (?,f,j,1)
    return torch.mean(actv_min_errs) # (1,)


def tc_orthographic_projection(poses_3d, projection_mtx, homogenous_one):
    pose_tsr_shape = poses_3d.shape
    bs = pose_tsr_shape[0]
    if projection_mtx is None:
        projection_mtx = torch.zeros((bs,1,17,4,4), dtype=torch.float32, device=torch.device(processor)) # (?,f,17,4,4)
        projection_mtx[:,:,:,[0,1,3],[0,1,3]] =  1
        homogenous_one = torch.ones((bs,1,17,1), dtype=torch.float32, device=torch.device(processor)) # (?,f,17,1)
    assert (pose_tsr_shape[2:]==(17,3)), 'poses_3d.shape:{}'.format(pose_tsr_shape) # (?,f,17,3)
    assert (bs<=homogenous_one.shape[0]), '{} vs. {}'.format(bs, homogenous_one.shape)

    homg_3d_poses = torch.cat([poses_3d, homogenous_one[:bs]], dim=-1) # (?,f,17,4)
    homg_3d_poses = torch.unsqueeze(homg_3d_poses, dim=4) # (?,f,17,4,1)
    proj_2d_poses = torch.matmul(projection_mtx[:bs], homg_3d_poses)
    assert (homg_3d_poses.shape[1:]==(1,17,4,1)), 'homg_3d_poses.shape:{}'.format(homg_3d_poses.shape) # (?,f,17,3) # (?,f,17,4,1)
    assert (proj_2d_poses.shape[1:]==(1,17,4,1)), 'proj_2d_poses.shape:{}'.format(proj_2d_poses.shape) # (?,f,17,3) # (?,f,17,4,1)
    assert (torch.all(torch.isclose(proj_2d_poses[:,:,:,2,0], torch_t(0.), atol=1e-05))) #*!
    assert (torch.all(torch.isclose(proj_2d_poses[:,:,:,3,0], torch_t(1.), atol=1e-05)))
    return proj_2d_poses[:,:,:,:2,0] # (?,f,17,2)

def tc_scale_normalize(predicted, target):
    assert (predicted.shape==target.shape) # (?,f,p,3)
    # Added by Lawrence on 05/22/22. Recommended to get best scale factor
    predicted -= predicted[:,:,[0]]
    target -= target[:,:,[0]]

    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return scale * predicted