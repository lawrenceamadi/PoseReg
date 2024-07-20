# -*- coding: utf-8 -*-
# @Time    : 4/25/2021 9:05 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : pose_regs.py
# @Software: videopose3d

import os
import sys
import scipy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import softmax
from sklearn.cluster import DBSCAN, KMeans, MeanShift, SpectralClustering, AgglomerativeClustering

sys.path.append('../')
from agents.visuals import plot_3d_pose
from agents.helper import * #get_keypoints_indexes, get_id_index, select_bone_ratios, all_bone_ratio_combo
from agents.rbo_transform_np import is_unit_vector, save_bone_prior_properties, save_jmc_priors, quaternion_rotate
from agents.rbo_transform_tc import log_likelihood_loss_func, torch_t, guard_div, to_numpy, FreeBoneOrientation

from agents.helper import processor
# processor = 'cpu'#"cpu"
# if torch.cuda.is_available():
#     processor = 'cuda'#"cuda:0"


def pose_reg_properties_extractor(train_generator, subjects_train, args, supr_subset_tag, frt_tag):
    log_jmc_meta = True
    log_bpc_meta = False
    log_bse_meta = False
    log_blen_meta = False
    visualize_pose = False
    is_torch_op = args.gen_pose_priors >= -2
    force_varying_bone_lengths = len(subjects_train)<=3
    bone_metadata = {'bone_lengths':[], 'bone_symms':[], 'bone_ratios':[]}
    fb_orientation_vecs = dict()
    ordered_joint_names = list(JMC_RMTX_JOINT_CONFIG.keys())

    for joint_name in ordered_joint_names:
        fb_orientation_vecs[joint_name] = [] # was joint_id

    if visualize_pose:
        n_rows, n_cols = 3, 6 # 2, 7
        SP_FIG, SP_AXS = plt.subplots(n_rows, n_cols, subplot_kw={'projection':'3d'}, figsize=(3*n_cols, 4*n_rows))
        SP_FIG.subplots_adjust(left=0.0, right=1.0, wspace=-0.0)
        SP_FIG2, SP_AXS2 = plt.subplots(n_rows, n_cols, subplot_kw={'projection':'3d'}, figsize=(3*n_cols, 4*n_rows))
        SP_FIG2.subplots_adjust(left=0.0, right=1.0, wspace=-0.0)
        FIGURE_JOINT_ORDER = {
           'Pelvis':(0,0), 'Spine':(0,1), 'Neck':(0,2), 'Head':(0,3), 'RWaist':(0,4), 'LWaist':(0,5),
           'RScapula':(1,0), 'LScapula':(1,1), 'RKnee':(1,2), 'LKnee':(1,3), 'RHip':(1,4), 'LHip':(1,5),
           'RShoulder':(2,0), 'LShoulder':(2,1), 'RElbow':(2,2), 'LElbow':(2,3), 'Camera-Pose':(2,4), 'Upright-Pose':(2,5),
        }

    # Extract and organize Pose Priors configuration parameters
    n_fbjnts = len(ordered_joint_names) # -->fbj
    quad_kpt_idxs_rmtx, xy_yx_axis_dirs, xy_yx_idxs, z_axis_ab_idxs, yx_axis_ab_idxs = \
        extract_rfboa_rotamatrix_config(JMC_RMTX_JOINT_CONFIG, ordered_joint_names, args.group_jmc, for_torch=is_torch_op)
    rot_matrix_fb = FreeBoneOrientation(args.batch_size, xy_yx_axis_dirs, z_axis_ab_idxs,
                                        yx_axis_ab_idxs, xy_yx_idxs, n_fb_joints=n_fbjnts, ret_mode=-1)
    quad_kpt_idxs_quat, axis1_quadrant, axis2_quadrant, plane_proj_multiplier, hflip_multiplier = \
        extract_rfboa_quaternion_config(JMC_QUAT_JOINT_CONFIG, ordered_joint_names, args.group_jmc, for_torch=is_torch_op)
    #***assert (quad_kpt_idxs_rmtx==quad_kpt_idxs_quat)
    quaternion_fb = FreeBoneOrientation(args.batch_size, quad_uvec_axes1=axis1_quadrant, quad_uvec_axes2=axis2_quadrant,
                                        plane_proj_mult=plane_proj_multiplier, hflip_multiplier=hflip_multiplier,
                                        n_fb_joints=n_fbjnts, validate_ops=False, rot_tfm_mode=1, ret_mode=-1)
    JMC_JOINT_CONFIG = eval('JMC_{}_JOINT_CONFIG'.format(args.rbo_ops_type.upper()))
    if args.n_bone_ratios==15:
        BPE_ORDERED_TAGS, BONE_RATIO_PAIR_IDXS_A, BONE_RATIO_PAIR_IDXS_B = select_bone_ratios(VIDEOPOSE3D_BONE_ID_2_IDX)
    else:
        BPE_ORDERED_TAGS, BONE_RATIO_PAIR_IDXS_A, BONE_RATIO_PAIR_IDXS_B = all_bone_ratio_combo(VIDEOPOSE3D_BONE_ID_2_IDX)

    jnt_cnt = 0
    batch_cnt = 1
    n_bones = 16
    n_ratios = len(BPE_ORDERED_TAGS) # -->brp
    jnts_meta_max = np.zeros((n_fbjnts,3), dtype=np.float32) #8
    jnts_meta_min = np.full((n_fbjnts,3), dtype=np.float32, fill_value=np.inf) #8
    #homog_pose_tsr = np.ones((args.batch_size,1,n_fbjnts,17,4), dtype=np.float32)
    pose_wrt_cam_frm_hom = np.ones((17,4), dtype=np.float32) # holds keypoint position in homogenous coordinates
    hflip_multiplier = to_numpy(hflip_multiplier)
    print('[INFO] Generating Pose Prior properties from supervised training subset..')
    if args.gen_pose_priors<=-2: os.makedirs(os.path.join('priors', supr_subset_tag, 'properties'), exist_ok=True)

    # Iterate over 3D-pose ground-truths and collect bone-length and joint orientation priors
    #for batch_3d in [TOY_3D_POSE]:
    for _, batch_3d, _ in train_generator.next_epoch():
        n_supr_samples, n_supr_frames = batch_3d.shape[:2]
        inputs_3d = batch_3d.astype(np.float32) # (?,f,17,3)
        # Pose post processing **IMPORTANT**
        inputs_3d[:,:,0] = 0  # necessary to orient pose at pelvis irrespective of trajectory (M-Hip)
        if is_torch_op: inputs_3d = torch.from_numpy(inputs_3d.astype('float32')).to(processor)

        if log_blen_meta or log_bse_meta or log_bpc_meta:
            dists_m = inputs_3d[:, :, BONE_CHILD_KPTS_IDXS] - inputs_3d[:, :, BONE_PARENT_KPTS_IDXS] # -->(?,f,16,3)
            all_bone_lengths_m = torch.linalg.norm(dists_m, dim=3)  # (?,f,16,3) --> (?,f,16)
            assert (torch.all(torch.ge(all_bone_lengths_m, 0.))), "shouldn't be negative {}".format(all_bone_lengths_m)

            # collect bse and bone length metadata
            if log_blen_meta: bone_metadata['bone_lengths'].append(to_numpy(all_bone_lengths_m))
            if log_bse_meta:
                bone_sym_diffs = torch.abs(all_bone_lengths_m[:,:,RGT_SYM_BONE_INDEXES] -
                                           all_bone_lengths_m[:,:,LFT_SYM_BONE_INDEXES])
                bone_metadata['bone_symms'].append(to_numpy(bone_sym_diffs))

            # collect bpc metadata
            if log_bpc_meta:
                assert (torch.all(torch.ge(all_bone_lengths_m, 0.))), "shouldn't be negative {}".format(all_bone_lengths_m)
                if force_varying_bone_lengths:
                    gaussian_factors = []
                    if is_torch_op:
                        for i in range(n_bones):
                            gaussian_factors.append(torch.normal(mean=1, std=0.01, size=(n_supr_samples, n_supr_frames)))
                        bone_gaussian_factors = torch.stack(gaussian_factors, dim=-1).to(processor) # (?,f,16)
                        all_bone_lengths_m *= bone_gaussian_factors
                    else:
                        for i in range(n_bones):
                            gaussian_factors.append(np.random.normal(1, 0.01, size=(n_supr_samples, n_supr_frames)))
                        bone_gaussian_factors = np.stack(gaussian_factors, axis=-1) # (?,f,16)
                        all_bone_lengths_m *= bone_gaussian_factors
                ##*else: assert (torch.all((all_bone_lengths_mm - (all_bone_lengths_m*1000))<=0.1))
                bone_jnt_ratios = guard_div(all_bone_lengths_m[:,:,BONE_RATIO_PAIR_IDXS_A],
                                            all_bone_lengths_m[:,:,BONE_RATIO_PAIR_IDXS_B]) # (?,f,15)
                bone_metadata['bone_ratios'].append(to_numpy(bone_jnt_ratios))

        # collect jmc metadata
        if log_jmc_meta:
            # necessary to convert coordinates from meters to millimeters
            ##if is_torch_op: inputs_3d_mm = inputs_3d * torch_t(1000)
            ##else: inputs_3d_mm = inputs_3d * 1000
            ##quadruplet_kpts = inputs_3d_mm[:,:,quad_kpt_idxs,:] # (?,f,17,3)->(?,f,fbj,4,3)
            quadruplet_kpts = inputs_3d[:,:,quad_kpt_idxs_rmtx,:] # (?,f,17,3)->(?,f,fbj,4,3)
            rmtx_pivot_kpts, rotation_mtxs, rot_transform_mtxs, rmtx_free_bone_uvecs = \
                rot_matrix_fb(quadruplet_kpts) # (?,f,j,4,3)->((?,f,j,1,3), (?,f,j,3,3), (?,f,j,4,4), (?,f,j,3))
            quat_pivot_kpts, quaternion_vec1, quaternion_vec2, quat_free_bone_uvecs = \
                quaternion_fb(quadruplet_kpts) # (?,f,j,4,3)->((?,f,j,1,3), (?,f,j,3,3), (?,f,j,4,4), (?,f,j,3))

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
            #***assert (all_are_similar_fbs)
            free_bone_uvecs = rmtx_free_bone_uvecs if args.rbo_ops_type=='rmtx' else quat_free_bone_uvecs

            jnt_cnt += n_supr_samples*n_supr_frames*n_fbjnts
            for j_idx, joint_name in enumerate(ordered_joint_names):
                jmc_metadata = free_bone_uvecs[:,0,j_idx,:]
                fb_orientation_vecs[joint_name].append(jmc_metadata)
                # log max and min
                jmc_metadata_min = np.min(jmc_metadata, axis=0)
                jmc_metadata_max = np.max(jmc_metadata, axis=0)
                jnts_meta_min[j_idx] = np.where(jmc_metadata_min<jnts_meta_min[j_idx], jmc_metadata_min, jnts_meta_min[j_idx])
                jnts_meta_max[j_idx] = np.where(jmc_metadata_max>jnts_meta_max[j_idx], jmc_metadata_max, jnts_meta_max[j_idx])

            if visualize_pose:
                smp_idx = 0
                rmtx_pvt_kpts = rmtx_pivot_kpts[smp_idx, 0, :, 0]
                rot_mtxs = rotation_mtxs[smp_idx, 0]
                tfm_mtxs = rot_transform_mtxs[smp_idx, 0]
                quat_pvt_kpts = quat_pivot_kpts[smp_idx, 0, :, 0]
                quat_v1 = quaternion_vec1[smp_idx, 0]
                quat_v2 = quaternion_vec2[smp_idx, 0]
                #pose_wrt_cam_frm = to_numpy(inputs_3d_mm[smp_idx,0]) if is_torch_op else inputs_3d_mm[smp_idx,0]
                pose_wrt_cam_frm = to_numpy(inputs_3d[smp_idx,0]) if is_torch_op else inputs_3d[smp_idx,0]
                #pose_wrt_cam_frm_hom[:,:3] = pose_wrt_cam_frm
                # todo: instead of flipping, rotate 180" to reverse the effect of inverted camera projection?
                upright_pose = pose_wrt_cam_frm * [-1,-1,1] # to reverse effect of inverted camera projection on x & y comp.
                # todo: relative bone orientation alignment should be executed on upright_pose after reversing camera inversion
                for j_idx, joint_name in enumerate(ordered_joint_names):
                    jnt_quad_kpts_idxs = get_keypoints_indexes(JMC_RMTX_JOINT_CONFIG[joint_name][0])
                    row_idx, col_idx = FIGURE_JOINT_ORDER[joint_name]
                    #pose_wrt_pvt_frm_hom = np.dot(tfm_mtx[j_idx], pose_wrt_cam_frm_hom.T).T
                    # translate then rotate
                    pose_wrt_cam_frm_hom[:,:3] = pose_wrt_cam_frm - rmtx_pvt_kpts[[j_idx]] # translate to pivot
                    #pose_wrt_pvt_frm_hom = np.dot(tfm_mtxs[j_idx], pose_wrt_cam_frm_hom.T).T # rotate about pivot
                    pose_wrt_pvt_frm_hom = np.matmul(tfm_mtxs[j_idx], pose_wrt_cam_frm_hom.T).T # rotate about pivot
                    pose_wrt_pvt_frm = pose_wrt_pvt_frm_hom[:,:3] / pose_wrt_pvt_frm_hom[:,[3]]
                    plot_3d_pose(pose_wrt_pvt_frm, SP_FIG, SP_AXS[row_idx,col_idx], joint_name, KPT_2_IDX, jnt_quad_kpts_idxs)
                    quat_pose = quaternion_rotate(pose_wrt_cam_frm, quat_v1[j_idx], quat_v2[j_idx], quat_pvt_kpts[[j_idx]])
                    if args.group_jmc: quat_pose *= hflip_multiplier[0,0,j_idx]
                    plot_3d_pose(quat_pose, SP_FIG2, SP_AXS2[row_idx,col_idx], joint_name, KPT_2_IDX, jnt_quad_kpts_idxs)
                plot_3d_pose(pose_wrt_cam_frm, SP_FIG2, SP_AXS2[2,4], 'Camera-Pose', KPT_2_IDX, [-1]*4)
                plot_3d_pose(upright_pose, SP_FIG2, SP_AXS2[2,5], 'Upright-Pose', KPT_2_IDX, [-1]*4, t_tag='Quaternion ')
                plot_3d_pose(pose_wrt_cam_frm, SP_FIG, SP_AXS[2,4], 'Camera-Pose', KPT_2_IDX, [-1]*4)
                plot_3d_pose(upright_pose, SP_FIG, SP_AXS[2,5], 'Upright-Pose', KPT_2_IDX, [-1]*4, t_tag='Rot-Matrix ',
                             display=True)
                sys.exit(0)

        batch_cnt += 1
        if args.gen_pose_priors<=-2 and batch_cnt%100==0:
            print('{:>17,} joints from {:>9,} poses or {:5,} batches passed..'.format(jnt_cnt, jnt_cnt//n_fbjnts, batch_cnt))
        if n_supr_samples!=args.batch_size:
            print('[INFO] batch:{} with size {} < {} standard batch-size'.format(batch_cnt, n_supr_samples, args.batch_size))

    # save onetime logged pose priors
    print('{:>17,} joints from {:>9,} poses or {:5,} batches passed..'.format(jnt_cnt, jnt_cnt//n_fbjnts, batch_cnt))
    if args.gen_pose_priors<=-2 and (log_blen_meta or log_bse_meta or log_bpc_meta):
        save_bone_prior_properties(bone_metadata, BPE_ORDERED_TAGS, args.data_augmentation,
                                   supr_subset_tag, frt_tag, n_ratios, from_torch=is_torch_op)

    if args.gen_pose_priors<=-2 and log_jmc_meta:
        save_jmc_priors(fb_orientation_vecs, ordered_joint_names, n_fbjnts, args.data_augmentation, args.group_jmc,
                        supr_subset_tag, frt_tag, args.rbo_ops_type, JMC_JOINT_CONFIG, from_torch=is_torch_op)

    print("Done.\n\tjnt_meta_min:{}\n\tjnt_meta_max:{}".format(jnts_meta_min, jnts_meta_max))


def inlier_subset(data_pts, pts_distance):
    q1, q3 = np.quantile(pts_distance, [0.25, 0.75])  # compute Q1 & Q3
    iqr = q3 - q1  # compute inter-quartile-range
    outlier_error_thresh = q3 + 1.5*iqr
    is_inlier = pts_distance <= outlier_error_thresh
    n_inliers = np.sum(is_inlier.astype(np.int32))
    inlier_indexes = np.squeeze(np.argwhere(is_inlier))
    cluster_inlier_pts = data_pts[inlier_indexes]
    return cluster_inlier_pts, n_inliers/len(data_pts), inlier_indexes


def extract_bpc_config_params(bone_priors, net_bone_id_2_idx):
    bone_prop_indexes_A = []
    bone_prop_indexes_B = []

    for bone_ratio_id in bone_priors['ratio_order']:
        boneA_id, boneB_id = bone_ratio_id.split('/')
        boneA_idx = net_bone_id_2_idx[boneA_id]
        boneB_idx = net_bone_id_2_idx[boneB_id]
        bone_prop_indexes_A.append(boneA_idx)
        bone_prop_indexes_B.append(boneB_idx)

    return bone_prop_indexes_A, bone_prop_indexes_B


def configure_bpc_likelihood(bpc_priors, log_func_modes, log_lihood_eps):
    ordered_tags = bpc_priors['ratio_order']
    loglihood_metadata = bpc_priors['br_logli_metadata']
    bone_ratio_mean_variance = bpc_priors['br_mean_variance']
    sigma_variance, exponent_coef, mu_mean, logli_min, logli_max, likeli_max, likeli_argmax, \
    logli_spread, logli_mean, logli_std, logli_wgt = [], [], [], [], [], [], [], [], [], [], []

    for bone_ratio_id in ordered_tags:
        (mu, sigma) = bone_ratio_mean_variance[bone_ratio_id]
        mu_mean.append(mu)
        sigma_variance.append(sigma)
        exp_coef = (2*np.pi*sigma)**(-1/2)
        exponent_coef.append(exp_coef)
        likeli_max.append(loglihood_metadata[bone_ratio_id][0])
        likeli_argmax.append(loglihood_metadata[bone_ratio_id][1])
        logli_spread.append(loglihood_metadata[bone_ratio_id][2])
        logli_mean.append(loglihood_metadata[bone_ratio_id][3])
        logli_std.append(loglihood_metadata[bone_ratio_id][4])
        logli_min.append(loglihood_metadata[bone_ratio_id][5])
        logli_max.append(loglihood_metadata[bone_ratio_id][6])
        logli_wgt.append(loglihood_metadata[bone_ratio_id][7])

    n_ratios = len(mu_mean)
    mu_mean = np.array(mu_mean, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    #mu_mean = np.around(mu_mean, 6)
    sigma_variance = np.array(sigma_variance, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    #sigma_variance = np.around(sigma_variance, 7)
    exponent_coef = np.array(exponent_coef, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    likeli_max = np.array(likeli_max, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    likeli_argmax = np.array(likeli_argmax, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    logli_spread = np.array(logli_spread, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    logli_mean = np.array(logli_mean, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    logli_std = np.array(logli_std, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    logli_min = np.array(logli_min, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    logli_max = np.array(logli_max, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
    logli_span = logli_max - logli_min # (1,1,r)
    logli_wgt = np.array(logli_wgt, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)

    per_comp_wgts = softmax(-logli_std)
    #per_comp_wgts_dict = dict(zip(ordered_tags, per_comp_wgts[0,0,:]))
    #print('per_comp_wgts_dict:\n{}'.format(per_comp_wgts_dict))
    per_comp_wgts = torch.from_numpy(per_comp_wgts).to(processor)

    mu_mean = torch.from_numpy(mu_mean).to(processor)
    sigma_variance = torch.from_numpy(sigma_variance).to(processor)
    exponent_coef = torch.from_numpy(exponent_coef).to(processor)
    likeli_max = torch.from_numpy(likeli_max).to(processor)
    likeli_argmax = torch.from_numpy(likeli_argmax).to(processor)
    logli_spread = torch.from_numpy(logli_spread).to(processor)
    logli_mean = torch.from_numpy(logli_mean).to(processor)
    logli_std = torch.from_numpy(logli_std).to(processor)
    logli_min = torch.from_numpy(logli_min).to(processor)
    logli_span = torch.from_numpy(logli_span).to(processor)
    logli_wgt = torch.from_numpy(logli_wgt).to(processor)
    log_of_spread = torch.log(logli_spread)
    move_up_const = \
        log_likelihood_loss_func(likeli_max, log_func_modes, logli_mean, logli_std, logli_min,
                                 logli_span, logli_wgt, logli_spread, log_of_spread, log_lihood_eps,
                                 move_up_const=torch_t(0), ret_move_const=True)

    k_rank_rcoef = np.int(np.ceil(1/3))
    k_rank_wgt = np.around(np.pi**-(2*k_rank_rcoef), k_rank_rcoef)
    #print('BPC - rank:{} k_rank:{} k_rank_rcoef:{} k_rank_wgt:{}'.format(0, 1, k_rank_rcoef, k_rank_wgt))
    loglihood_wgts = np.full((1,1,n_ratios), fill_value=k_rank_wgt, dtype=np.float32)
    loglihood_wgts = torch.from_numpy(loglihood_wgts).to(processor)

    return sigma_variance, exponent_coef, mu_mean, likeli_max, likeli_argmax, logli_spread, logli_mean, \
           logli_std, logli_min, logli_span, logli_wgt, log_of_spread, move_up_const, per_comp_wgts, loglihood_wgts


def extract_rfboa_quaternion_config(joint_cfgs, ordered_joint_names, group_jnts=True, for_torch=True):
    quad_kpt_idxs = []
    axis1_quadrant_uvecs = []
    axis2_quadrant_uvecs = []
    #xy_idxs = [] # only for ops validation
    hflip_multiplier =[]
    hflip_fbjnts = ('LHip', 'LShoulder', 'LKnee', 'LElbow', 'LWaist', 'LScapula')

    for joint_name in ordered_joint_names:
        (quad_kpts, quadrant_uvecs_pair) = joint_cfgs[joint_name]
        quad_kpt_idxs.append(get_keypoints_indexes(quad_kpts))
        axis1_quadrant_uvecs.append(quadrant_uvecs_pair[0])
        axis2_quadrant_uvecs.append(quadrant_uvecs_pair[1])
        #xy_idxs.append(abs(quadrant_uvecs_pair[0][1])) # lazy trick to get idx of first axis
        if group_jnts:
            hflip = [-1,1,1] if joint_name in hflip_fbjnts else [1,1,1]
            hflip_multiplier.append(hflip) # flip on x-axis

    n_joints = len(quad_kpt_idxs)
    axis1_quadrant_uvecs = np.array(axis1_quadrant_uvecs, dtype=np.float32).reshape((1,1,n_joints,3))
    axis2_quadrant_uvecs = np.array(axis2_quadrant_uvecs, dtype=np.float32).reshape((1,1,n_joints,3))
    plane_proj_multiplier = np.mod(np.abs(axis1_quadrant_uvecs)+1, 2) # changes [-1/1,0,0]->[0,1,1] [0,-1/1,0]->[1,0,1]
    assert (np.all(plane_proj_multiplier+np.abs(axis1_quadrant_uvecs)==np.ones((3,))))
    if group_jnts:
        hlip_multiplier = np.array(hflip_multiplier, dtype=np.float32).reshape((1,1,n_joints,1,3))
    else: hlip_multiplier = None

    if for_torch:
        axis1_quadrant_uvecs = torch.from_numpy(axis1_quadrant_uvecs).to(processor)
        axis2_quadrant_uvecs = torch.from_numpy(axis2_quadrant_uvecs).to(processor)
        plane_proj_multiplier = torch.from_numpy(plane_proj_multiplier).to(processor)
        if group_jnts: hlip_multiplier = torch.from_numpy(hlip_multiplier).to(processor)

    return quad_kpt_idxs, axis1_quadrant_uvecs, axis2_quadrant_uvecs, plane_proj_multiplier, hlip_multiplier#, xy_idxs


def extract_rfboa_rotamatrix_config(joint_cfgs, ordered_joint_names, group_jnts=True, rev_direction=-1, for_torch=True):
    quad_kpt_idxs = []
    xy_idxs, yx_idxs = [], []
    xy_axis_dirs, yx_axis_dirs = [], []
    z_axis_a_idxs, z_axis_b_idxs = [], []
    yx_axis_a_idxs, yx_axis_b_idxs = [], []
    change_direction = [('LHip', 'LShoulder'), ('LKnee', 'LElbow', 'LWaist', 'LScapula')]

    for joint_name in ordered_joint_names:
        (quad_kpts, xy_yx_dir, xy_idx, yx_idx, z_axb_indexes, yx_axb_indexes) = joint_cfgs[joint_name][:6]
        quad_kpt_idxs.append(get_keypoints_indexes(quad_kpts))
        if group_jnts and joint_name in change_direction[0]:
            xy_axis_dirs.append(xy_yx_dir[0]*rev_direction)
        else: xy_axis_dirs.append(xy_yx_dir[0])
        if group_jnts and joint_name in change_direction[1]:
            yx_axis_dirs.append(xy_yx_dir[1]*rev_direction)
        else: yx_axis_dirs.append(xy_yx_dir[1])
        xy_idxs.append(xy_idx)
        yx_idxs.append(yx_idx)
        z_axis_a_idxs.append(z_axb_indexes[0])
        z_axis_b_idxs.append(z_axb_indexes[1])
        yx_axis_a_idxs.append(yx_axb_indexes[0])
        yx_axis_b_idxs.append(yx_axb_indexes[1])

    n_joints = len(quad_kpt_idxs)
    xy_axis_dirs = np.array(xy_axis_dirs, dtype=np.float32).reshape((1,1,n_joints,1))
    yx_axis_dirs = np.array(yx_axis_dirs, dtype=np.float32).reshape((1,1,n_joints,1))

    if for_torch:
        xy_axis_dirs = torch.from_numpy(xy_axis_dirs).to(processor)
        yx_axis_dirs = torch.from_numpy(yx_axis_dirs).to(processor)

    return quad_kpt_idxs, (xy_axis_dirs, yx_axis_dirs), (xy_idxs, yx_idxs), \
           (z_axis_a_idxs, z_axis_b_idxs), (yx_axis_a_idxs, yx_axis_b_idxs)


def configure_jmc_likelihood(jmc_priors, log_func_modes, log_lihood_eps, jmc_ranks=1):
    ordered_joint_names = jmc_priors['joint_order']
    sets_of_joints_combo = jmc_priors['jnt_rank_sets']
    uvec_means = jmc_priors['fb_axes_means']
    covariance_matrices = jmc_priors['fb_covariance']
    inv_covariance_matrices = jmc_priors.get('fb_inv_covariance', None)
    loglihood_metadata = jmc_priors['fb_logli_metadata']
    jmc_rank_priors = []
    loglihood_wgts = []
    cum_logli_std = []
    jm_orient_ids = []

    for r_idx, rank_ordered_jnt_tags in enumerate(sets_of_joints_combo[:jmc_ranks]):
        uvec_mean_kx1, covariance_kxk, inv_covariance_kxk, k_logli_min, k_logli_max, k_likeli_max, k_likeli_argmax, \
        k_logli_spread, k_logli_mean, k_logli_std, k_logli_wgt = [], [], [], [], [], [], [], [], [], [], []
        rank, k_rank = loglihood_metadata[r_idx]['rank_const']
        k_rank_rcoef = np.int(np.ceil(k_rank/3))
        k_rank_wgt = np.around(np.pi**-(2*k_rank_rcoef), k_rank_rcoef)
        #print('JMC - rank:{} k_rank:{} k_rank_rcoef:{} k_rank_wgt:{}'.format(rank, k_rank, k_rank_rcoef, k_rank_wgt))
        loglihood_wgts.append(np.full((1,1,len(rank_ordered_jnt_tags)), fill_value=k_rank_wgt, dtype=np.float32))
        k_jnts_indexes = []
        for i in range(rank): k_jnts_indexes.append([])

        for (orient_id, orient_jnts) in rank_ordered_jnt_tags.items():
            jm_orient_ids.append(orient_id)
            uvec_mu = uvec_means[r_idx][orient_id]
            #uvec_mu = np.around(uvec_mu, 7)
            covariance_mtx = covariance_matrices[r_idx][orient_id]
            #covariance_mtx = np.around(covariance_mtx, 7)
            if inv_covariance_matrices is not None:
                inv_covariance_mtx = inv_covariance_matrices[r_idx][orient_id]
                #inv_covariance_mtx = np.around(inv_covariance_mtx, 7)
            else: inv_covariance_mtx = np.linalg.inv(covariance_mtx)

            #print('{} covariance matrix:\n{}'.format(orient_id, covariance_mtx))
            cov_dot_inv = covariance_mtx.dot(inv_covariance_mtx)
            all_are_identity = np.all(np.isclose(cov_dot_inv, np.eye(k_rank), atol=1e-02))
            assert(all_are_identity), '{} inverse matrix-{} test: inv-covariance:\n{}'.format(orient_id, k_rank, cov_dot_inv)

            for pair_idx, jnt_id in enumerate(orient_jnts):
                k_jnts_indexes[pair_idx].append(get_id_index(ordered_joint_names, jnt_id))

            uvec_mean_kx1.append(uvec_mu)
            covariance_kxk.append(covariance_mtx)
            inv_covariance_kxk.append(inv_covariance_mtx)
            k_likeli_max.append(loglihood_metadata[r_idx][orient_id][0])
            k_likeli_argmax.append(loglihood_metadata[r_idx][orient_id][1])
            k_logli_spread.append(loglihood_metadata[r_idx][orient_id][2])
            k_logli_mean.append(loglihood_metadata[r_idx][orient_id][3])
            k_logli_std.append(loglihood_metadata[r_idx][orient_id][4])
            k_logli_min.append(loglihood_metadata[r_idx][orient_id][5])
            k_logli_max.append(loglihood_metadata[r_idx][orient_id][6])
            k_logli_wgt.append(loglihood_metadata[r_idx][orient_id][7])

        cum_logli_std.extend(k_logli_std)
        n_components = len(k_jnts_indexes[0])
        uvec_mean_kx1 = np.array(uvec_mean_kx1, dtype=np.float32).reshape((1,1,n_components,k_rank,1))
        covariance_kxk = np.array(covariance_kxk, dtype=np.float32).reshape((1,1,n_components,k_rank,k_rank))
        inv_covariance_kxk = np.array(inv_covariance_kxk, dtype=np.float32).reshape((1,1,n_components,k_rank,k_rank))
        #inv_covariance_kxk = np.linalg.inv(covariance_kxk) # (1,1,j,k,k)
        k_exponent_coefs = mvpdf_exponent_coefficient(covariance_kxk, k=k_rank) # (1,1,j,1,1)
        k_likeli_max = np.array(k_likeli_max, dtype=np.float32).reshape((1,1,n_components))
        #k_likeli_argmax = np.array(k_likeli_argmax, dtype=np.float32).reshape((1,1,n_components,k_rank))
        k_logli_spread = np.array(k_logli_spread, dtype=np.float32).reshape((1,1,n_components))
        k_logli_mean = np.array(k_logli_mean, dtype=np.float32).reshape((1,1,n_components))
        k_logli_std = np.array(k_logli_std, dtype=np.float32).reshape((1,1,n_components))
        k_logli_min = np.array(k_logli_min, dtype=np.float32).reshape((1,1,n_components))
        k_logli_max = np.array(k_logli_max, dtype=np.float32).reshape((1,1,n_components))
        k_logli_span = k_logli_max - k_logli_min # (1,1,j)
        k_logli_wgt = np.array(k_logli_wgt, dtype=np.float32).reshape((1,1,n_components))

        uvec_mean_kx1 = torch.from_numpy(uvec_mean_kx1).to(processor)
        inv_covariance_kxk = torch.from_numpy(inv_covariance_kxk).to(processor)
        k_exponent_coefs = torch.from_numpy(k_exponent_coefs).to(processor)
        k_likeli_max = torch.from_numpy(k_likeli_max).to(processor)
        #k_likeli_argmax = torch.from_numpy(k_likeli_argmax).to(processor)
        k_logli_spread = torch.from_numpy(k_logli_spread).to(processor)
        k_logli_mean = torch.from_numpy(k_logli_mean).to(processor)
        k_logli_std = torch.from_numpy(k_logli_std).to(processor)
        k_logli_min = torch.from_numpy(k_logli_min).to(processor)
        k_logli_span = torch.from_numpy(k_logli_span).to(processor)
        k_logli_wgt = torch.from_numpy(k_logli_wgt).to(processor)
        k_log_of_spread = torch.log(k_logli_spread)
        k_move_up_const = \
            log_likelihood_loss_func(k_likeli_max, log_func_modes, k_logli_mean, k_logli_std, k_logli_min,
                                     k_logli_span, k_logli_wgt, k_logli_spread, k_log_of_spread, log_lihood_eps,
                                     move_up_const=torch_t(0), ret_move_const=True)

        # excluding: k_likeli_max and k_likeli_argmax because not needed atm
        jmc_rank_priors.append((k_jnts_indexes, uvec_mean_kx1, inv_covariance_kxk, k_exponent_coefs,
                                k_logli_spread, k_logli_mean, k_logli_std, k_logli_min, k_logli_span,
                                k_logli_wgt, k_log_of_spread, k_move_up_const))

    loglihood_wgts = np.concatenate(loglihood_wgts, axis=-1)
    loglihood_wgts = torch.from_numpy(loglihood_wgts).to(processor)
    cum_logli_std = np.array(cum_logli_std, dtype=np.float32).reshape((1,1,-1))
    per_comp_wgts = softmax(-cum_logli_std)
    #per_comp_wgts_dict = dict(zip(jm_orient_ids, per_comp_wgts[0,0,:]))
    #print('per_comp_wgts_dict:\n{}'.format(per_comp_wgts_dict))
    per_comp_wgts = torch.from_numpy(per_comp_wgts).to(processor)

    return jmc_rank_priors, loglihood_wgts, per_comp_wgts


# def configure_jmc_uvecdist(cluster_meta, for_torch=True):
#     centroids_jxkx3, radius_jxkx1 = [], []
#     for joint_name in cluster_meta.keys():
#         centroids_jxkx3.append(cluster_meta[joint_name][0])
#         radius_jxkx1.append(cluster_meta[joint_name][1])
#
#     n_joints = len(centroids_jxkx3)
#     centroids_jxkx3 = np.array(centroids_jxkx3, dtype=np.float32).reshape((1,1,n_joints,8,3))
#     radius_jxkx1 = np.array(radius_jxkx1, dtype=np.float32).reshape((1,1,n_joints,8,1))
#
#     if for_torch:
#         centroids_jxkx3 = torch.from_numpy(centroids_jxkx3).to(processor)
#         radius_jxkx1 = torch.from_numpy(radius_jxkx1).to(processor)
#
#     return centroids_jxkx3, radius_jxkx1
#
#
# def centroid_clustering(data_tag, data_points, k=8, p=250):
#     # Cluster 3d-points using KMeans centroid-type clustering
#     #k = int(np.ceil(len(data_points)/p))
#     model = KMeans(n_clusters=k, random_state=len(data_points))
#     #model = AgglomerativeClustering(n_clusters=k, linkage='single')
#     model.fit_predict(data_points)
#     #visualize_clusters(data_tag, data_points, model.labels_)
#     return model
#
#
# def cluster_centroids_range(model, uvec_data_pts, d_nms, k_clusters):
#     centroids, radius = [], []
#     cluster_labels = np.full(len(uvec_data_pts), fill_value=k_clusters+2, dtype=np.int32)
#     for i in range(len(model.cluster_centers_)):
#         cluster_ctr = model.cluster_centers_[i].astype(np.float32)
#         cluster_ctr_uvec = cluster_ctr / np.linalg.norm(cluster_ctr)
#         cluster_ctr_uvec = cluster_ctr_uvec.astype(np.float32)
#         cluster_indexes = np.argwhere(model.labels_==i)
#         if len(cluster_indexes)==1:
#             # one time for face
#             centroids.append(centroids[-1])
#             radius.append(radius[-1])
#             continue
#         else: cluster_indexes = np.squeeze(cluster_indexes)
#         cluster_data_pts = uvec_data_pts[cluster_indexes]
#         dist_2_ctr_uvec = np.linalg.norm(cluster_data_pts - cluster_ctr_uvec, axis=1)
#
#         cluster_avg = np.squeeze(np.mean(cluster_data_pts, axis=0))
#         is_ctr_avg = abs(np.linalg.norm(cluster_ctr - cluster_avg)) < 1e-4
#         dist_2_avg_uvec = np.linalg.norm(cluster_data_pts - cluster_ctr_uvec, axis=1)
#         cluster_inlier_pts, inlier_percentage, inlier_indexes = inlier_subset(cluster_data_pts, dist_2_avg_uvec)
#
#         inlier_cluster_indexes = cluster_indexes[inlier_indexes]
#         cluster_labels[inlier_cluster_indexes] = i
#
#         inliers_avg = np.mean(cluster_inlier_pts, axis=0)
#         inliers_avg_uvec = inliers_avg / np.linalg.norm(inliers_avg)
#         dist_2_inlier_avg_uvec = np.linalg.norm(cluster_inlier_pts - inliers_avg_uvec, axis=1)
#         cluster_radius = np.max(dist_2_inlier_avg_uvec) + d_nms
#
#         centroids.append(inliers_avg_uvec)
#         radius.append(cluster_radius)
#         print('  {}. ctr:{} {} rad:{:.4f} - ctr:{} {} rad:{:.4f} - {} {:.2f}'.
#               format(i+1, np.around(cluster_ctr_uvec, 7), is_unit_vector(cluster_ctr_uvec), np.max(dist_2_ctr_uvec),
#                      np.around(inliers_avg_uvec, 7), is_unit_vector(inliers_avg_uvec), cluster_radius,
#                      is_ctr_avg, inlier_percentage))
#     return np.float32(centroids), np.float32(radius), cluster_labels
#
#
# if __name__ == '__main__':
#     # group_joints = True
#     # joint_names = ['RShoulder', 'LShoulder', 'RHip', 'LHip', 'Pelvis',
#     #                'Spine', 'Neck', 'Face', 'RElbow', 'LElbow', 'RKnee', 'LKnee']
#     # joint_groups = ['Shoulder', 'Hip', 'Pelvis', 'Spine', 'Neck', 'Face', 'Elbow', 'Knee']
#     # grp_2_jts = {'Shoulder':['RShoulder','LShoulder'], 'Hip':['RHip','LHip'], 'Pelvis':['Pelvis'], 'Spine':['Spine'],
#     #              'Neck':['Neck'], 'Face':['Face'], 'Elbow':['RElbow','LElbow'], 'Knee':['RKnee','LKnee']}
#     # joint_ids = joint_groups if group_joints else joint_names
#     # k_clusters = 8
#     # d_nms = 0.025
#     # n_rows = 2
#     # n_cols = 4 if group_joints else 6
#     # title = 'Joint Orientation (Mobility Sphere) Priors'
#     # SP_FIG, SP_AXS = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={'projection':'3d'},
#     #                               figsize=(4*n_cols, 4*n_rows))
#     # cluster_meta = dict()
#     # for fig_idx, joint_tag in enumerate(joint_ids):
#     #     print(' clustering {} data-points..'.format(joint_tag))
#     #     file_path = os.path.join('../priors', '{}_jm_nms_uvecs_{}.npy'.format(joint_tag, d_nms))
#     #     free_limb_uvecs = np.load(file_path) # (n, 2, 3)
#     #
#     #     model = centroid_clustering(joint_tag, free_limb_uvecs, k=k_clusters)
#     #     centroids, radius, pnt_labels = cluster_centroids_range(model, free_limb_uvecs, d_nms, k_clusters)
#     #
#     #     for joint_name in grp_2_jts[joint_tag]:
#     #         cluster_meta[joint_name] = (centroids, radius)
#     #     row_idx, col_idx = fig_idx//n_cols, fig_idx%n_cols
#     #     visualize_clusters(joint_tag, free_limb_uvecs, pnt_labels, #model.labels_,
#     #                        SP_AXS[row_idx,col_idx], SP_FIG, display=fig_idx==(n_rows*n_cols-1))
#     #
#     # pickle_file = '../priors/uvec_sphere/cluster_meta.pickle'
#     # with open(pickle_file, 'wb') as file_handle:
#     #     pickle.dump(cluster_meta, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     from agents.helper import pickle_write
#
#     bse_wgt = np.array([.2881, .106, .106, .2881, .106, .106], np.float32)
#     rev_bse_wgt = softmax(-bse_wgt)
#     print('rev_bse_wgt: {}\n'.format(rev_bse_wgt))
#
#     jmc_wgt = np.array([.0534, .0534, .0534, .0534, .0534, .0534, .1451, .0534, .0534, .1451, .1384, .1445], np.float32)
#     rev_jmc_wgt = softmax(-jmc_wgt)
#     print('rev_jmc_wgt: {}\n'.format(rev_jmc_wgt))
#
#     bpc_wgt = np.array([.0424, .0424, .1152, .1152, .0424, .0424, .0424, .1152,
#                         .1152, .1152, .0424, .0424, .0424, .0424, 0.0424], np.float32)
#     rev_bpc_wgt = softmax(-bpc_wgt)
#     print('rev_bpc_wgt: {}\n'.format(rev_bpc_wgt))
#
#     pose_reg_wgts = {'bse_comp_wgts': rev_bse_wgt.reshape((1,1,-1)),
#                      'bpc_comp_wgts': rev_bpc_wgt.reshape((1,1,-1)),
#                      'jmc_comp_wgts': rev_jmc_wgt.reshape((1,1,-1))}
#     pickle_write(pose_reg_wgts, '../priors/pose_reg_wgts_6v12v15.pickle')