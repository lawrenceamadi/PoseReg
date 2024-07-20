# -*- coding: utf-8 -*-
# @Time    : 4/25/2021 9:05 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : pose_regs.py
# @Software: pose.reg

import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import softmax

sys.path.append('../')
from agents.helper import *
from agents.visuals import plot_3d_pose
from agents.rbo_transform_np import save_bone_prior_properties, save_jmc_priors, are_freebones_similar, quaternion_rotate
from agents.rbo_transform_np import save_pose_structure_prior_properties
from agents.rbo_transform_tc import log_likelihood_loss_func, torch_t, guard_div1, to_numpy, FreeBoneOrientation

from agents.helper import processor


def pose_reg_properties_extractor(train_generator, args, supr_subset_tag, frt_tag):
    log_jmc_meta = True
    log_bpc_meta = False
    log_bse_meta = False
    log_blen_meta = False
    visualize_pose = True
    n_viz_subplots = 16 # 16 or 18
    is_torch_op = args.gen_pose_priors >= -2
    induce_varying_bone_lengths = True#len(subjects_train)<=3
    blen_std = 0.05 #*** was .01 [09/03/22], paper .05
    bone_metadata = {'bone_lengths':[], 'bone_symms':[], 'bone_ratios':[]}
    JMC_JOINT_CONFIG = eval('JMC_{}_JOINT_CONFIG'.format(args.jmc_fbo_ops_type.upper()))
    ordered_joint_names = list(JMC_JOINT_CONFIG.keys())

    fb_orientation_vecs = {}
    for joint_name in ordered_joint_names:
        fb_orientation_vecs[joint_name] = [] # was joint_id

    if visualize_pose:
        n_rows, n_cols = (3, 6) if n_viz_subplots==18 else (2, 8)
        figsize_wxh = (15, 10) if n_viz_subplots==18 else (20, 8)
        SP_FIG, SP_AXS = plt.subplots(n_rows, n_cols, subplot_kw={'projection':'3d'}, figsize=figsize_wxh)
        SP_FIG.subplots_adjust(left=0.0, right=1.0, wspace=-0.0)
        SP_FIG2, SP_AXS2 = plt.subplots(n_rows, n_cols, subplot_kw={'projection':'3d'}, figsize=figsize_wxh)
        SP_FIG2.subplots_adjust(left=0.0, right=1.0, wspace=-0.0)
        FIG_BONE18_ORDER = {
           'Abdomen':(0,0), 'Thorax':(0,1), 'Head':(0,2), 'UFace':(0,3), 'RHip':(0,4), 'LHip':(0,5),
           'RShoulder':(1,0), 'LShoulder':(1,1), 'RLeg':(1,2), 'LLeg':(1,3), 'RThigh':(1,4), 'LThigh':(1,5),
           'RBicep':(2,0), 'LBicep':(2,1), 'RForearm':(2,2), 'LForearm':(2,3), 'Camera-Pose':(2,4), 'Upright-Pose':(2,5)}
        FIG_BONE16_ORDER = {'UFace':(0,0), 'Head':(0,1), 'Thorax':(1,0), 'Abdomen':(1,1),
            'RHip':(0,2), 'RThigh':(0,3), 'RLeg':(0,4), 'RShoulder':(0,5), 'RBicep':(0,6), 'RForearm':(0,7),
            'LHip':(1,2), 'LThigh':(1,3), 'LLeg':(1,4), 'LShoulder':(1,5), 'LBicep':(1,6), 'LForearm':(1,7)}
        FIG_BONE_ORDER = FIG_BONE18_ORDER if n_viz_subplots==18 else FIG_BONE16_ORDER

        # Extract and organize Pose Priors configuration parameters
    n_fbjnts = len(ordered_joint_names) # -->fbj
    quad_kpt_idxs_rmtx, xy_yx_axis_dirs, xy_yx_idxs, z_axis_ab_idxs, yx_axis_ab_idxs, nr_fb_idxs, _, _ = fboa_rotamatrix_config(
        JMC_RMTX_JOINT_CONFIG, ordered_joint_names, args.group_jmc, args.quintuple, for_torch=is_torch_op)
    rot_matrix_fb = FreeBoneOrientation(args.batch_size, xy_yx_axis_dirs, z_axis_ab_idxs, yx_axis_ab_idxs, xy_yx_idxs,
                                        n_fbs=n_fbjnts, quintuple=args.quintuple, validate_ops=True, rot_tfm_mode=0,
                                        nr_fb_idxs=nr_fb_idxs, ret_mode=-1)
    quad_kpt_idxs_quat, axis1_quadrant, axis2_quadrant, plane_proj_mult, hflip_mult, nr_fb_idxs = fboa_quaternion_config(
        JMC_QUAT_JOINT_CONFIG, list(JMC_QUAT_JOINT_CONFIG.keys()), args.group_jmc, args.quintuple, for_torch=is_torch_op)
    quaternion_fb = FreeBoneOrientation(args.batch_size, quad_uvec_axes1=axis1_quadrant, quad_uvec_axes2=axis2_quadrant,
                                        plane_proj_mult=plane_proj_mult, hflip_multiplier=hflip_mult, rot_tfm_mode=1,
                                        nr_fb_idxs=nr_fb_idxs, quintuple=args.quintuple, validate_ops=True, ret_mode=-1)
    if args.n_bone_ratios==15:
        BPE_ORDERED_TAGS, BONE_RATIO_PAIR_IDXS_A, BONE_RATIO_PAIR_IDXS_B = select_bone_ratios(VPOSE3D_BONE_ID_2_IDX)
    else:
        BPE_ORDERED_TAGS, BONE_RATIO_PAIR_IDXS_A, BONE_RATIO_PAIR_IDXS_B = all_bone_ratio_combo(VPOSE3D_BONE_ID_2_IDX)

    jnt_cnt = 0
    batch_cnt = 1
    n_bones = 16
    n_ratios = len(BPE_ORDERED_TAGS) # -->brp
    jnts_meta_max = np.zeros((n_fbjnts,3), dtype=np.float32) #8
    jnts_meta_min = np.full((n_fbjnts,3), dtype=np.float32, fill_value=np.inf) #8

    if hflip_mult is not None: hflip_multiplier = to_numpy(hflip_mult)
    print('[INFO] Generating Pose Prior properties from supervised training subset..')
    if args.gen_pose_priors<=-2: os.makedirs(os.path.join('priors', supr_subset_tag, 'properties'), exist_ok=True)

    # Iterate over 3D-pose ground-truths and collect bone-length and joint orientation priors
    for _, batch_3d, _ in train_generator.next_epoch():
        n_supr_samples, n_supr_frames = batch_3d.shape[:2]
        inputs_3d = batch_3d.astype(np.float32) # (?,f,17,3)
        # Pose post processing **IMPORTANT**
        inputs_3d[:,:,0] = 0  # necessary to orient pose at pelvis irrespective of trajectory
        if is_torch_op: inputs_3d = torch.from_numpy(inputs_3d.astype('float32')).to(processor)

        if log_blen_meta or log_bse_meta or log_bpc_meta:
            dists_m = inputs_3d[:, :, BONE_CHILD_KPTS_IDXS] - inputs_3d[:, :, BONE_PARENT_KPTS_IDXS] # -->(?,f,16,3)
            all_bone_lengths_m = torch.linalg.norm(dists_m, dim=3)  # (?,f,16,3) --> (?,f,16)
            assert (torch.all(torch.ge(all_bone_lengths_m, 0.))), "shouldn't be negative {}".format(all_bone_lengths_m)

            # collect bse and bone length metadata
            if log_blen_meta:
                bone_metadata['bone_lengths'].append(to_numpy(all_bone_lengths_m))
            if log_bse_meta:
                bone_sym_diffs = torch.abs(all_bone_lengths_m[:,:,RGT_SYM_BONE_INDEXES] -
                                           all_bone_lengths_m[:,:,LFT_SYM_BONE_INDEXES])
                bone_metadata['bone_symms'].append(to_numpy(bone_sym_diffs))

            # collect bpc metadata
            if log_bpc_meta:
                if induce_varying_bone_lengths:
                    gaussian_factors = []
                    if is_torch_op:
                        for i in range(n_bones):
                            gaussian_factors.append(torch.normal(mean=1, std=blen_std, size=(n_supr_samples, n_supr_frames)))
                        bone_gaussian_factors = torch.stack(gaussian_factors, dim=-1).to(processor) # (?,f,16)
                        all_bone_lengths_m *= bone_gaussian_factors
                    else:
                        for i in range(n_bones):
                            gaussian_factors.append(np.random.normal(1, blen_std, size=(n_supr_samples, n_supr_frames)))
                        bone_gaussian_factors = np.stack(gaussian_factors, axis=-1) # (?,f,16)
                        all_bone_lengths_m *= bone_gaussian_factors
                bone_jnt_ratios = guard_div1(all_bone_lengths_m[:,:,BONE_RATIO_PAIR_IDXS_A],
                                             all_bone_lengths_m[:,:,BONE_RATIO_PAIR_IDXS_B]) # (?,f,15)
                bone_metadata['bone_ratios'].append(to_numpy(bone_jnt_ratios))

        # collect jmc metadata
        if log_jmc_meta:
            # necessary to convert coordinates from meters to millimeters
            #if is_torch_op: inputs_3d_mm = inputs_3d * torch_t(1000)
            #else: inputs_3d_mm = inputs_3d * 1000

            rmtx_quadruplet_kpts = inputs_3d[:,:,quad_kpt_idxs_rmtx,:] # (?,f,17,3)->(?,f,fbj,4,3)
            rmtx_pivot_kpts, rotation_mtxs_3x3, _, rmtx_free_bone_uvecs = \
                rot_matrix_fb(rmtx_quadruplet_kpts) # (?,f,j,4,3)->((?,f,j,1,3), (?,f,j,3,3), (?,f,j,4,4), (?,f,j,3))
            quat_quadruplet_kpts = inputs_3d[:,:,quad_kpt_idxs_quat,:] # (?,f,17,3)->(?,f,fbj,4,3)
            quat_pivot_kpts, quaternion_vec1, quaternion_vec2, quat_free_bone_uvecs = \
                quaternion_fb(quat_quadruplet_kpts) # (?,f,j,4,3)->((?,f,j,1,3), (?,f,j,3,3), (?,f,j,4,4), (?,f,j,3))

            #are_freebones_similar(rmtx_free_bone_uvecs, quat_free_bone_uvecs)
            free_bone_uvecs = rmtx_free_bone_uvecs if args.jmc_fbo_ops_type=='rmtx' else quat_free_bone_uvecs

            jnt_cnt += n_supr_samples*n_supr_frames*n_fbjnts
            for j_idx, joint_name in enumerate(ordered_joint_names):
                jmc_metadata = free_bone_uvecs[:,0,j_idx,:] # (?,3)
                fb_orientation_vecs[joint_name].append(jmc_metadata)
                # log max and min
                jmc_metadata_min = np.min(jmc_metadata, axis=0)
                jmc_metadata_max = np.max(jmc_metadata, axis=0)
                jnts_meta_min[j_idx] = np.where(jmc_metadata_min<jnts_meta_min[j_idx], jmc_metadata_min, jnts_meta_min[j_idx])
                jnts_meta_max[j_idx] = np.where(jmc_metadata_max>jnts_meta_max[j_idx], jmc_metadata_max, jnts_meta_max[j_idx])

            if visualize_pose:
                smp_idx = 0
                rmtx_pvt_kpts = rmtx_pivot_kpts[smp_idx, 0, :, 0]
                rot_mtxs = rotation_mtxs_3x3[smp_idx, 0]
                quat_pvt_kpts = quat_pivot_kpts[smp_idx, 0, :, 0]
                quat_v1 = quaternion_vec1[smp_idx, 0]
                quat_v2 = quaternion_vec2[smp_idx, 0]
                pose_wrt_cam_frm = to_numpy(inputs_3d[smp_idx,0]) if is_torch_op else inputs_3d[smp_idx,0]
                for j_idx, joint_name in enumerate(ordered_joint_names):
                    row_idx, col_idx = FIG_BONE_ORDER[joint_name]
                    # Show quaternion aligned bones
                    quat_pose = quaternion_rotate(pose_wrt_cam_frm, quat_v1[j_idx], quat_v2[j_idx], quat_pvt_kpts[[j_idx]])
                    if args.group_jmc: quat_pose *= hflip_multiplier[0,0,j_idx]
                    jnt_quad_kpts_idxs = get_keypoints_indexes(JMC_QUAT_JOINT_CONFIG[joint_name][0])
                    if len(jnt_quad_kpts_idxs)==5: jnt_quad_kpts_idxs.pop(3) # remove 2nd plane-kpt if quintuple
                    plot_3d_pose(quat_pose, SP_FIG2, SP_AXS2[row_idx,col_idx], joint_name, KPT_2_IDX,
                                 jnt_quad_kpts_idxs, t_tag='Quaternion ')
                    # Show rotation-matrix aligned bones. First translate, then rotate
                    pose_wrt_pvt_frm = pose_wrt_cam_frm - rmtx_pvt_kpts[[j_idx]] # translate to pivot
                    pose_wrt_pvt_frm = np.matmul(rot_mtxs[j_idx], pose_wrt_pvt_frm.T).T # rotate about pivot
                    jnt_quad_kpts_idxs = get_keypoints_indexes(JMC_RMTX_JOINT_CONFIG[joint_name][0])
                    if len(jnt_quad_kpts_idxs)==5: jnt_quad_kpts_idxs.pop(3) # remove 2nd plane-kpt if quintuple
                    plot_3d_pose(pose_wrt_pvt_frm, SP_FIG, SP_AXS[row_idx,col_idx], joint_name, KPT_2_IDX,
                                 jnt_quad_kpts_idxs, t_tag='Rot-Matrix ', display=(n_viz_subplots==16 and j_idx==15))
                if n_viz_subplots==18:
                    # todo: instead of flipping, rotate 180" to reverse the effect of inverted camera projection?
                    upright_pose = pose_wrt_cam_frm * [-1,-1,1] # to reverse effect of inverted camera projection on x & y comp.
                    # todo: relative bone orientation alignment should be executed on upright_pose after reversing camera inversion
                    plot_3d_pose(pose_wrt_cam_frm, SP_FIG2, SP_AXS2[2,4], 'Camera-Pose', KPT_2_IDX, [-1]*4)
                    plot_3d_pose(upright_pose, SP_FIG2, SP_AXS2[2,5], 'Upright-Pose', KPT_2_IDX, [-1]*4, t_tag='Quaternion ')
                    plot_3d_pose(pose_wrt_cam_frm, SP_FIG, SP_AXS[2,4], 'Camera-Pose', KPT_2_IDX, [-1]*4)
                    plot_3d_pose(upright_pose, SP_FIG, SP_AXS[2,5], 'Upright-Pose', KPT_2_IDX, [-1]*4, t_tag='Rot-Matrix ', display=True)
                sys.exit(0)

        batch_cnt += 1
        if args.gen_pose_priors<=-2 and batch_cnt%100==0:
            print('{:>17,} joints from {:>9,} poses or {:5,} batches passed..'.format(jnt_cnt, jnt_cnt//n_fbjnts, batch_cnt))
        if n_supr_samples!=args.batch_size:
            print('[INFO] batch:{} with size {} < {} standard batch-size'.format(batch_cnt, n_supr_samples, args.batch_size))

    # save onetime logged pose priors
    print('{:>17,} joints from {:>9,} poses or {:5,} batches passed..'.format(jnt_cnt, jnt_cnt//n_fbjnts, batch_cnt))
    if args.gen_pose_priors<=-2 and (log_blen_meta or log_bse_meta or log_bpc_meta):
        save_bone_prior_properties(bone_metadata, BPE_ORDERED_TAGS, blen_std, args.data_augmentation,
                                   supr_subset_tag, frt_tag, n_ratios, from_torch=is_torch_op)

    if args.gen_pose_priors<=-2 and log_jmc_meta:
        save_jmc_priors(fb_orientation_vecs, ordered_joint_names, n_fbjnts, args,
                        supr_subset_tag, frt_tag, JMC_JOINT_CONFIG, from_torch=is_torch_op)

    print("Done.\n\njnt_meta_min:\n{}\n\njnt_meta_max:\n{}".format(jnts_meta_min, jnts_meta_max))


def extract_pose_properties_for_build(train_generator, args, supr_subset_tag, frt_tag):
    log_jmc_meta = True # 16x3 bone orientations
    log_bpc_meta = True # 9 bone ratios, ie. 16-(6+1): excluding half of symmetric bones (left-side) and lower-spine bone
    log_bse_meta = True # 6 normalized difference (diff/right-len) of 6 pairs of symmetric bones
    log_tlen_meta = True # 1 length of torso, ie. thorax + abdomen bone length
    log_ploc_meta = True # 1x3 position of pose wrt camera as the location of pelvis joint
    log_pori_meta = True # 1x3, orientation of pose wrt camera as the vector normal to Nck->Plv->RHp plane
    visualize_pose = False
    induce_varying_bone_lengths = True#len(subjects_train)<=3
    n_viz_subplots = 16 # 16 or 18
    ordered_joint_names_for_jm = list(BONE_ALIGN_CONFIG.keys())
    ordered_bone_names_for_bp = CENTERD_BONES[0:3] + RGT_SYM_BONES # (9,) excluding lower-spine/abdomen and left symmetric bones
    ordered_symm_bone_names = [name[1:len(name)] for name in RGT_SYM_BONES] # (6,)
    bpc_bone_idxs = CENTERD_BONE_INDEXES[0:3] + RGT_SYM_BONE_INDEXES # (9,) excluding lower-spine/abdomen bone
    torso_bone_idxs = CENTERD_BONE_INDEXES[2:4] # thorax/upper-spine and abdomen/lower-spine
    nck_idx, spn_idx, plv_idx, rhp_idx = KPT_2_IDX['Nck'], KPT_2_IDX['Spn'], KPT_2_IDX['Plv'], KPT_2_IDX['RHp']
    blen_std = 0.05 #*** was .01 [09/03/22], paper .05

    fb_orientation_vecs = {}
    for joint_name in ordered_joint_names_for_jm:
        fb_orientation_vecs[joint_name] = [] # was joint_id
    pose_metadata = {'bone_symms':[], 'torso_lens':[], 'b2t_ratios':[], 'pose_locat':[], 'pose_orients':[]}

    if visualize_pose:
        n_rows, n_cols = (3, 6) if n_viz_subplots==18 else (2, 8)
        figsize_wxh = (15, 10) if n_viz_subplots==18 else (20, 8)
        SP_FIG, SP_AXS = plt.subplots(n_rows, n_cols, subplot_kw={'projection':'3d'}, figsize=figsize_wxh)
        SP_FIG.subplots_adjust(left=0.0, right=1.0, wspace=-0.0)
        FIG_BONE18_ORDER = {
            'Abdomen':(0,0), 'Thorax':(0,1), 'Head':(0,2), 'UFace':(0,3), 'RHip':(0,4), 'LHip':(0,5),
            'RShoulder':(1,0), 'LShoulder':(1,1), 'RLeg':(1,2), 'LLeg':(1,3), 'RThigh':(1,4), 'LThigh':(1,5),
            'RBicep':(2,0), 'LBicep':(2,1), 'RForearm':(2,2), 'LForearm':(2,3), 'Camera-Pose':(2,4), 'Upright-Pose':(2,5)}
        FIG_BONE16_ORDER = {'UFace':(0,0), 'Head':(0,1), 'Thorax':(1,0), 'Abdomen':(1,1),
                            'RHip':(0,2), 'RThigh':(0,3), 'RLeg':(0,4), 'RShoulder':(0,5), 'RBicep':(0,6), 'RForearm':(0,7),
                            'LHip':(1,2), 'LThigh':(1,3), 'LLeg':(1,4), 'LShoulder':(1,5), 'LBicep':(1,6), 'LForearm':(1,7)}
        FIG_BONE_ORDER = FIG_BONE18_ORDER if n_viz_subplots==18 else FIG_BONE16_ORDER

    # Extract and organize Pose Priors configuration parameters
    n_fbjnts = len(ordered_joint_names_for_jm) # -->fbj
    quad_kpt_idxs_rmtx, xy_yx_axis_dirs, xy_yx_idxs, z_axis_ab_idxs, yx_axis_ab_idxs, nr_fb_idxs, _, _ = fboa_rotamatrix_config(
        BONE_ALIGN_CONFIG, ordered_joint_names_for_jm, args.group_jmc, args.quintuple, for_torch=True)
    rot_matrix_fb = FreeBoneOrientation(args.batch_size, xy_yx_axis_dirs, z_axis_ab_idxs, yx_axis_ab_idxs, xy_yx_idxs,
                                        n_fbs=n_fbjnts, quintuple=args.quintuple, validate_ops=True, rot_tfm_mode=0,
                                        nr_fb_idxs=nr_fb_idxs, ret_mode=-1)

    jnt_cnt = 0
    batch_cnt = 1
    n_bones = 16
    n_bones_for_bp = len(ordered_bone_names_for_bp)
    jnts_meta_max = np.zeros((n_fbjnts,3), dtype=np.float32) #8
    jnts_meta_min = np.full((n_fbjnts,3), dtype=np.float32, fill_value=np.inf) #8
    print('[INFO] Generating Pose Prior properties from reconstructing 3D pose..')
    if args.gen_pose_priors<=-2: os.makedirs(os.path.join('priors', supr_subset_tag, 'properties'), exist_ok=True)

    # Iterate over 3D-pose ground-truths and collect bone-length and joint orientation priors
    for _, batch_3d, _ in train_generator.next_epoch():
        n_supr_samples, n_supr_frames = batch_3d.shape[:2]
        inputs_3d = batch_3d.astype(np.float32) # (?,f,17,3)

        # collect pose location metadata
        if log_ploc_meta:
            # Note, copy is necessary otherwise, values are overridden when set to 0 later
            pose_metadata['pose_locat'].append(np.copy(inputs_3d[:,:,0])) # (?,f,3)

        # Pose post processing **IMPORTANT**
        inputs_3d[:,:,0] = 0  # necessary to orient pose at pelvis irrespective of trajectory

        # collect pose orientation metadata
        if log_pori_meta:
            whole_spine_bones = inputs_3d[:,:,nck_idx] - inputs_3d[:,:,plv_idx] # (?,f,3)
            right_waist_bones = inputs_3d[:,:,rhp_idx] - inputs_3d[:,:,plv_idx] # (?,f,3)
            normal_vecs = np.cross(whole_spine_bones, right_waist_bones, axis=-1) # (?,f,3)
            vec_magnitudes = np.linalg.norm(normal_vecs, axis=-1, keepdims=True) # (?,f,1)
            # Safe division to avoid division-by-zero
            # If you don't pass `out` the indices where (b == 0) will be uninitialized!
            normal_unit_vecs = np.divide(normal_vecs, vec_magnitudes, out=np.zeros_like(normal_vecs), where=vec_magnitudes!=0)
            pose_metadata['pose_orients'].append(normal_unit_vecs) # (?,f,3)

        inputs_3d = torch.from_numpy(inputs_3d.astype('float32')).to(processor)

        if log_bse_meta or log_bpc_meta or log_tlen_meta:
            dists_m = inputs_3d[:, :, BONE_CHILD_KPTS_IDXS] - inputs_3d[:, :, BONE_PARENT_KPTS_IDXS] # -->(?,f,16,3)
            all_bone_lengths_m = torch.linalg.norm(dists_m, dim=3)  # (?,f,16,3) --> (?,f,16)
            assert (torch.all(torch.ge(all_bone_lengths_m, 0.))), "shouldn't be negative {}".format(all_bone_lengths_m)

            # Induce varying bone lengths
            # this is done to generate varying pose metadata from a limited number of subjects (6 or less in H36M)
            # Note: use the same scale factors for symmetric bones
            if induce_varying_bone_lengths:
                bone_indexes = list(range(n_bones))
                bone_gaussian_factors = torch.zeros((n_supr_samples,n_supr_frames,n_bones),
                                                    dtype=torch.float32, device=torch.device(processor)) # (?,f,16)
                for idx in range(n_bones-6):
                    gaussian_factors = torch.normal(mean=1, std=blen_std, size=(n_supr_samples, n_supr_frames))
                    if idx<6: # first 6 guassian_factors are assigned to symmetric bones
                        rgt_bone_idx = VPOSE3D_BONE_ID_2_IDX[RGT_SYM_BONES[idx]]
                        bone_gaussian_factors[:,:,rgt_bone_idx] = gaussian_factors
                        lft_bone_idx = VPOSE3D_BONE_ID_2_IDX[LFT_SYM_BONES[idx]]
                        bone_gaussian_factors[:,:,lft_bone_idx] = gaussian_factors
                        bone_indexes.remove(rgt_bone_idx)
                        bone_indexes.remove(lft_bone_idx)
                    else:
                        ctr_bone_idx = VPOSE3D_BONE_ID_2_IDX[CENTERD_BONES[idx-6]]
                        bone_gaussian_factors[:,:,ctr_bone_idx] = gaussian_factors
                        bone_indexes.remove(ctr_bone_idx)
                assert (len(bone_indexes)==0), 'bone_indexes: {}'.format(bone_indexes)
                all_bone_lengths_m *= bone_gaussian_factors # (?,f,16)
                assert (torch.all(torch.ge(all_bone_lengths_m, 0.))), "shouldn't be negative {}".format(all_bone_lengths_m)

            # collect bsc metadata
            if log_bse_meta:
                bone_sym_diffs = all_bone_lengths_m[:,:,RGT_SYM_BONE_INDEXES] - all_bone_lengths_m[:,:,LFT_SYM_BONE_INDEXES] # (?,f,6)
                norm_sym_diffs = guard_div1(bone_sym_diffs, all_bone_lengths_m[:,:,RGT_SYM_BONE_INDEXES]) # (?,f,6)
                pose_metadata['bone_symms'].append(to_numpy(norm_sym_diffs))

            # collect bpc metadata
            if log_bpc_meta:
                torso_bone_lens = torch.sum(all_bone_lengths_m[:,:,torso_bone_idxs], dim=-1, keepdim=True) # (?,f,1)
                select_bone_ratios = guard_div1(all_bone_lengths_m[:,:,bpc_bone_idxs], torso_bone_lens) # (?,f,9)
                pose_metadata['b2t_ratios'].append(to_numpy(select_bone_ratios))

            # collect torso length metadata
            if log_tlen_meta:
                if not log_bpc_meta:
                    torso_bone_lens = torch.sum(all_bone_lengths_m[:,:,torso_bone_idxs], dim=-1) # (?,f)
                    pose_metadata['torso_lens'].append(torso_bone_lens) # (?,f)
                else: pose_metadata['torso_lens'].append(to_numpy(torso_bone_lens[:,:,0])) # (?,f)

        # collect jmc metadata
        if log_jmc_meta:
            # necessary to convert coordinates from meters to millimeters
            rmtx_quadruplet_kpts = inputs_3d[:,:,quad_kpt_idxs_rmtx,:] # (?,f,17,3)->(?,f,fbj,4,3)
            rmtx_pivot_kpts, rotation_mtxs_3x3, _, free_bone_uvecs = \
                rot_matrix_fb(rmtx_quadruplet_kpts) # (?,f,j,4,3)->((?,f,j,1,3), (?,f,j,3,3), (?,f,j,4,4), (?,f,j,3))

            jnt_cnt += n_supr_samples*n_supr_frames*n_fbjnts
            for j_idx, joint_name in enumerate(ordered_joint_names_for_jm):
                jmc_metadata = free_bone_uvecs[:,0,j_idx,:] # (?,3)
                fb_orientation_vecs[joint_name].append(jmc_metadata)
                # log max and min
                jmc_metadata_min = np.min(jmc_metadata, axis=0)
                jmc_metadata_max = np.max(jmc_metadata, axis=0)
                jnts_meta_min[j_idx] = np.where(jmc_metadata_min<jnts_meta_min[j_idx], jmc_metadata_min, jnts_meta_min[j_idx])
                jnts_meta_max[j_idx] = np.where(jmc_metadata_max>jnts_meta_max[j_idx], jmc_metadata_max, jnts_meta_max[j_idx])

            if visualize_pose:
                smp_idx = 0
                rmtx_pvt_kpts = rmtx_pivot_kpts[smp_idx, 0, :, 0]
                rot_mtxs = rotation_mtxs_3x3[smp_idx, 0]
                pose_wrt_cam_frm = to_numpy(inputs_3d[smp_idx,0])
                for j_idx, joint_name in enumerate(ordered_joint_names_for_jm):
                    row_idx, col_idx = FIG_BONE_ORDER[joint_name]
                    # translate, then rotate
                    pose_wrt_pvt_frm = pose_wrt_cam_frm - rmtx_pvt_kpts[[j_idx]] # translate to pivot
                    pose_wrt_pvt_frm = np.matmul(rot_mtxs[j_idx], pose_wrt_pvt_frm.T).T # rotate about pivot
                    jnt_quad_kpts_idxs = get_keypoints_indexes(BONE_ALIGN_CONFIG[joint_name][0])
                    if len(jnt_quad_kpts_idxs)==5: jnt_quad_kpts_idxs.pop(3) # remove 2nd plane-kpt if quintuple
                    plot_3d_pose(pose_wrt_pvt_frm, SP_FIG, SP_AXS[row_idx,col_idx], joint_name, KPT_2_IDX,
                                 jnt_quad_kpts_idxs, t_tag='Rot-Matrix ', display=(n_viz_subplots==16 and j_idx==15))
                if n_viz_subplots==18:
                    # todo: instead of flipping, rotate 180" to reverse the effect of inverted camera projection?
                    upright_pose = pose_wrt_cam_frm * [-1,-1,1] # to reverse effect of inverted camera projection on x & y comp.
                    # todo: relative bone orientation alignment should be executed on upright_pose after reversing camera inversion
                    plot_3d_pose(pose_wrt_cam_frm, SP_FIG, SP_AXS[2,4], 'Camera-Pose', KPT_2_IDX, [-1]*4)
                    plot_3d_pose(upright_pose, SP_FIG, SP_AXS[2,5], 'Upright-Pose', KPT_2_IDX, [-1]*4, t_tag='Rot-Matrix ', display=True)
                sys.exit(0)

        batch_cnt += 1
        if args.gen_pose_priors<=-2 and batch_cnt%100==0:
            print('{:>17,} joints from {:>9,} poses or {:5,} batches passed..'.format(jnt_cnt, jnt_cnt//n_fbjnts, batch_cnt))
        if n_supr_samples!=args.batch_size:
            print('[INFO] batch:{} with size {} < {} standard batch-size'.format(batch_cnt, n_supr_samples, args.batch_size))

    # save onetime logged pose priors
    print('{:>17,} joints from {:>9,} poses or {:5,} batches passed..'.format(jnt_cnt, jnt_cnt//n_fbjnts, batch_cnt))
    if args.gen_pose_priors<=-2 and (log_tlen_meta or log_bse_meta or log_bpc_meta):
        save_pose_structure_prior_properties(pose_metadata, fb_orientation_vecs, BONE_ALIGN_CONFIG, args,
                            ordered_joint_names_for_jm, ordered_bone_names_for_bp, ordered_symm_bone_names,
                                             n_fbjnts, n_bones_for_bp, blen_std, supr_subset_tag, frt_tag)

    print("Done.\n\njnt_meta_min:\n{}\n\njnt_meta_max:\n{}".format(jnts_meta_min, jnts_meta_max))


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
    sigma_variance = np.array(sigma_variance, dtype=np.float32).reshape((1,1,n_ratios)) # (1,1,r)
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
    loglihood_wgts = np.full((1,1,n_ratios), fill_value=k_rank_wgt, dtype=np.float32)
    loglihood_wgts = torch.from_numpy(loglihood_wgts).to(processor)

    return sigma_variance, exponent_coef, mu_mean, likeli_max, likeli_argmax, logli_spread, logli_mean, \
           logli_std, logli_min, logli_span, logli_wgt, log_of_spread, move_up_const, per_comp_wgts, loglihood_wgts


def fboa_quaternion_config(joint_cfgs, ordered_joint_names, group_jnts=True, quintuple=False, for_torch=True):
    qset_kpt_idxs = []
    axis1_quadrant_uvecs = []
    axis2_quadrant_uvecs = []
    hflip_multiplier = []
    non_reflect_idxs = []
    hflip_fbjnts = ('LHip','LThigh','LLeg','LShoulder','LBicep','LForearm','LWaist','LScapula') # Last 2 for deprecation

    for j_idx, joint_name in enumerate(ordered_joint_names):
        (qset_kpts, quadrant_uvecs_pair) = joint_cfgs[joint_name]
        if not quintuple and len(qset_kpts)==5: qset_kpts.pop(3) # remove 2nd plane-kpt
        qset_kpt_idxs.append(get_keypoints_indexes(qset_kpts))
        axis1_quadrant_uvecs.append(quadrant_uvecs_pair[0])
        axis2_quadrant_uvecs.append(quadrant_uvecs_pair[1])
        if group_jnts:
            if joint_name in hflip_fbjnts:
                hflip = [-1,1,1]
            else:
                hflip = [1,1,1]
                non_reflect_idxs.append(j_idx)
            hflip_multiplier.append(hflip) # flip on x-axis
        else: non_reflect_idxs.append(j_idx)

    n_joints = len(qset_kpt_idxs)
    axis1_quadrant_uvecs = np.array(axis1_quadrant_uvecs, dtype=np.float32).reshape((1,1,n_joints,3))
    axis2_quadrant_uvecs = np.array(axis2_quadrant_uvecs, dtype=np.float32).reshape((1,1,n_joints,3))
    plane_proj_multiplier = np.mod(np.abs(axis1_quadrant_uvecs)+1, 2) # changes [-1/1,0,0]->[0,1,1] [0,-1/1,0]->[1,0,1]
    assert (np.all(plane_proj_multiplier+np.abs(axis1_quadrant_uvecs)==np.ones((3,))))
    if group_jnts:
        hlip_multiplier = np.array(hflip_multiplier, dtype=np.float32).reshape((1,1,n_joints,3))
    else: hlip_multiplier = None

    if for_torch:
        axis1_quadrant_uvecs = torch.from_numpy(axis1_quadrant_uvecs).to(processor)
        axis2_quadrant_uvecs = torch.from_numpy(axis2_quadrant_uvecs).to(processor)
        plane_proj_multiplier = torch.from_numpy(plane_proj_multiplier).to(processor)
        if group_jnts: hlip_multiplier = torch.from_numpy(hlip_multiplier).to(processor)

    return qset_kpt_idxs, axis1_quadrant_uvecs, axis2_quadrant_uvecs, plane_proj_multiplier, hlip_multiplier, non_reflect_idxs


def fboa_rotamatrix_config(joint_cfgs_dict, ordered_joint_names, group_jnts=True, quintuple=False,
                           drop_face_bone=False, rev_direction=-1, for_torch=True, display_summary=False):
    qset_kpt_idxs = []
    xy_idxs, yx_idxs = [], []
    xy_axis_dirs, yx_axis_dirs = [], []
    z_axis_a_idxs, z_axis_b_idxs = [], []
    yx_axis_a_idxs, yx_axis_b_idxs = [], []
    non_reflect_idxs = []
    change_direction = [('LThigh','LBicep'), ('LLeg','LForearm','LHip','LShoulder','LWaist','LScapula')] # Last 2 for deprecation
    joint_cfgs = copy.deepcopy(joint_cfgs_dict) # work with deep copy so that alterations will not affect original dictionary

    for j_idx, fb_jnt_id in enumerate(ordered_joint_names):
        (qset_kpts, xy_yx_dir, xy_idx, yx_idx, z_axb_indexes, yx_axb_indexes) = joint_cfgs[fb_jnt_id][:6]
        # Note, pop() operation affects qset_kpts list in joint_cfgs
        assert (not quintuple or len(qset_kpts)==5), 'quintuple:{} -> {} are 5'.format(quintuple, qset_kpts)
        if not quintuple and len(qset_kpts)==5: qset_kpts.pop(3) # remove 2nd plane-kpt
        qset_kpt_idxs.append(get_keypoints_indexes(qset_kpts))
        if group_jnts and fb_jnt_id in change_direction[0]:
            xy_axis_dirs.append(xy_yx_dir[0]*rev_direction)
        else: xy_axis_dirs.append(xy_yx_dir[0])
        if group_jnts and fb_jnt_id in change_direction[1]:
            yx_axis_dirs.append(xy_yx_dir[1]*rev_direction)
        else: yx_axis_dirs.append(xy_yx_dir[1])
        if group_jnts:
            if not (fb_jnt_id in change_direction[0] or fb_jnt_id in change_direction[1]):
                non_reflect_idxs.append(j_idx)
        else: non_reflect_idxs.append(j_idx)
        xy_idxs.append(xy_idx)
        yx_idxs.append(yx_idx)
        z_axis_a_idxs.append(z_axb_indexes[0])
        z_axis_b_idxs.append(z_axb_indexes[1])
        yx_axis_a_idxs.append(yx_axb_indexes[0])
        yx_axis_b_idxs.append(yx_axb_indexes[1])

    n_joints = len(qset_kpt_idxs)
    xy_axis_dirs = np.array(xy_axis_dirs, dtype=np.float32).reshape((1,1,n_joints,1))
    yx_axis_dirs = np.array(yx_axis_dirs, dtype=np.float32).reshape((1,1,n_joints,1))

    if for_torch:
        xy_axis_dirs = torch.from_numpy(xy_axis_dirs).to(processor)
        yx_axis_dirs = torch.from_numpy(yx_axis_dirs).to(processor)

    # Build keypoint/joint-to-bone MPBOE error mapping configuration
    joint_2_bone_mapping = {}
    for fb_idx, fb_jnt_id in enumerate(ordered_joint_names):
        if drop_face_bone and fb_jnt_id=='UFace':
            continue # skip to avoid accumulating UFace MPBOE into J-MPBOE of 'Skl','Nck','RSh','LSh' joints
        fb_qset_kpts = list(joint_cfgs[fb_jnt_id][0])
        for qs_idx, jnt in enumerate(fb_qset_kpts):
            qs_kpt_type, qs_kpt_wgt = QUINTUPLET_KPT_WGT[qs_idx] if quintuple else QUADRUPLET_KPT_WGT[qs_idx]
            if not joint_2_bone_mapping.get(jnt, False): joint_2_bone_mapping[jnt] = (([],[]), ([],[]))
            joint_2_bone_mapping[jnt][0][0].append(fb_idx)
            joint_2_bone_mapping[jnt][1][0].append(fb_jnt_id)
            joint_2_bone_mapping[jnt][0][1].append(qs_kpt_wgt)
            joint_2_bone_mapping[jnt][1][1].append(qs_kpt_type)

    if display_summary:
        print('joint_2_bone_mapping:')
        for key, value in joint_2_bone_mapping.items():
            contributors = {}
            contributing_types = value[1][1]
            for quad_kpt_type in set(contributing_types):
                quad_kpt_type_cnt = 0
                for kpt_type in contributing_types:
                    if quad_kpt_type==kpt_type: quad_kpt_type_cnt += 1
                contributors[quad_kpt_type] = quad_kpt_type_cnt
            print('{}: {} unique bone contributors >> {}\n\t{}\n\t{}'.format(
                key, len(set(value[1][0])), contributors, value[0], value[1])) #, sys.exit()

    # Generate index list for rearranging videopose3d bones to orientation bones
    net_2_orient_bone_idxs = []
    for bone_id in ordered_joint_names:
        net_2_orient_bone_idxs.append(VPOSE3D_BONE_ID_2_IDX[bone_id])

    return np.asarray(qset_kpt_idxs), (xy_axis_dirs, yx_axis_dirs), (xy_idxs, yx_idxs), \
           (z_axis_a_idxs, z_axis_b_idxs), (yx_axis_a_idxs, yx_axis_b_idxs), \
           non_reflect_idxs, joint_2_bone_mapping, net_2_orient_bone_idxs


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
        assert (k_rank_wgt==0.1 or k_rank_wgt==0.01), 'k_rank_wgt={} for rank:{} k_rank:{}'.format(k_rank_wgt, rank, k_rank)

        loglihood_wgts.append(np.full((1,1,len(rank_ordered_jnt_tags)), fill_value=k_rank_wgt, dtype=np.float32))
        k_jnts_indexes = []
        for i in range(rank): k_jnts_indexes.append([])

        for (orient_id, orient_jnts) in rank_ordered_jnt_tags.items():
            jm_orient_ids.append(orient_id)
            uvec_mu = uvec_means[r_idx][orient_id]
            covariance_mtx = covariance_matrices[r_idx][orient_id]
            if inv_covariance_matrices is not None:
                inv_covariance_mtx = inv_covariance_matrices[r_idx][orient_id]
            else: inv_covariance_mtx = np.linalg.inv(covariance_mtx)

            cov_dot_inv = covariance_mtx.dot(inv_covariance_mtx)
            all_are_identity = np.all(np.isclose(cov_dot_inv, np.eye(k_rank), atol=5e-03))
            assert(all_are_identity), '{} inverse matrix-{} test: inv-covariance:\n{}'.format(orient_id, k_rank, cov_dot_inv)

            for pair_idx, jnt_id in enumerate(orient_jnts):
                k_jnts_indexes[pair_idx].append(get_id_index(ordered_joint_names, jnt_id))

            uvec_mean_kx1.append(uvec_mu)
            covariance_kxk.append(covariance_mtx)
            inv_covariance_kxk.append(inv_covariance_mtx)
            k_likeli_max.append(loglihood_metadata[r_idx][orient_id][0])
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
        k_exponent_coefs = mvpdf_exponent_coefficient(covariance_kxk, k=k_rank, expand_dims=True) # (1,1,j,1,1)
        k_likeli_max = np.array(k_likeli_max, dtype=np.float32).reshape((1,1,n_components))
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
    per_comp_wgts = torch.from_numpy(per_comp_wgts).to(processor)

    return jmc_rank_priors, loglihood_wgts, per_comp_wgts
