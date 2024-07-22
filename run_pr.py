# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Extensive modification of VideoPose3D source code
# by researchers at the Visual Computing Lab @ IIT


from common.arguments import parse_args
import torch
torch.cuda.empty_cache()

import torch.optim as optim
import time
import errno

from agents.pose_regs import *
from agents.rbo_transform_tc import *
from agents.helper import *
from common.camera import *
from common.model import *
from common.loss import *
from common.utils import deterministic_random
from common.generators import ChunkedGenerator, UnchunkedGenerator
from data.data_tap import fetch

np.set_printoptions(precision=4, linewidth=128, suppress=True)
torch.set_printoptions(precision=7)

args = parse_args()

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = args.dataset_home_dir + 'data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
    print('[INFO] Training on Human36M 3D poses')
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
    print('[INFO] Training on HumanEva 3D poses')
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset(args.dataset_home_dir + 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
    print('[INFO] Training on {} custom dataset 3D poses'.format(args.dataset))
else:
    raise KeyError('Invalid dataset')

root_kpt_idx = KPT_2_IDX[args.root_keypoint]
print('Preparing data with root-kpt:{} at index:{} ...'.format(args.root_keypoint, root_kpt_idx))
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pose_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                # Remove global offset, but keep trajectory in root-kpt (eg. pelvis) position
                pose_offset = pose_3d[:, [root_kpt_idx]] # (?,1,3)
                assert (np.all(pose_offset[:,:,2]>=1)), 'Trajectory W-MPJPE assumes depth is always positive (i.e. >=0)'
                pose_3d -= pose_offset # (?,j,3)
                pose_3d[:, root_kpt_idx] = pose_offset[:, 0]
                positions_3d.append(pose_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
# load gt 2D keypoints just to retrieve symmetric keypoint indexes (kps_left, kps_right)
gt2d_keypoints = np.load(args.dataset_home_dir + 'data_2d_' + args.dataset + '_gt.npz', allow_pickle=True)
keypoints_metadata = gt2d_keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left = list(keypoints_symmetry[0]) #  [4, 5, 6, 11, 12, 13]
kps_right = list(keypoints_symmetry[1]) # [1, 2, 3, 14, 15, 16]
gt2d_keypoints =  gt2d_keypoints['positions_2d'].item()
joints_left = list(dataset.skeleton().joints_left()) #   >> [4, 5, 6, 11, 12, 13] for 17-skeleton-joints
joints_right = list(dataset.skeleton().joints_right()) # >> [1, 2, 3, 14, 15, 16] for 17-skeleton-joints
# then load preset 'args.keypoints' (gt, hr, cpn, or detectron, etc.) for training and inference
joints_inbtw = get_keypoints_indexes(['Plv','Spn','Nck','Nse','Skl'])
keypoints = np.load(args.dataset_home_dir + 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints = keypoints['positions_2d'].item()

# Set checkpoint directory (Added by Lawrence, 09/09/21)
log_func_modes = \
    (args.logli_func_mode, args.norm_logli_func_mode, args.reverse_logli_func_mode, args.shift_logli_func_mode)
supv_subjects = args.subjects_train.replace(",", ".")
supv_sbj_subset = '' if args.subset==1 else str(args.subset).replace("0.", ".")
supv_subset_tag = '{}{}'.format(supv_subjects, supv_sbj_subset)
frm_rate_tag = '' if args.downsample==1 else '_ds{}'.format(args.downsample)
pose2d_tag = args.keypoints[:3]+'2d'
arc_tag = 'a'+args.architecture.replace(',', '')

# Prune 2d-poses for 1-1 correspondence with 3d-poses
for subject in dataset.subjects():
    assert (subject in keypoints), 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert (action in keypoints[subject]), 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        # Short-Data: Skip 2D-poses without corresponding 3D-poses
        if 'positions_3d' not in dataset[subject][action]:
            print('[ALERT] Missing corresponding 3D-poses for {}:"{}"'.format(subject, action))
            continue
        sa_n_cam_views = len(keypoints[subject][action])
        assert (sa_n_cam_views==4), "We assume all subject's actions have 4 views not {}".format(sa_n_cam_views)
        for cam_idx in range(sa_n_cam_views):
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            kpt2d_length = keypoints[subject][action][cam_idx].shape[0]
            msg_tmp = 'GT 2D kpts frames of {}-{} at cam {} is less than {} 3D frames'
            assert (kpt2d_length >= mocap_length), msg_tmp.format(subject, action, cam_idx, mocap_length)
            assert (gt2d_keypoints[subject][action][cam_idx].shape[0]==kpt2d_length), \
                '{} vs. {}'.format(gt2d_keypoints[subject][action][cam_idx].shape[0], kpt2d_length)

            if kpt2d_length > mocap_length:
                # Shorten sequence
                print('[ALERT] Excluding {} extra 2d-pose sequence of {}:"{}"'.format(kpt2d_length-mocap_length, subject, action))
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                gt2d_keypoints[subject][action][cam_idx] = gt2d_keypoints[subject][action][cam_idx][:mocap_length]

        assert (len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d']))

# Normalize 2d pose pixel coordinates
for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps
            gt_kps = gt2d_keypoints[subject][action][cam_idx]
            gt_kps[..., :2] = normalize_screen_coordinates(gt_kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            gt2d_keypoints[subject][action][cam_idx] = gt_kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else: subjects_test = [args.viz_subject]

semi_supervised = len(subjects_semi) > 0
if semi_supervised and not dataset.supports_semi_supervised():
    raise RuntimeError('Semi-supervised training is not implemented for this dataset')

mpjpe_coef, traj_coef, mpboe_coef = \
    torch.from_numpy(np.float32([float(x) for x in args.supervised_loss_coef.split(',')])).to(processor)
print('[INFO] Fully-Supervised branch loss coefficients >> 3D-POSE:{:.4f} - '
      'AUXILIARY (TRAJECTORY or SCALE):{:.4f} - POSTURE:{:.4F}'.format(mpjpe_coef, traj_coef, mpboe_coef))
if semi_supervised:
    rp2d_coef, mble_coef, pto_coef, bse_coef, bpc_coef, jmc_coef, mce_wcx_coef, mce_ncx_coef = \
        torch.from_numpy(np.float32([float(x) for x in args.semi_supervised_loss_coef.split(',')])).to(processor)
    print('[INFO] Weakly-Supervised branch loss coefficients:\n\t2D-PROJ:{:.5f} - MBLE:{:.5f} - '
          'PTO:{:.5f} - SBE:{:.5f} - BPE:{:.5f} - JMC:{:.5f} - MCE_wcx:{:.5f} - MCE_ncx:{:.5f}'.
          format(rp2d_coef, mble_coef if args.mean_bone_length_term else 0, pto_coef if args.pelvis_placement_term else 0,
                 bse_coef if args.bone_symmetry_term else 0, bpc_coef if args.bone_proportion_term else 0,
                 jmc_coef if args.joint_mobility_term else 0, mce_wcx_coef, mce_ncx_coef))
    r_template = '{:>5} {:>4} {:<6} {:<6} {:<8} | {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}' \
                 ' {:<6} {:<6} {:<6} | {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}'
    r_header = r_template.format('Epoch','mins','iterFS','iterWS','lr',
                                 '3dp-e','auxi-e','mcPose','mcPost','2dp-e','mble', 'pto-e','bse','bpc-e','jmc-e',
                                 '2dTrnF','2dTrnW','2dpVal','axTrnF','axTrnW','auxVal','3dTrnF','3dTrnW','3dpVal')
else:
    r_template = '{:>5} {:>4} {:<6} {:<8} | {:<6} {:<6} | {:<6} {:<6} {:<6} | {:<6} {:<6} | {:<6} {:<6}'
    r_header = r_template.format('Epoch','mins','iterFS','lr','pose-c','post-c','pose-e','pae',
                                 'post-e','trjTrn','trjVal','3dEvaF','3dpVal')
auxi_print_fmt = ' {:>6.1f} {:>6.1f} {:>6.1f} ' if args.projection_type>0 else ' {:>6.4f} {:>6.4f} {:>6.4f} '
auxi_s = 1000 if args.projection_type>0 else 1
db_mode = args.db_mode

ignore_nose_kpt = False # Default setting unless estimating less than 17 joint skeletal pose
training_with_multi_gpu = torch.cuda.device_count()>1 and args.multi_gpu_training
print('[INFO] semi_supervised:{} - use_auxi_model:{} - multi-gpu-use:{}'.
      format(semi_supervised, args.use_auxiliary_model, training_with_multi_gpu))


# Use pre-parameterized pose prior regularizer parameters
# ------------------------------------------------------------------------------------------------------------------
if semi_supervised and args.gen_pose_priors>=0:
    ls_tag = args.log_likelihood_spread
    nbr_tag = args.n_bone_ratios
    cov_tag = args.pdf_covariance_type # best: 'noctr'
    jmc_grp_tag = 'grpjnt' if args.group_jmc else 'perjnt'
    bpc_grp_tag = 'grpprp' if args.group_bpc else 'perprp'
    dup_tag = '' if args.duplicate==0 else '+{}'.format(args.duplicate)
    frt_tag = '' if args.gen_pose_priors==2 else frm_rate_tag
    if args.gen_pose_priors==2: set_tag = 'S1.S5.S6.S7.S8'
    else: set_tag = supv_subset_tag
    print('[INFO] generating PPR parameters for {} subset...'.format(set_tag))

    # Initialize necessary torch constants
    log_lihood_eps = torch_t(args.log_likelihood_epsilon)  # 1e-5

    # Configure MCE, BPC, & JMC
    whole_num = int(np.ceil(args.mce_type))
    if whole_num == args.mce_type: # no decimal
        with_cam_ext_mce_type, wout_cam_ext_mce_type = 0, 0
        if whole_num > 0: with_cam_ext_mce_type = whole_num
        elif whole_num < 0: wout_cam_ext_mce_type = whole_num
    else: # contains whole number and decimal
        assert (args.mce_type < 0), 'mce_type must be <0 if it contains decimal'
        wout_cam_ext_mce_type = whole_num
        with_cam_ext_mce_type = int((args.mce_type - whole_num) * -10)
    assert (wout_cam_ext_mce_type<=0 and with_cam_ext_mce_type>=0)
    apply_mv_pose_consistency = with_cam_ext_mce_type!=0 or wout_cam_ext_mce_type!=0 and args.multi_cam_views>1
    assert (args.use_auxiliary_model or (args.projection_type<0 and with_cam_ext_mce_type==0)), 'auxi->(orth & no pose MCE)'
    print('[INFO] wout_cam_ext_mce_type:{} with_cam_ext_mce_type:{} apply_mv_pose_consistency:{}'.
          format(wout_cam_ext_mce_type, with_cam_ext_mce_type, apply_mv_pose_consistency))
    print('[INFO] Computing: Auxiliary:{} - 2D-Proj:{} - MBLE:{} - PTO:{} - BSE:{} - BPC:{} - JMC:{} - MCE:{}'.
          format(args.use_auxiliary_model, args.projection_type!=0, args.mean_bone_length_term, args.pelvis_placement_term,
                 args.bone_symmetry_term, args.bone_proportion_term, args.joint_mobility_term, apply_mv_pose_consistency))
    print('[INFO] Other weakly-supervised conditional checks: Bone_len:{} - Repos_Plv:{}'.format(
            args.bone_symmetry_term or args.bone_proportion_term, args.reposition_to_origin and (
                args.bone_symmetry_term or args.bone_proportion_term or args.joint_mobility_term)))

    # Extract and organize Pose Priors configuration parameters
    if args.gen_pose_priors>0:
        # BPC
        std_tag = str(args.induced_bonelen_std).replace('0.', '.')
        bpc_priors = pickle_load('./priors/{}/br_prior_params_blstd{}_wxaug_{}{}_{}_{:.0e}.pickle'.
                                 format(set_tag, std_tag, nbr_tag, frt_tag, bpc_grp_tag, ls_tag))
        bone_prop_pair_idxs_A, bone_prop_pair_idxs_B = extract_bpc_config_params(bpc_priors, VPOSE3D_BONE_ID_2_IDX)
        bpc_variance, bpc_exponent_coefs, bpc_mean, bpc_max_likelihoods, bpc_likeli_argmax, bpc_logli_spread, \
        bpc_logli_mean, bpc_logli_std, bpc_logli_min, bpc_logli_span, bpc_ilmw_wgts, bpc_log_of_spread, bpc_move_up_const, \
        per_comp_bpc_wgts, bpc_loglihood_wgts = configure_bpc_likelihood(bpc_priors, log_func_modes, log_lihood_eps)

        # JMC
        fbq_tag = 5 if args.quintuple else 4
        rot_tag = '_quat' if args.jmc_fbo_ops_type=='quat' else '_rmtx'#''
        jmc_priors = pickle_load('./priors/{}/fb_prior_params{}_{}{}{}_wxaug_16_{}{}_{:.0e}.pickle'.
                                 format(set_tag, frt_tag, cov_tag, rot_tag, fbq_tag, jmc_grp_tag, dup_tag, ls_tag))
        if args.jmc_fbo_ops_type=='quat':
            jmc_qset_kpt_idxs, axis1_quadrant, axis2_quadrant, plane_proj_mult, hflip_mult, nr_fb_idxs = fboa_quaternion_config(
                jmc_priors['joint_align_config'], jmc_priors['joint_order'], args.group_jmc, args.quintuple)
            jmc_fb_orient = FreeBoneOrientation(args.ws_batch_size, quad_uvec_axes1=axis1_quadrant, quad_uvec_axes2=axis2_quadrant,
                                                plane_proj_mult=plane_proj_mult, hflip_multiplier=hflip_mult,
                                                quintuple=args.quintuple, validate_ops=db_mode, ret_mode=0, rot_tfm_mode=1)
        else:
            jmc_qset_kpt_idxs, xy_yx_axis_dir, xy_yx_idxs, z_axis_ab_idxs, yx_axis_ab_idxs, nr_fb_idxs, _, _ = \
                fboa_rotamatrix_config(jmc_priors['joint_align_config'], jmc_priors['joint_order'], args.group_jmc, args.quintuple)
            jmc_fb_orient = FreeBoneOrientation(args.ws_batch_size, xy_yx_axis_dir, z_axis_ab_idxs, yx_axis_ab_idxs, xy_yx_idxs,
                                nr_fb_idxs=nr_fb_idxs, quintuple=args.quintuple, validate_ops=db_mode, ret_mode=0, rot_tfm_mode=0)
        jmc_params, jmc_loglihood_wgts, per_comp_jmc_wgts = \
            configure_jmc_likelihood(jmc_priors, log_func_modes, log_lihood_eps, args.jmc_ranks)

    # MCE - without camera extrinsic
    mcv_cnt = args.multi_cam_views # multi-cam-views used in semi-supervised pose consistency loss
    if wout_cam_ext_mce_type<0:
        ret_mode = 1 if wout_cam_ext_mce_type==-1 else 0
        mce_qset_kpt_idxs, xy_yx_axis_dir, xy_yx_idxs, z_axis_ab_idxs, yx_axis_ab_idxs, \
        nr_fb_idxs, _, mce_fbo_v2o_bone_idxs = fboa_rotamatrix_config(
            MPBOE_BONE_ALIGN_CONFIG, list(MPBOE_BONE_ALIGN_CONFIG.keys()), group_jnts=False, quintuple=args.quintuple)
        mce_fb_orient = FreeBoneOrientation(args.ws_batch_size, xy_yx_axis_dir, z_axis_ab_idxs, yx_axis_ab_idxs, xy_yx_idxs,
                    nr_fb_idxs=nr_fb_idxs, quintuple=args.quintuple, validate_ops=db_mode, ret_mode=ret_mode, rot_tfm_mode=0)

elif not semi_supervised:
    # For full supervision
    optimize_posture_loss = False if args.mbpoe_posture_type=='none' else True
    if optimize_posture_loss:
        assert (args.projection_type==1)
        ret_fb = 0 if args.mbpoe_posture_type.find('uvec')>=0 else 1 #-2
        scale_by_targ_blen = True if args.mbpoe_posture_type.find('bs-uvec')>=0 else False
        posture_loss_lnorm = True if args.mbpoe_posture_type.find('norm')>=0 else False
        assert (not (ret_fb==1 and posture_loss_lnorm==False)), 'cosine loss does not make sense for fb-vecs. No vec-cos'
        print('[INFO] Fully-supervised posture type: {}{} (l{}-norm:{})'.format(args.mbpoe_posture_type,
            ' - with gt bone length scaling' if scale_by_targ_blen else '', args.posture_norm, posture_loss_lnorm))
        pose_qset_kpt_idxs, xy_yx_axis_dir, xy_yx_idxs, z_axis_ab_idxs, yx_axis_ab_idxs, \
        nr_fb_idxs, _, pose_fbo_v2o_bone_idxs = fboa_rotamatrix_config(
            MPBOE_BONE_ALIGN_CONFIG, list(MPBOE_BONE_ALIGN_CONFIG.keys()), group_jnts=False, quintuple=True)
        targ_pose_fbo = FreeBoneOrientation(args.batch_size, xy_yx_axis_dir, z_axis_ab_idxs, yx_axis_ab_idxs, xy_yx_idxs,
                            nr_fb_idxs=nr_fb_idxs, quintuple=True, validate_ops=db_mode, ret_mode=ret_fb, rot_tfm_mode=0)
        pred_pose_fbo = FreeBoneOrientation(args.batch_size, xy_yx_axis_dir, z_axis_ab_idxs, yx_axis_ab_idxs, xy_yx_idxs,
                            nr_fb_idxs=nr_fb_idxs, quintuple=True, validate_ops=db_mode, ret_mode=ret_fb, rot_tfm_mode=0)
    plv_kpt_idx, spn_kpt_idx, rhp_kpt_idx = KPT_2_IDX['Plv'], KPT_2_IDX['Spn'], KPT_2_IDX['RHp']
    pto_coef, poe_coef, ble_coef = torch_t(1.), torch_t(1.), torch_t(10.)
# ------------------------------------------------------------------------------------------------------------------

# Reconstruct pose by sampling pose prior spaces
if args.gen_pose_priors==3:
    hdir = './priors/S1'
    rpbo_file = 'rpbo_prior_params_noctr_rmtx4_wxaug_16_perjnt+1_1e+00.pickle'
    rpbo_priors_params = pickle_load(os.path.join(hdir, rpbo_file))
    rpps_file = 'rpps_prior_params_blstd.05_wxaug_9_perprp_1e+00.pickle'
    rpps_priors_params = pickle_load(os.path.join(hdir, rpps_file))
    # Reconstruct bones from JMC's Free bone orientation priors
    rbo_qset_kpt_idxs, rbo_xy_yx_axis_dir, rbo_xy_yx_idxs, rbo_z_axis_ab_idxs, rbo_yx_axis_ab_idxs, rbo_nr_fb_idxs, _, _ = \
        fboa_rotamatrix_config(rpbo_priors_params['joint_align_config'], rpbo_priors_params['joint_order'], False, False)
    rpbo_orient = FreeBoneConstruct(rbo_qset_kpt_idxs, 1, rbo_xy_yx_axis_dir, rbo_z_axis_ab_idxs, rbo_yx_axis_ab_idxs,
                                    rbo_xy_yx_idxs, nr_fb_idxs=rbo_nr_fb_idxs, validate_ops=db_mode)
    ordered_fbjnt_recon = ['Abdomen', 'LHip', 'RThigh', 'LThigh', 'RLeg', 'LLeg', 'RShoulder', 'LShoulder',
                           'RBicep', 'LBicep', 'RForearm', 'LForearm', 'Head', 'UFace']
    sampled_pose = torch.zeros((1, 1, 17, 3), dtype=torch.float32, device='cuda') # (?,f,j,3)

    # Sample torso length
    tlen_mean, tlen_var = rpps_priors_params['tlen_mean_variance']
    torso_len = np.random.normal(tlen_mean, np.sqrt(tlen_var))

    # Sample bone/torso ratios (center and right-side bones)
    bone_lengths = {}
    for bone_id in rpps_priors_params['ratio_order']:
        mean, var = rpps_priors_params['br_mean_variance'][bone_id]
        bone_ratio = np.random.normal(mean, np.sqrt(var))
        bone_lengths[bone_id] = bone_ratio * torso_len
        if bone_id=='Thorax':
            bone_lengths['Abdomen'] = (1 - bone_ratio) * torso_len

    # Sample bone symmetry normalized diffs (left-side bones)
    for bone_id in rpps_priors_params['symm_order']:
        mean, var = rpps_priors_params['bsym_mean_variance'][bone_id]
        sym_ndiff = np.random.normal(mean, np.sqrt(var))
        rgt_sym_tag = 'R' + bone_id[1:]
        rgt_sym_blen = bone_lengths[rgt_sym_tag]
        bone_lengths[bone_id] = rgt_sym_blen - rgt_sym_blen*sym_ndiff

    # Sample bone orientations
    bone_orients = {}
    for bone_id in rpbo_priors_params['joint_order']:
        # multivariate normal distribution sampling
        axes_means = rpbo_priors_params['fb_axes_means'][0][bone_id]
        covariance = rpbo_priors_params['fb_covariance'][0][bone_id]
        L = np.linalg.cholesky(covariance)
        Y_mag = 0
        while Y_mag==0:
            X = np.random.randn(1, 3) # sample 1x3 from a normal distribution with mean at 0 and variance 1
            Y = axes_means + X @ L.T
            Y_mag = np.linalg.norm(Y)
            if Y_mag==0: print('Repeating {} orientation sampling..'.format(Y_mag))
        bone_orients[bone_id] = Y / Y_mag # bone orientation unit vector

    # Start by setting RHp (-x,0,0), Plv (0,0,0) and Spn (x,x,x)
    # assuming lower-spine/abdomen free-bone alignment state
    plv_kpt_idx, spn_kpt_idx, nck_kpt_idx, rhp_kpt_idx = KPT_2_IDX['Plv'], KPT_2_IDX['Spn'], KPT_2_IDX['Nck'], KPT_2_IDX['RHp']
    sampled_pose[:,:,rhp_kpt_idx,0] = -bone_lengths['RHip'] # right waist bone
    lspine_bone_vec = bone_lengths['Abdomen'] * bone_orients['Abdomen'] # len * uvec
    sampled_pose[:,:,spn_kpt_idx] = lspine_bone_vec

    # calculate Neck kpt position along the y-axis (0,x,0) given torso length
    lspine_xz_len_sq = np.square(lspine_bone_vec[0]) + np.square(lspine_bone_vec[2])
    lspine_half_len = np.sqrt(np.square(bone_lengths['Abdomen']) - lspine_xz_len_sq)
    uspine_half_len = np.sqrt(np.square(bone_lengths['Thorax']) - lspine_xz_len_sq)
    sampled_pose[:,:,nck_kpt_idx,1] = lspine_half_len + uspine_half_len

    for idx, bone_id in enumerate(ordered_fbjnt_recon):
        b_idx = get_id_index(rpbo_priors_params['joint_order'], bone_id)
        sampled_pose = rpbo_orient(sampled_pose, b_idx) # (?>,f,j,3)
        fb_jnt_idx = rbo_qset_kpt_idxs[idx, -1]
        sampled_pose[0, 0, fb_jnt_idx] = bone_lengths[bone_id] * bone_orients[bone_id] # len * uvec


        

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cams_in_val, _, poses_3d_val, gt_poses_2d_val, poses_2d_val, ignore_nose_kpt = \
    fetch(keypoints, gt2d_keypoints, dataset, subjects_test, action_filter, args.downsample, subset_type='Test')
print('Test, ignore_nose_kpt:{}'.format(ignore_nose_kpt))
if ignore_nose_kpt:
    assert (args.keypoints=='hr'), "This feature has only been tested for data_2d_h36m_hr.npz"
    koi_indexes = list(range(0, 17)) # keypoints-of-interest-indices
    koi_indexes.remove(KPT_2_IDX['Nse']) # remove nose kpt from consideration
    fbones_oi_indices = list(range(0, 16)) # free-bones-of-interest-indices
    if semi_supervised and args.gen_pose_priors>0:
        face_fb_idx = get_id_index(jmc_priors['joint_order'], 'UFace')
    else: face_fb_idx = get_id_index(list(JMC_RMTX_JOINT_CONFIG.keys()), 'UFace')
    fbones_oi_indices.remove(face_fb_idx) # remove UFace free-bone (link to nose kpt) from consideration

filter_widths = [int(x) for x in args.architecture.split(',')]

if not args.disable_optimizations and not args.dense and args.stride == 1:
    # Use optimized model for single-frame predictions
    model_pose_train = TemporalModelOptimized1f(poses_2d_val[0].shape[-2], poses_2d_val[0].shape[-1],
                            dataset.skeleton().num_joints(), filter_widths=filter_widths, causal=args.causal,
                            dropout=args.dropout, channels=args.channels)
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pose_train = TemporalModel(poses_2d_val[0].shape[-2], poses_2d_val[0].shape[-1],
                            dataset.skeleton().num_joints(), filter_widths=filter_widths, causal=args.causal,
                            dropout=args.dropout, channels=args.channels, dense=args.dense)
# Note: model_pose_train is the trained instance, whose weights are loaded into model_pose for evaluation
model_pose = TemporalModel(poses_2d_val[0].shape[-2], poses_2d_val[0].shape[-1], dataset.skeleton().num_joints(),
                          filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                          dense=args.dense)

receptive_field = model_pose.receptive_field()
print('[INFO] Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if args.causal:
    print('[INFO] Using causal convolutions')
    causal_shift = pad
else: causal_shift = 0

model_params = 0
for parameter in model_pose.parameters():
    model_params += parameter.numel()
print('[INFO] Trainable parameter count:', model_params)

if run_on_available_gpu:
    model_pose = model_pose.cuda()
    model_pose_train = model_pose_train.cuda()
    # Edited by Lawrence for multi-gpu
    if training_with_multi_gpu:
        print('[INFO] Using {} GPUs for model_pose*'.format(torch.cuda.device_count()))
        model_pose = nn.DataParallel(model_pose)
        model_pose_train = nn.DataParallel(model_pose_train)

if args.resume or args.evaluate:
    chk_filename = args.resume if args.resume else args.evaluate
    chk_fname_tags = chk_filename.split('_')
    assert (chk_fname_tags[1]==pose2d_tag), '2D pose source {} must match reloaded model {}'.format(args.keypoints, chk_filename)
    assert (chk_fname_tags[2]==arc_tag), 'Set architecture {} must match reloaded model {}'.format(args.architecture, chk_filename)
    chk_filepath = os.path.join(args.checkpoint, chk_filename)
    print('Loading checkpoint', chk_filepath)
    checkpoint = torch.load(chk_filepath, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    pose_net_key = 'model_pos' if checkpoint.get('model_pose', False)==False else 'model_pose'
    model_pose_train.load_state_dict(checkpoint[pose_net_key])
    model_pose.load_state_dict(checkpoint[pose_net_key])

    auxi_net_key = 'model_traj' if checkpoint.get('model_auxi', False)==False else 'model_auxi'
    if checkpoint[auxi_net_key] is not None:
        # Load trajectory/scale model if it is contained in the checkpoint (e.g. for inference in the wild)
        model_auxi = TemporalModel(poses_2d_val[0].shape[-2], poses_2d_val[0].shape[-1], 1,
                            filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)
        if run_on_available_gpu:
            model_auxi = model_auxi.cuda()
        model_auxi.load_state_dict(checkpoint[auxi_net_key])
    else: model_auxi = None

inference_poses_2d = gt_poses_2d_val if args.test_with_2dgt else poses_2d_val
test_gen = UnchunkedGenerator(cams_in_val, poses_3d_val, inference_poses_2d,
                              pad=pad, causal_shift=causal_shift, augment=False,
                              kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('[INFO] Validating on {} frames of {}'.format(test_gen.num_frames(), args.subjects_test))


if not args.evaluate:
    print('3D Supervised Training')
    cams_in_trn, _, poses_3d_trn, gt_poses_2d_trn, poses_2d_trn, ignore_nose_kpt = \
        fetch(keypoints, gt2d_keypoints, dataset, subjects_train, action_filter, args.downsample,
              subset=args.subset, subset_type='3D Supervised Training')

    lr = args.learning_rate
    if args.use_auxiliary_model:
        n_units = 3 if args.projection_type>0 else 1 # trajectory=3, scale=1
        if not args.disable_optimizations and not args.dense and args.stride == 1:
            # Use optimized model for single-frame predictions
            model_auxi_train = TemporalModelOptimized1f(poses_2d_val[0].shape[-2], poses_2d_val[0].shape[-1], 1,
                                                        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                                        channels=args.channels, out_tsr_units=n_units)
        else:
            # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
            model_auxi_train = TemporalModel(poses_2d_val[0].shape[-2], poses_2d_val[0].shape[-1], 1,
                                             filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                             channels=args.channels, dense=args.dense, out_tsr_units=n_units)
        # Note: model_auxi_train is the trained instance, whose weights are loaded into model_auxi for evaluation
        model_auxi = TemporalModel(poses_2d_val[0].shape[-2], poses_2d_val[0].shape[-1], 1,
                                   filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                   channels=args.channels, dense=args.dense, out_tsr_units=n_units)
        if run_on_available_gpu:
            model_auxi = model_auxi.cuda()
            model_auxi_train = model_auxi_train.cuda()
            # For multi-gpu
            if training_with_multi_gpu:
                print('[INFO] Using {} GPUs for model_auxi*'.format(torch.cuda.device_count()))
                model_auxi = nn.DataParallel(model_auxi)
                model_auxi_train = nn.DataParallel(model_auxi_train)

        optimizer = optim.Adam(list(model_pose_train.parameters()) + list(model_auxi_train.parameters()), lr=lr, amsgrad=True)
    else: optimizer = optim.Adam(model_pose_train.parameters(), lr=lr, amsgrad=True)


    if semi_supervised:
        print('Semi-Supervised Training')
        cams_in_semi, cams_ex_semi, poses_3d_semi, gt_poses_2d_semi, poses_2d_semi, ignore_nose_kpt = \
            fetch(keypoints, gt2d_keypoints, dataset, subjects_semi, action_filter, args.downsample,
                  parse_3d_poses=True, subset_type='Semi-Supervised Training') # was False

        losses_r2dp_trn_semi, losses_r2dp_trn_supv_eval, losses_r2dp_trn_semi_eval, losses_r2dp_val = [], [], [], []
        # Weakly supervised losses and regularizers
        losses_mble_trn_semi = []
        losses_mce_wcx_trn_semi, losses_mce_ncx_trn_semi = [], []
        losses_pto_trn_semi, losses_bse_trn_semi = [], []
        losses_bpc_trn_semi, losses_jmc_trn_semi = [], []

    lr_decay = args.lr_decay
    decay_freq = args.decay_frequency
    if args.pose_v_posture_weight_decay<1:
        tug_const = mpjpe_coef.clone()
        tug_decay = torch_t(args.pose_v_posture_weight_decay)
        assert (mpjpe_coef==1.0 and mpboe_coef==0.0)
    else: tug_decay = None

    losses_3d_pose_trn, losses_3d_pose_trn_eval, losses_3d_pose_trn_semi_eval, losses_3d_pose_val = [], [], [], []
    losses_auxi_trn, losses_auxi_trn_eval, losses_auxi_trn_semi_eval, losses_auxi_val = [], [], [], []
    losses_posture_trn, losses_pae_trn = [], []
    losses_mps_ble, losses_pto_trn, losses_poe_trn = [], [], []

    epoch = 0
    n_full_supv_iters = 0
    n_semi_supv_iters = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    in_poses_2d_trn = gt_poses_2d_trn if args.train_with_2dgt else poses_2d_trn
    train_gen = ChunkedGenerator(args.batch_size//args.stride, cams_in_trn, None, poses_3d_trn, in_poses_2d_trn, None, args.stride,
                            pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation, rot_ang=args.rot_augment,
                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_gen_eval = UnchunkedGenerator(cams_in_trn, poses_3d_trn, in_poses_2d_trn, pad=pad, causal_shift=causal_shift, augment=False)
    full_supv_steps_per_epoch = train_gen.steps_per_data_epoch()
    total_iters = full_supv_steps_per_epoch * args.epochs
    print('[INFO] Full supervision on {:,} frames of {:.1%} {} for {} epochs ({:,} total iterations at {:,} steps per'
          ' epoch)\n\twith lr:{}, lr-decay:{} and BatchNorm momentum exponential decayed from {} to {} every {} steps'.
          format(train_gen_eval.num_frames(), args.subset, args.subjects_train, args.epochs, total_iters,
                 full_supv_steps_per_epoch, lr, lr_decay, initial_momentum, final_momentum, decay_freq))
    if semi_supervised:
        ws_poses_2d = gt_poses_2d_semi if args.train_with_2dgt and args.ws_with_2dgt else poses_2d_semi
        semi_gen = ChunkedGenerator(args.ws_batch_size//args.stride, cams_in_semi, cams_ex_semi, None, ws_poses_2d, gt_poses_2d_semi,
                                    args.stride, pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                    rot_ang=args.rot_augment, random_seed=4321, multi_cams=mcv_cnt, mce_flip=wout_cam_ext_mce_type<0,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right, endless=True)
        semi_gen_eval = UnchunkedGenerator(cams_in_semi, poses_3d_semi, ws_poses_2d, pad=pad, causal_shift=causal_shift, augment=False)
        if full_supv_steps_per_epoch>0:
            semi_supv_steps_per_epoch = semi_gen.steps_per_data_epoch()
            full_supv_epochs_per_semi_supv_epoch = semi_supv_steps_per_epoch / full_supv_steps_per_epoch
            semi_supv_total_iters = int(semi_supv_steps_per_epoch * (args.epochs / full_supv_epochs_per_semi_supv_epoch))
            print('[INFO] Semi supervision on {:,} frames of {} with {:,} semi-supervised total iterations at {:,} steps per epoch\n\t'
                  'One semi-supervised epoch is completed after {:.1f} fully-supervised epochs (i.e. semi-supervised data is seen {:.1f} times)'.
                format(semi_gen_eval.num_frames(), args.subjects_unlabeled, semi_supv_total_iters, semi_supv_steps_per_epoch,
                       full_supv_epochs_per_semi_supv_epoch, args.epochs / full_supv_epochs_per_semi_supv_epoch))

    if args.resume:
        if semi_supervised:
            if args.use_auxiliary_model:
                model_auxi_train.load_state_dict(checkpoint['model_auxi'])
                model_auxi.load_state_dict(checkpoint['model_auxi'])
            semi_gen.set_random_state(checkpoint['random_state_semi'])

        if args.reset_optimization_params:
            if args.use_auxiliary_model:
                optimizer = optim.Adam(list(model_pose_train.parameters()) +
                                       list(model_auxi_train.parameters()), lr=lr, amsgrad=True)
            print('[NOTE] Resumed training has been reset to start from '
                  'epoch:{} with lr:{:.5f} and a reinitialized optimizer'.format(epoch, lr))
        else:
            lr = checkpoint['lr']
            epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                train_gen.set_random_state(checkpoint['random_state'])
            else:
                print('[WARNING] This checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

    print('[NOTE] Reported losses are averaged over all batch-&-frame poses without inference-time augmentation'
          '\n\tThe final evaluation will be carried out after the last training epoch.')

    # Extract pose priors for parameterization
    if args.gen_pose_priors<0:
        pose_reg_properties_extractor(train_gen, args, supv_subset_tag, frm_rate_tag)
        # extract_pose_properties_for_build(train_gen, args, supv_subset_tag, frm_rate_tag)
        sys.exit(0)

    last_warmup_epoch = args.warmup
    plot_log_freq = np.floor(args.epochs/5) # plots are logged 5 or 6 times during training
    if args.projection_type==1: projection_func = project_to_2d
    elif args.projection_type==2: projection_func = project_to_2d_linear

    log_epsilon = torch_t(1e+0) # sigm:1e-6
    rootkpt_to_origin_arr = np.ones((1,1,17,1), dtype=np.float32)
    rootkpt_to_origin_arr[0,0,root_kpt_idx,0] = 0.
    rootkpt_to_origin_tsr = torch_t(rootkpt_to_origin_arr)

    def project2d_reconstruction(predicted_3d_pose, predicted_auxi, cam, target, swap_dims=False):
        if pad > 0:
            target = target[:, pad:-pad, :, :2].contiguous()
        else: target = target[:, :, :, :2].contiguous()
        if ignore_nose_kpt: target = target[:,:,koi_indexes] # for HRNet (data_2d_h36m_hr.npz) without nose kpt

        if args.projection_type>0:
            reconstruction = projection_func(predicted_3d_pose + predicted_auxi, cam)
        elif args.projection_type<0:
            if swap_dims:
                #assert(predicted_3d_pose.shape[0]==1), '{}'.format(predicted_3d_pose.shape)
                target = torch.swapaxes(target, 0, 1) # (1,frames,j,3) >> (frames,1,j,3)
                predicted_3d_pose = torch.swapaxes(predicted_3d_pose, 0, 1) # (1,frames,j,3) >> (frames,1,j,3)
                if args.use_auxiliary_model: predicted_auxi = torch.swapaxes(predicted_auxi, 0, 1) # >> (frames,1,1,3)

            if args.projection_type>=-4:
                # scale-down projected-3dto2d-poses
                if not args.use_auxiliary_model:
                    scale_3dto2d = tc_scale_normalize(
                        predicted_3d_pose[:,:,:,:2], target, root_kpt_idx, 0) # (?,f,1,1)
                else: scale_3dto2d = predicted_auxi
                reconstruction = orth_project_and_scaledown_align(predicted_3d_pose, scale_3dto2d,
                                    target[:,:,[root_kpt_idx]], root_kpt_idx, args.projection_type)
            else: # args.projection_type<=-5
                # for when target-2d-poses are scaled-up
                reconstruction = predicted_3d_pose[:,:,:,:2] # (?,f,j,2)
                if args.projection_type==-6: reconstruction = reconstruction - reconstruction[:,:,[root_kpt_idx]]

            if args.projection_type<=-4:
                target = target - target[:,:,[root_kpt_idx]] # move poses to place pelvis at origin
                if args.projection_type<=-5:
                    scale_up_factor = 1/predicted_auxi
                    target = scale_up_factor * target # scale-up target-2d-pose to match predicted 3d-pose
        else:
            reconstruction = target.clone() # no 2d-reprojection, so return target as reconstruction to get an error of 0

        return reconstruction, target

    # Pos model only
    train_t0 = time.time()
    while epoch < args.epochs:
        epoch_t0 = time.time()
        epoch_loss_3d_pose_trn, epoch_loss_auxi_trn, epoch_loss_2d_trn_semi = 0, 0, 0
        epoch_loss_posture_trn, epoch_loss_pae_trn = 0, 0
        epoch_loss_mps_ble, epoch_loss_pto_trn, epoch_loss_poe_trn = 0, 0, 0
        epoch_loss_mble_trn_semi = 0
        epoch_loss_mce_wcx_trn_semi, epoch_loss_mce_ncx_trn_semi = 0, 0
        epoch_loss_pto_trn_semi, epoch_loss_bse_trn_semi = 0, 0
        epoch_loss_bpc_trn_semi, epoch_loss_jmc_trn_semi = 0, 0

        N, N_semi = 0, 0
        N_semi_mv_wcx, N_semi_mv_ncx = 0, 0
        model_pose_train.train()
        if args.use_auxiliary_model: model_auxi_train.train()

        if semi_supervised:
            # Semi-supervised scenario
            # Begin iteration for 1 epoch -----------------------------------------------------------------------------
            for (_, batch_3d, batch_2d), (nonflip_idxs_semi, flipped_idxs_semi, cam_ex_semi, cam_in_semi,
                                          batch_2d_semi, batch_gt2d_semi) \
                    in zip(train_gen.next_epoch(), semi_gen.next_epoch()):

                # Fall back to supervised training for the first few epochs (to avoid instability)
                n_full_supv_iters += 1
                skip = epoch < args.warmup
                cam_ex_semi = torch.from_numpy(cam_ex_semi.astype('float32'))
                cam_in_semi = torch.from_numpy(cam_in_semi.astype('float32'))
                inputs_3d = torch.from_numpy(batch_3d.astype('float32')) # (?,f,j,3)
                if run_on_available_gpu:
                    cam_ex_semi = cam_ex_semi.cuda()
                    cam_in_semi = cam_in_semi.cuda()
                    inputs_3d = inputs_3d.cuda()

                if args.use_auxiliary_model and args.projection_type>0: # for perspective projection
                    inputs_auxi = inputs_3d[:, :, [root_kpt_idx]].clone()
                inputs_3d[:, :, root_kpt_idx] = 0 # Note, affects only 3D supervised branch

                # Split point between labeled and unlabeled samples in the batch
                split_idx = inputs_3d.shape[0]
                inputs_2d = torch.from_numpy(batch_2d.astype('float32')) # (?,rf,j,2)
                inputs_2d_semi = torch.from_numpy(batch_2d_semi.astype('float32')) # (?>,rf,j,2)
                output_2d_semi = torch.from_numpy(batch_gt2d_semi.astype('float32')) # (?>,rf,j,2)
                if run_on_available_gpu:
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_semi = inputs_2d_semi.cuda()
                    output_2d_semi = output_2d_semi.cuda()

                # Source of over-fitting when warmup<epoch and all semi-supervised loss coefficients are 0
                inputs_2d_cat = torch.cat((inputs_2d, inputs_2d_semi), dim=0) if not skip else inputs_2d
                if args.use_auxiliary_model and args.projection_type<0: # for orthographic projection
                    inputs_auxi = tc_scale_normalize(
                        inputs_3d[:,:,:,:2].clone(), inputs_2d, root_kpt_idx, pad).detach() # (?,f,1,1)
                optimizer.zero_grad()

                # Computes Supervised Losses: loss_3d_pose & loss_auxi
                # with top subset of batch corresponding to x% of --subjects-train (eg. 10% of S1)
                # starting with 3D pose loss
                predicted_3d_pose_cat = model_pose_train(inputs_2d_cat)
                n_supv_batch_frames = inputs_3d.shape[0]*inputs_3d.shape[1]
                if ignore_nose_kpt: # for 16 keypoints (15 Bones) HRNet pose
                    loss_3d_pose = mpjpe(predicted_3d_pose_cat[:split_idx,:,koi_indexes], inputs_3d[:,:,koi_indexes])
                else: loss_3d_pose = mpjpe(predicted_3d_pose_cat[:split_idx], inputs_3d)
                epoch_loss_3d_pose_trn += n_supv_batch_frames * loss_3d_pose.item()
                N += n_supv_batch_frames
                loss_total = loss_3d_pose

                # Compute global trajectory/scale
                if args.use_auxiliary_model:
                    predicted_auxi_cat = model_auxi_train(inputs_2d_cat)
                    if args.projection_type>0: # trajectory for perspective projection
                        w = 1 / torch.abs(inputs_auxi[:, :, :, 2]) # Weight inversely proportional to depth
                        loss_auxi = weighted_mpjpe(predicted_auxi_cat[:split_idx], inputs_auxi, w)
                    else: # args.projection_type<0: # scale for orthographic projection
                        # w = 1 / inputs_auxi # Weight inversely proportional to scale
                        loss_auxi = torch.mean(torch.abs(predicted_auxi_cat[:split_idx] - inputs_auxi)) # (?,f,1,1)>>(1,)
                    epoch_loss_auxi_trn += n_supv_batch_frames * loss_auxi.item()
                    #assert(inputs_auxi.shape[0]*inputs_auxi.shape[1] == n_supv_batch_frames)
                    loss_total += loss_auxi

                if not skip: # Semi-Supervised
                    # Semi-supervised loss for unlabeled samples
                    # Computes Self-Supervised Losses: Reprojected 2D MPJPE and 3D MCE, PTO, BSE, BPC, JMC losses
                    # - 1st half of batch corresponding to --subjects-train (labelled subset with 3D gt. Eg. S5,S6)
                    # - 2nd half of batch corresponding to --subjects-unlabeled (subset without 3D gt. Eg. S7,S8)
                    # predicted_3d_pose_cat: (batch, predicted-frames-per-example, kpts, xyz-cord): (128, 1, 17, 3)
                    n_semi_supv_iters += 1
                    #assert(~apply_mv_pose_consistency or len(nonflip_idxs_semi)+len(flipped_idxs_semi)>0)

                    if apply_mv_pose_consistency:
                        # separate true-batch (without added multiview poses) from batch superset
                        semi_pred_3d_pose = predicted_3d_pose_cat[split_idx:]
                        n_semi_samples, n_semi_frames = semi_pred_3d_pose.shape[:2]
                        if args.use_auxiliary_model:
                            semi_pred_auxi = predicted_auxi_cat[split_idx:] # (?",f,1,3)
                    else:
                        semi_pred_3d_pose = predicted_3d_pose_cat[split_idx:] # (?>,f,j,3)
                        n_semi_samples, n_semi_frames = semi_pred_3d_pose.shape[:2]
                        if args.use_auxiliary_model: semi_pred_auxi = predicted_auxi_cat[split_idx:] # (?>,f,1,1)
                        n_semi_mv_wcx_batch_frames = 1 # To avoid division by zero
                        n_semi_mv_ncx_batch_frames = 1 # To avoid division by zero
                    n_semi_batch_frames = n_semi_samples * n_semi_frames

                    # 2D MPJPE loss
                    if args.projection_type!=0:
                        if pad > 0:
                            target_2d_semi = output_2d_semi[:, pad:-pad, :, :2].contiguous() # (?>,f,j,2)
                        else: target_2d_semi = output_2d_semi[:, :, :, :2].contiguous() # (?>,f,j,2)

                        if args.projection_type>0:
                            reconstruction_semi = projection_func(semi_pred_3d_pose+semi_pred_auxi, cam_in_semi) # (?>,f,j,2)
                        else: # args.projection_type<0:
                            if args.projection_type>=-4:
                                # scale-down projected-3dto2d-poses
                                if not args.use_auxiliary_model:
                                    scale_3dto2d = tc_scale_normalize(
                                        semi_pred_3d_pose[:,:,:,:2], target_2d_semi, root_kpt_idx, 0).detach() # (?,f,1,1)
                                else: scale_3dto2d = semi_pred_auxi.detach()
                                reconstruction_semi = orth_project_and_scaledown_align(semi_pred_3d_pose, scale_3dto2d,
                                        target_2d_semi[:,:,[root_kpt_idx]].clone(), root_kpt_idx, args.projection_type)
                            else: # args.projection_type<=-5
                                # for when target-2d-poses are scaled-up
                                reconstruction_semi = semi_pred_3d_pose[:,:,:,:2] # (?,f,j,2)
                                if args.projection_type==-6:
                                    reconstruction_semi = reconstruction_semi - reconstruction_semi[:,:,[root_kpt_idx]]

                            if args.projection_type<=-4: # move poses to place root-kpt (eg. pelvis) at origin
                                target_2d_semi = target_2d_semi - target_2d_semi[:,:,[root_kpt_idx]]
                                if args.projection_type<=-5: # scale-up target-2d-pose to match predicted 3d-pose
                                    scale_up_factor = 1/semi_pred_auxi
                                    target_2d_semi = scale_up_factor * target_2d_semi
                                target_2d_semi = target_2d_semi.detach()

                        if ignore_nose_kpt: # for 16 keypoints (15 Bones) HRNet pose
                            reconstruction_semi = reconstruction_semi[:,:,koi_indexes]
                            target_2d_semi = target_2d_semi[:,:,koi_indexes]
                        loss_reconstruction = mpjpe(reconstruction_semi, target_2d_semi) # On 2D poses todo check
                        epoch_loss_2d_trn_semi += n_semi_batch_frames * loss_reconstruction.item()
                        loss_total += rp2d_coef * loss_reconstruction

                    # Bone Length Error (BLE) term to enforce kinematic constraints
                    if args.mean_bone_length_term:
                        dists = predicted_3d_pose_cat[:, :, 1:] - \
                                predicted_3d_pose_cat[:, :, dataset.skeleton().parents()[1:]] # -->(?,f,b,3)
                        avg_frm_bone_lens = torch.mean(torch.linalg.norm(dists, dim=3), dim=1) # (?,f,b,3)->(?,f,b)->(?,b)
                        penalty = torch.mean(torch.abs(torch.mean(avg_frm_bone_lens[:split_idx], dim=0) -
                                                       torch.mean(avg_frm_bone_lens[split_idx:], dim=0))) # (?,b)->(b,)->(1,)
                        epoch_loss_mble_trn_semi += n_semi_batch_frames * penalty.item()
                        loss_total += mble_coef * penalty

                    # 3D Pose Pelvis-to-Origin (PTO) loss term
                    if args.pelvis_placement_term: # todo: rename to rootkpt_at_origin_term (RTO)
                        root_kpt_dto = -semi_pred_3d_pose[:,:,root_kpt_idx] # plv distance to origin
                        loss_pto = torch.mean(torch.linalg.norm(root_kpt_dto, dim=-1))
                        epoch_loss_pto_trn_semi += n_semi_batch_frames * loss_pto.item()
                        loss_total += pto_coef * loss_pto

                    if args.reposition_to_origin:
                        # re-position pelvis to origin, do so after computing PTO and before computing BPC & JMC
                        semi_pred_3d_pose = semi_pred_3d_pose * rootkpt_to_origin_tsr # 0 at root-kpt index, 1 at other keypoint indexes

                    if args.bone_symmetry_term or args.bone_proportion_term:
                        semi_dists = semi_pred_3d_pose[:,:,BONE_CHILD_KPTS_IDXS] - semi_pred_3d_pose[:,:,BONE_PARENT_KPTS_IDXS] # -->(?>,f,b,3)
                        semi_bone_lengths = torch.linalg.norm(semi_dists, dim=3)  # (?>,f,b,3) --> (?>,f,b)
                        #assert(torch.all(semi_bone_lengths>=0.)), "shouldn't be negative {}".format(semi_bone_lengths)

                    # Multi-view (pose & trajectory) Consistency Error (MCE) loss
                    if apply_mv_pose_consistency:
                        #assert(((len(nonflip_idxs_semi)+len(flipped_idxs_semi))//mcv_cnt)>0), 'apply_mv_pose_consistency --> >0'

                        # Pose-MCE variant >> with camera extrinsic
                        if with_cam_ext_mce_type>0 and len(nonflip_idxs_semi)>0:
                            wcx_mv_semi_pred_3d_pose = semi_pred_3d_pose[nonflip_idxs_semi] # (?'*,f,j,3) with-cam-ext
                            # Transform poses to a common frame (aka world-frame)
                            wcx_mv_semi_pred_auxi = semi_pred_auxi[nonflip_idxs_semi] # (?'*,f,1,3)
                            wcx_mv_cam_ex_semi = torch.reshape(cam_ex_semi[nonflip_idxs_semi], (-1,1,1,7)) # (?'*,7)->(?'*,1,1,7)
                            wcx_mv_cam_ex_semi = torch.tile(wcx_mv_cam_ex_semi, (1,1,17,1)) # (?'*,1,17,7)
                            camfrm_wcx_mv_poses = wcx_mv_semi_pred_3d_pose + wcx_mv_semi_pred_auxi # (?'*,f,j,3)
                            wldfrm_wcx_mv_poses = qrot(wcx_mv_cam_ex_semi[:,:,:,:4], camfrm_wcx_mv_poses, debug=db_mode) \
                                                  + wcx_mv_cam_ex_semi[:,:,:,4:] # (?'*,f,j,3)
                            wldfrm_wcx_mv_poses = wldfrm_wcx_mv_poses.view(-1,mcv_cnt,n_semi_frames,17,3) # (?*,m,f,j,3)

                            # Translate poses to origin to reduce sensitivity ot trajectory model
                            if not args.enforce_mce_trajectory_alignment:
                                wldfrm_wcx_mv_poses -= wldfrm_wcx_mv_poses[:,:,:,[root_kpt_idx]] # (?*,m,f,j,3)

                            if with_cam_ext_mce_type==1:
                                # Anchor-to-Each-Positive: anchor pose is compared to each positive pose
                                each_ach_2_pose_errs = []
                                for view_idx in range(1, mcv_cnt):
                                    each_ach_2_pose_errs.append(torch.linalg.norm(
                                        wldfrm_wcx_mv_poses[:,0] - wldfrm_wcx_mv_poses[:,view_idx], dim=-1)) # (?*,f,j)
                                agg_kpt_wcx_mce_per_pose = torch.stack(each_ach_2_pose_errs, dim=3) # (m-1)*(?*,f,j)->(?*,f,j,m-1)

                            elif with_cam_ext_mce_type==2:
                                # Adjacent-Pairs-Comparison: pairs of adjacent multi-view poses are compared
                                each_adj_pair_errs = []
                                for view_idx in range(mcv_cnt-1):
                                    each_adj_pair_errs.append(torch.linalg.norm(
                                        wldfrm_wcx_mv_poses[:,view_idx] - wldfrm_wcx_mv_poses[:,view_idx+1], dim=-1)) # (?*,f,j)
                                agg_kpt_wcx_mce_per_pose = torch.stack(each_adj_pair_errs, dim=3) # (m-1)*(?*,f,j)->(?*,f,j,m-1)

                            n_semi_mv_wcx_batch_frames = agg_kpt_wcx_mce_per_pose.shape[0] * n_semi_frames
                            if ignore_nose_kpt: # for 16 keypoints (15 Bones) HRNet pose
                                loss_mce_wcx = torch.mean(agg_kpt_wcx_mce_per_pose[:,:,koi_indexes])
                            else: loss_mce_wcx = torch.mean(agg_kpt_wcx_mce_per_pose)
                            epoch_loss_mce_wcx_trn_semi += n_semi_mv_wcx_batch_frames * loss_mce_wcx.item()
                            loss_total += mce_wcx_coef * loss_mce_wcx
                        else: n_semi_mv_wcx_batch_frames = 1 # To avoid division by zero

                        # Posture-MCE variant >> without camera extrinsic
                        if with_cam_ext_mce_type>0: wout_cam_ext_indices = flipped_idxs_semi
                        else: wout_cam_ext_indices = nonflip_idxs_semi + flipped_idxs_semi
                        if wout_cam_ext_mce_type<0 and len(wout_cam_ext_indices)>0:
                            nex_mv_semi_pred_3d_pose = semi_pred_3d_pose[wout_cam_ext_indices] # (?'^,f,j,3)
                            nex_mv_semi_pred_qset_kpts = nex_mv_semi_pred_3d_pose[:,:,mce_qset_kpt_idxs,:] # (?'^,f,b,q,3)

                            if wout_cam_ext_mce_type>=-3: # i.e. -1, -2 or -3
                                # Compare aligned free-bone vectors
                                # Anchor-to-Each-Positive: anchor pose is compared to each positive pose
                                nex_mv_free_bone_vecs = mce_fb_orient(nex_mv_semi_pred_qset_kpts) # vec/uvec: (?'^,f,b,3)
                                nex_mv_free_bone_vecs = nex_mv_free_bone_vecs.view(-1,mcv_cnt,n_semi_frames,16,3) # (?^,m,f,b,3)

                                # compute supervised target poses' average bone lengths
                                supv_targ_dist = inputs_3d[:,:,BONE_CHILD_KPTS_IDXS] - inputs_3d[:,:,BONE_PARENT_KPTS_IDXS] # (?,f,b,3)
                                supv_targ_bone_len = torch.linalg.norm(supv_targ_dist, dim=3, keepdim=True) # (?,f,b,1)
                                supv_targ_bone_len = supv_targ_bone_len[:,:,mce_fbo_v2o_bone_idxs] # (?,f,b,1)
                                supv_targ_per_bone_avg_len = torch.mean(supv_targ_bone_len, dim=(0,1), keepdim=True) # (1,1,b,1)

                                each_ach_2_pose_errs = []
                                for view_idx in range(1, mcv_cnt):
                                    each_ach_2_pose_errs.append(torch.linalg.norm(
                                        nex_mv_free_bone_vecs[:,0]*supv_targ_per_bone_avg_len - # was ord=None # (?^,f,b)
                                        nex_mv_free_bone_vecs[:,view_idx]*supv_targ_per_bone_avg_len, dim=-1, ord=args.posture_norm))

                                agg_kpt_ncx_mce_per_pose = torch.stack(each_ach_2_pose_errs, dim=3) # (m-1)*(?^,f,b)->(?^,f,b,m-1)
                                if ignore_nose_kpt: # for 16 keypoints (15 Bones) HRNet pose
                                    agg_kpt_ncx_mce_per_pose = agg_kpt_ncx_mce_per_pose[:,:,fbones_oi_indices]

                                if wout_cam_ext_mce_type==-1: # FreeBone Vectors
                                    loss_mce_ncx = torch.mean(agg_kpt_ncx_mce_per_pose) # fb vecs in meters

                                elif wout_cam_ext_mce_type<=-2: # FreeBone Unit-Vectors
                                    loss_mce_ncx = torch.mean(agg_kpt_ncx_mce_per_pose)
                                    if wout_cam_ext_mce_type==-3:
                                        # also compare bone lengths of multi-view poses
                                        nex_mv_bone_dists = nex_mv_semi_pred_3d_pose[:,:,1:] - \
                                                            nex_mv_semi_pred_3d_pose[:,:,dataset.skeleton().parents()[1:]] # (?'^,f,b,3)
                                        nex_mv_bone_lens = torch.linalg.norm(nex_mv_bone_dists, dim=3)  # (?'^,f,b,3)->(?'^,f,b)
                                        per_ncx_mv_bone_lens = nex_mv_bone_lens.view(-1,mcv_cnt,n_semi_frames,16) # (?^,m,f,b)
                                        each_ach_2_pose_dist = []
                                        for view_idx in range(1, mcv_cnt):
                                            each_ach_2_pose_dist.append(torch.abs(
                                                per_ncx_mv_bone_lens[:,0] - per_ncx_mv_bone_lens[:,view_idx])) # (?^,f,b)
                                        agg_kpt_blen_dist_err = torch.stack(each_ach_2_pose_dist, dim=3) # (m-1)*(?^,f,b)->(?^,f,b,m-1)
                                        loss_mce_ncx += torch.mean(agg_kpt_blen_dist_err)

                            elif wout_cam_ext_mce_type==-4:
                                # MPCE from Procrustes Rigid Aligned Pose
                                plvfrm_ncx_mv_poses = nex_mv_semi_pred_3d_pose.view(-1,mcv_cnt,n_semi_frames,17, 3) # (?^,m,f,j,3)
                                each_ach_2_pose_errs = []
                                for view_idx in range(mcv_cnt-1):
                                    each_ach_2_pose_errs.append(torch_p_mpjpe(
                                        plvfrm_ncx_mv_poses[0, view_idx].view(-1,17,3),
                                        plvfrm_ncx_mv_poses[0, view_idx+1].view(-1,17,3)))
                                agg_kpt_ncx_mce_per_pose = torch.stack(each_ach_2_pose_errs, dim=2) # (m-1)*(?^,j) -> (?^,j,m-1)
                                if ignore_nose_kpt: # for 16 keypoints (15 Bones) HRNet pose
                                    loss_mce_ncx = torch.mean(agg_kpt_ncx_mce_per_pose[:,koi_indexes])
                                else: loss_mce_ncx = torch.mean(agg_kpt_ncx_mce_per_pose)

                            n_semi_mv_ncx_batch_frames = agg_kpt_ncx_mce_per_pose.shape[0] * n_semi_frames
                            epoch_loss_mce_ncx_trn_semi += n_semi_mv_ncx_batch_frames * loss_mce_ncx.item()
                            loss_total += mce_ncx_coef * loss_mce_ncx
                        else: n_semi_mv_ncx_batch_frames = 1 # To avoid division by zero

                    # 3D Bone Symmetry Error (BSE) Loss
                    if args.bone_symmetry_term:
                        semi_rgt_sym_bone_lengths = semi_bone_lengths[:, :, RGT_SYM_BONE_INDEXES] # (?>,f,6)
                        semi_lft_sym_bone_lengths = semi_bone_lengths[:, :, LFT_SYM_BONE_INDEXES] # (?>,f,6)
                        semi_sym_bones_difference = semi_rgt_sym_bone_lengths - semi_lft_sym_bone_lengths  # (?>,f,6)
                        loss_bse = torch.mean(torch.abs(semi_sym_bones_difference))  # (?>,f,6)->(1,)
                        epoch_loss_bse_trn_semi += n_semi_batch_frames * loss_bse.item()
                        loss_total += bse_coef * loss_bse

                    # 3D Bone Proportion Constraint (BPC) Regularizer
                    if args.bone_proportion_term:
                        semi_ratio_prop = guard_div1(semi_bone_lengths[:,:,bone_prop_pair_idxs_A],
                                                    semi_bone_lengths[:,:,bone_prop_pair_idxs_B]) # (?>,f,15)
                        bpc_likelihood = bpc_likelihood_func(semi_ratio_prop, bpc_variance, bpc_exponent_coefs, bpc_mean)
                        bpc_actv_likelihood = -bpc_loglihood_wgts * torch.log(bpc_likelihood + log_epsilon)
                        loss_bpc = torch.mean(bpc_actv_likelihood)
                        epoch_loss_bpc_trn_semi += n_semi_batch_frames * loss_bpc.item()
                        loss_total += bpc_coef * loss_bpc

                    # 3D Joint Mobility Constraint (JMC) or Bone Orientation Constraint Regularizer
                    if args.joint_mobility_term:
                        # Adjust pose for JMC
                        semi_pred_qset_kpts = semi_pred_3d_pose[:,:,jmc_qset_kpt_idxs,:] # (?>,f,j,3)-->(?>,f,b,q,3)
                        semi_pred_fb_uvecs = jmc_fb_orient(semi_pred_qset_kpts) # (?>,f,b,3)

                        jmc_likelihood_list = []
                        for r_idx in range(len(jmc_params)):
                            k_jnts_indexes, uvec_mean_kx1, inv_covariance_kxk, k_exponent_coefs, \
                            k_logli_spread, k_logli_mean, k_logli_std, k_logli_min, k_logli_span, \
                            k_logli_wgt, k_log_of_spread, k_move_up_const = jmc_params[r_idx]
                            fb_uvecs_list = []
                            for jnt_idxs in k_jnts_indexes:
                                fb_uvecs_list.append(semi_pred_fb_uvecs[:,:,jnt_idxs,:]) # (?,f,n,3) xyz
                            fb_uvecs = torch.cat(fb_uvecs_list, dim=-1) # r*(?,f,n,3)-->(?,f,n,3*r)
                            k_likelihood = jmc_likelihood_func(fb_uvecs, uvec_mean_kx1, inv_covariance_kxk, k_exponent_coefs) # (?,f,n)
                            jmc_likelihood_list.append(k_likelihood)

                        jmc_likelihood = torch.cat(jmc_likelihood_list, dim=-1) # (?,f,n)+(?,f,m)+..-->(?,f,n+m+..)
                        jmc_actv_likelihood = -jmc_loglihood_wgts * torch.log(jmc_likelihood + log_epsilon)
                        loss_jmc = torch.mean(jmc_actv_likelihood)
                        epoch_loss_jmc_trn_semi += n_semi_batch_frames * loss_jmc.item()
                        loss_total += jmc_coef * loss_jmc

                    N_semi += n_semi_batch_frames
                    N_semi_mv_wcx += n_semi_mv_wcx_batch_frames
                    N_semi_mv_ncx += n_semi_mv_ncx_batch_frames

                else:
                    N_semi += 1 # To avoid division by zero
                    N_semi_mv_wcx += 1 # To avoid division by zero
                    N_semi_mv_ncx += 1 # To avoid division by zero

                loss_total.backward()
                optimizer.step()

                # Decay hyper-parameters
                if n_full_supv_iters % decay_freq == 0:
                    # Decay learning rate exponentially
                    lr *= lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_decay
                    # Decay BatchNorm momentum (Edited by Lawrence 03/25/2021)
                    momentum = initial_momentum * np.exp(-n_full_supv_iters/total_iters * np.log(initial_momentum/final_momentum))
                    if training_with_multi_gpu:
                        model_pose_train.module.set_bn_momentum(momentum)
                    else: model_pose_train.set_bn_momentum(momentum)
                    if args.use_auxiliary_model:
                        if training_with_multi_gpu:
                            model_auxi_train.module.set_bn_momentum(momentum)
                        else: model_auxi_train.set_bn_momentum(momentum)
            # End iteration for 1 epoch -------------------------------------------------------------------------------

            # log mean of certain metrics wrt. all batches & frames per epoch
            losses_auxi_trn.append(epoch_loss_auxi_trn / N)
            losses_r2dp_trn_semi.append(epoch_loss_2d_trn_semi / N_semi)
            losses_mble_trn_semi.append(epoch_loss_mble_trn_semi / N_semi)
            losses_mce_wcx_trn_semi.append(epoch_loss_mce_wcx_trn_semi / N_semi_mv_wcx)
            losses_mce_ncx_trn_semi.append(epoch_loss_mce_ncx_trn_semi / N_semi_mv_ncx)
            losses_pto_trn_semi.append(epoch_loss_pto_trn_semi / N_semi)
            losses_bse_trn_semi.append(epoch_loss_bse_trn_semi / N_semi)
            losses_bpc_trn_semi.append(epoch_loss_bpc_trn_semi / N_semi)
            losses_jmc_trn_semi.append(epoch_loss_jmc_trn_semi / N_semi)

        else: # Fully Supervised
            # Regular supervised scenario
            for _, batch_3d, batch_2d in train_gen.next_epoch():
                n_full_supv_iters += 1
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if run_on_available_gpu:
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                if args.use_auxiliary_model: # for perspective projection
                    inputs_traj = inputs_3d[:, :, [root_kpt_idx]].clone()
                inputs_3d[:, :, root_kpt_idx] = 0

                if optimize_posture_loss:
                    # extract target poses' free-bone unit vectors
                    targ_fb_uvecs = targ_pose_fbo(inputs_3d[:,:,pose_qset_kpt_idxs,:]).detach()#.clone() # (?,f,b,3)
                    if scale_by_targ_blen: # extract target poses' bone lengths
                        targ_dist = inputs_3d[:,:,BONE_CHILD_KPTS_IDXS] - inputs_3d[:,:,BONE_PARENT_KPTS_IDXS] # (?,f,b,3)
                        targ_bone_len = torch.linalg.norm(targ_dist, dim=3, keepdim=True).detach()#.clone() # (?,f,b,1)
                        targ_bone_len = targ_bone_len[:,:,pose_fbo_v2o_bone_idxs] # (?,f,b,1) to scale fb-uvec by gt bone len

                optimizer.zero_grad()

                # Predict 3D poses
                predicted_3d_pose = model_pose_train(inputs_2d) # (?,f,j,3)
                n_supv_batch_frames = inputs_3d.shape[0]*inputs_3d.shape[1]
                loss_3d_pose = mpjpe(predicted_3d_pose, inputs_3d)
                epoch_loss_3d_pose_trn += n_supv_batch_frames * loss_3d_pose.item()
                loss_total = mpjpe_coef * loss_3d_pose

                # Compute global trajectory/scale
                if args.use_auxiliary_model:
                    predicted_traj = model_auxi_train(inputs_2d)
                    w = 1 / inputs_traj[:, :, :, 2] # Weight inversely proportional to depth
                    loss_auxi = weighted_mpjpe(predicted_traj, inputs_traj, w)
                    epoch_loss_auxi_trn += n_supv_batch_frames * loss_auxi.item()
                    loss_total += loss_auxi

                # compute posture anchor error as base keypoints' mpjpe to orient posture
                if args.alternating_loss_epochs>0:
                    loss_pae = mpjpe(predicted_3d_pose[:,:,[0,1,4,7]], inputs_3d[:,:,[0,1,4,7]]) # (?,f,4,3)
                    epoch_loss_pae_trn += n_supv_batch_frames * loss_pae.item()
                    loss_total += mpboe_coef * loss_pae

                # compute poses' MPBOE from free bone unit-vector orientation
                if optimize_posture_loss:
                    pred_fb_uvecs = pred_pose_fbo(predicted_3d_pose[:,:,pose_qset_kpt_idxs,:]) # (?,f,b,3)
                    if posture_loss_lnorm:
                        if scale_by_targ_blen: # scale free-bone unit vectors to target bone lengths
                            pred_fb_uvecs = targ_bone_len * pred_fb_uvecs
                            targ_fb_uvecs = targ_bone_len * targ_fb_uvecs
                        fb_uvecs_dist = torch.linalg.norm(
                            pred_fb_uvecs - targ_fb_uvecs, dim=-1, ord=args.posture_norm) # (?,f,b) was ord=None
                    else: # posture_loss_cosine
                        cosine_btw_vecs = torch_vecdot(pred_fb_uvecs, targ_fb_uvecs, keepdim=False) # (?,f,b)
                        if scale_by_targ_blen:
                            targ_bone_len_wgt = targ_bone_len[:,:,:,0] * 10 # decimeter
                            fb_uvecs_dist = targ_bone_len_wgt * ((1. - cosine_btw_vecs) / 2)
                        else: fb_uvecs_dist = 1. - cosine_btw_vecs # todo: map to [0,1] >> ((1. - cosine_btw_vecs) / 2)
                    loss_posture = torch.mean(fb_uvecs_dist)
                    epoch_loss_posture_trn += n_supv_batch_frames * loss_posture.item()
                    loss_total += mpboe_coef * loss_posture

                N += n_supv_batch_frames

                loss_total.backward()
                optimizer.step()

                # Decay hyper-parameters
                if n_full_supv_iters % decay_freq == 0:
                    # Decay learning rate exponentially
                    lr *= lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_decay
                    # Decay BatchNorm momentum (Edited by Lawrence 03/25/2021)
                    momentum = initial_momentum * np.exp(-n_full_supv_iters/total_iters * np.log(initial_momentum/final_momentum))
                    if training_with_multi_gpu:
                        model_pose_train.module.set_bn_momentum(momentum)
                    else: model_pose_train.set_bn_momentum(momentum)
                    if args.use_auxiliary_model:
                        if training_with_multi_gpu:
                            model_auxi_train.module.set_bn_momentum(momentum)
                        else: model_auxi_train.set_bn_momentum(momentum)
                    # Decay Fully-supervised MPJPE vs. MPBOE loss weight tug control constant
                    if tug_decay is not None:
                        tug_const *= tug_decay # decayed from 1 to 0
                        if args.decay_mpjpe: mpjpe_coef = tug_const.clone()
                        mpboe_coef = torch_t(1.0) - tug_const

        losses_3d_pose_trn.append(epoch_loss_3d_pose_trn / N)
        losses_posture_trn.append(epoch_loss_posture_trn / N)
        losses_pae_trn.append(epoch_loss_pae_trn / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pose.load_state_dict(model_pose_train.state_dict())
            model_pose.eval()
            if args.use_auxiliary_model:
                model_auxi.load_state_dict(model_auxi_train.state_dict())
                model_auxi.eval()

            epoch_loss_3d_pose_val, epoch_loss_auxi_val, epoch_loss_2d_val = 0, 0, 0
            N = 0
            if not args.no_eval:
                # Evaluate on test set
                for cam, batch_3d, batch_2d in test_gen.next_epoch():
                    inputs_3d = torch.from_numpy(batch_3d.astype('float32')) # (?,f,j,3) where ?=1 or 2, f>1 (# of session frames)
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32')) # (?,f,j,2) where ?=1 or 2, f>1 (# of session frames)
                    if run_on_available_gpu:
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    if args.use_auxiliary_model and args.projection_type>0: # for perspective projection
                        inputs_auxi = inputs_3d[:, :, [root_kpt_idx]].clone()
                    inputs_3d[:, :, root_kpt_idx] = 0
                    if args.use_auxiliary_model and args.projection_type<0: # for orthographic projection
                        inputs_auxi = tc_scale_normalize(inputs_3d[:,:,:,:2].clone(), inputs_2d, root_kpt_idx, pad) # (?,f,1,1)

                    # Estimate 3D poses and compute 3D pose loss
                    predicted_3d_pose = model_pose(inputs_2d)
                    if ignore_nose_kpt:
                        # activated for HRNet (data_2d_h36m_hr.npz) without nose kpt
                        inputs_3d = inputs_3d[:,:,koi_indexes,:]
                        predicted_3d_pose = predicted_3d_pose[:,:,koi_indexes,:]
                    loss_3d_pose = mpjpe(predicted_3d_pose, inputs_3d)
                    n_test_batch_frames = inputs_3d.shape[0]*inputs_3d.shape[1]
                    epoch_loss_3d_pose_val += n_test_batch_frames * loss_3d_pose.item()
                    N += n_test_batch_frames

                    if args.use_auxiliary_model:
                        predicted_auxi = model_auxi(inputs_2d)
                        if args.projection_type>0: # trajectory for perspective projection
                            loss_auxi = mpjpe(predicted_auxi, inputs_auxi)
                        else: # args.projection_type<0: # scale for orthographic projection
                            loss_auxi = torch.mean(torch.abs(predicted_auxi - inputs_auxi)) # (?,f,1,1)>>(1,)
                        epoch_loss_auxi_val += n_test_batch_frames * loss_auxi.item()
                        #assert(inputs_auxi.shape[0]*inputs_auxi.shape[1] == n_test_batch_frames)
                    else: predicted_auxi = None

                    if semi_supervised:
                        cam = torch.from_numpy(cam.astype('float32'))
                        if run_on_available_gpu: cam = cam.cuda()

                        reconstruction, target = project2d_reconstruction(
                            predicted_3d_pose, predicted_auxi, cam, inputs_2d, swap_dims=True)
                        loss_reconstruction = mpjpe(reconstruction, target) # On 2D poses
                        epoch_loss_2d_val += n_test_batch_frames * loss_reconstruction.item()
                        #assert(reconstruction.shape[0]*reconstruction.shape[1] == n_test_batch_frames)

                losses_3d_pose_val.append(epoch_loss_3d_pose_val / N)
                if args.use_auxiliary_model: losses_auxi_val.append(epoch_loss_auxi_val / N)
                if semi_supervised: losses_r2dp_val.append(epoch_loss_2d_val / N)

                # Evaluate on full-supervised subset of training set, this time in evaluation mode
                epoch_loss_3d_pose_trn_eval, epoch_loss_auxi_trn_eval, epoch_loss_2d_trn_supv_eval = 0, 0, 0
                N = 0
                for cam, batch_3d, batch_2d in train_gen_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if run_on_available_gpu:
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    if args.use_auxiliary_model and args.projection_type>0: # for perspective projection
                        inputs_auxi = inputs_3d[:, :, [root_kpt_idx]].clone()
                    inputs_3d[:, :, root_kpt_idx] = 0
                    if args.use_auxiliary_model and args.projection_type<0: # for orthographic projection
                        inputs_auxi = tc_scale_normalize(inputs_3d[:,:,:,:2].clone(), inputs_2d, root_kpt_idx, pad) # (?,f,1,1)

                    # Estimate 3D poses and compute 3D pose loss
                    predicted_3d_pose = model_pose(inputs_2d)
                    if ignore_nose_kpt:
                        # activated for HRNet (data_2d_h36m_hr.npz) without nose kpt
                        inputs_3d = inputs_3d[:,:,koi_indexes,:]
                        predicted_3d_pose = predicted_3d_pose[:,:,koi_indexes,:]
                    loss_3d_pose = mpjpe(predicted_3d_pose, inputs_3d)
                    n_eval_batch_frames = inputs_3d.shape[0]*inputs_3d.shape[1]
                    epoch_loss_3d_pose_trn_eval += n_eval_batch_frames * loss_3d_pose.item()
                    N += n_eval_batch_frames

                    if args.use_auxiliary_model:
                        predicted_auxi = model_auxi(inputs_2d)
                        if args.projection_type>0: # trajectory for perspective projection
                            loss_auxi = mpjpe(predicted_auxi, inputs_auxi)
                        else: # args.projection_type<0: # scale for orthographic projection
                            loss_auxi = torch.mean(torch.abs(predicted_auxi - inputs_auxi)) # (?,f,1,1)>>(1,)
                        epoch_loss_auxi_trn_eval += n_eval_batch_frames * loss_auxi.item()
                        #assert(inputs_auxi.shape[0]*inputs_auxi.shape[1] == n_eval_batch_frames)
                    else: predicted_auxi = None

                    if semi_supervised:
                        cam = torch.from_numpy(cam.astype('float32'))
                        if run_on_available_gpu: cam = cam.cuda()

                        reconstruction, target = project2d_reconstruction(
                            predicted_3d_pose, predicted_auxi, cam, inputs_2d, swap_dims=True)
                        loss_reconstruction = mpjpe(reconstruction, target)
                        epoch_loss_2d_trn_supv_eval += n_eval_batch_frames * loss_reconstruction.item()
                        #assert(reconstruction.shape[0]*reconstruction.shape[1] == n_eval_batch_frames)

                losses_3d_pose_trn_eval.append(epoch_loss_3d_pose_trn_eval / N)
                if args.use_auxiliary_model: losses_auxi_trn_eval.append(epoch_loss_auxi_trn_eval / N)
                if semi_supervised: losses_r2dp_trn_supv_eval.append(epoch_loss_2d_trn_supv_eval / N)

                # Evaluate semi-supervised subset and 2D loss on unlabeled training set (in evaluation mode)
                epoch_loss_3d_pose_trn_semi_eval, epoch_loss_auxi_trn_semi_eval, epoch_loss_2d_trn_semi_eval = 0, 0, 0
                N_semi = 0
                if semi_supervised:
                    for cam, batch_3d, batch_2d in semi_gen_eval.next_epoch():
                        cam_semi = torch.from_numpy(cam.astype('float32'))
                        inputs_3d_semi = torch.from_numpy(batch_3d.astype('float32'))
                        inputs_2d_semi = torch.from_numpy(batch_2d.astype('float32'))
                        if run_on_available_gpu:
                            cam_semi = cam_semi.cuda()
                            inputs_3d_semi = inputs_3d_semi.cuda()
                            inputs_2d_semi = inputs_2d_semi.cuda()
                        if args.use_auxiliary_model and args.projection_type>0: # for perspective projection
                            inputs_auxi_semi = inputs_3d_semi[:, :, [root_kpt_idx]].clone()
                        inputs_3d_semi[:, :, root_kpt_idx] = 0
                        if args.use_auxiliary_model and args.projection_type<0: # for orthographic projection
                            inputs_auxi_semi = tc_scale_normalize(inputs_3d_semi[:,:,:,:2].clone(), inputs_2d_semi, root_kpt_idx, pad) # (?,f,1,1)

                        # Estimate 3D poses and compute 3D pose loss
                        predicted_3d_pose_semi = model_pose(inputs_2d_semi)
                        if ignore_nose_kpt:
                            # activated for HRNet (data_2d_h36m_hr.npz) without nose kpt
                            inputs_3d_semi = inputs_3d_semi[:,:,koi_indexes,:]
                            predicted_3d_pose_semi = predicted_3d_pose_semi[:,:,koi_indexes,:]
                        loss_3d_pose_semi = mpjpe(predicted_3d_pose_semi, inputs_3d_semi)
                        n_eval_batch_frames_semi = inputs_3d_semi.shape[0]*inputs_3d_semi.shape[1]
                        epoch_loss_3d_pose_trn_semi_eval += n_eval_batch_frames_semi * loss_3d_pose_semi.item()
                        N_semi += n_eval_batch_frames_semi

                        if args.use_auxiliary_model:
                            predicted_auxi_semi = model_auxi(inputs_2d_semi)
                            if args.projection_type>0: # trajectory for perspective projection
                                loss_auxi_semi = mpjpe(predicted_auxi_semi, inputs_auxi_semi)
                            else: # args.projection_type<0: # scale for orthographic projection
                                loss_auxi_semi = torch.mean(torch.abs(predicted_auxi_semi - inputs_auxi_semi)) # (?,f,1,1)>>(1,)
                            epoch_loss_auxi_trn_semi_eval += n_eval_batch_frames_semi * loss_auxi_semi.item()
                            #assert(inputs_auxi_semi.shape[0]*inputs_auxi_semi.shape[1] == n_eval_batch_frames_semi)
                        else: predicted_auxi_semi = None

                        reconstruction_semi, target_2d_semi = project2d_reconstruction(
                            predicted_3d_pose_semi, predicted_auxi_semi, cam_semi, inputs_2d_semi, swap_dims=True)
                        loss_reconstruction_semi = mpjpe(reconstruction_semi, target_2d_semi)
                        epoch_loss_2d_trn_semi_eval += n_eval_batch_frames_semi * loss_reconstruction_semi.item()
                        #assert(reconstruction_semi.shape[0]*reconstruction_semi.shape[1] == n_eval_batch_frames_semi)

                    losses_3d_pose_trn_semi_eval.append(epoch_loss_3d_pose_trn_semi_eval / N_semi)
                    losses_auxi_trn_semi_eval.append(epoch_loss_auxi_trn_semi_eval / N_semi)
                    losses_r2dp_trn_semi_eval.append(epoch_loss_2d_trn_semi_eval / N_semi)

        elapsed = (time.time() - epoch_t0)/60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (epoch+1, elapsed, lr, losses_3d_pose_trn[-1]*1000))
        else:
            if semi_supervised:
                if epoch%50==0: print('{:-^170}\n{}\n{:-^170}'.format('', r_header, ''))
                print(('[{:>3}] {:4.1f} {:>6} {:>6} {:.2e} | {:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} '
                      '{:>6.4f} {:>6.3f} {:>6.3f} | {:>6.4f} {:>6.4f} {:>6.4f}'+auxi_print_fmt+'{:>6.1f} {:>6.1f} {:>6.1f}').
                      format(epoch+1, elapsed, n_full_supv_iters, n_semi_supv_iters, lr,
                             # losses
                             losses_3d_pose_trn[-1], losses_auxi_trn[-1],
                             losses_mce_wcx_trn_semi[-1], losses_mce_ncx_trn_semi[-1],
                             losses_r2dp_trn_semi[-1], losses_mble_trn_semi[-1],
                             losses_pto_trn_semi[-1], losses_bse_trn_semi[-1],
                             losses_bpc_trn_semi[-1], losses_jmc_trn_semi[-1],
                             # evaluations
                             losses_r2dp_trn_supv_eval[-1], losses_r2dp_trn_semi_eval[-1], losses_r2dp_val[-1],
                             losses_auxi_trn_eval[-1]*auxi_s, losses_auxi_trn_semi_eval[-1]*auxi_s, losses_auxi_val[-1]*auxi_s,
                             losses_3d_pose_trn_eval[-1]*1000, losses_3d_pose_trn_semi_eval[-1]*1000, losses_3d_pose_val[-1]*1000))
            else:
                if epoch%50==0: print('{:-^97}\n{}\n{:-^97}'.format('', r_header, ''))
                print('[{:>3}] {:4.1f} {:>6} {:.2e} | {:>6.4f} {:>6.4f} | {:>6.4f} {:>6.4f} {:>6.4f} |'
                      ' {:>6.1f} {:>6.1f} | {:>6.1f} {:>6.1f}'
                    .format(epoch+1, elapsed, n_full_supv_iters, lr,
                            # 3d pose & posture coefficients
                            mpjpe_coef, mpboe_coef,
                            # 3d pose and posture loss
                            losses_3d_pose_trn[-1], losses_pae_trn[-1], losses_posture_trn[-1],
                            losses_auxi_trn_eval[-1]*auxi_s, losses_auxi_val[-1]*auxi_s,
                            losses_3d_pose_trn_eval[-1] * 1000, losses_3d_pose_val[-1] * 1000))

        epoch += 1

        # Alternate between optimizing Fully-supervised MPJPE vs. MPBOE loss
        if not semi_supervised and args.alternating_loss_epochs>0:
            if epoch%args.alternating_loss_epochs==0:
                # swap loss coefficients
                temp_coef = mpjpe_coef.clone()
                mpjpe_coef = mpboe_coef.clone()
                mpboe_coef = temp_coef

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            model_metadata = {
                'epoch': epoch, 'lr': lr,
                'optimizer': optimizer.state_dict(),
                'random_state': train_gen.random_state(),
                'model_pose': model_pose_train.module.state_dict()
                    if training_with_multi_gpu else model_pose_train.state_dict(),
                'random_state_semi': None, 'model_auxi': None
            }
            if args.use_auxiliary_model:
                model_metadata['random_state_semi'] = semi_gen.random_state()
                model_metadata['model_auxi'] = model_auxi_train.module.state_dict() \
                    if training_with_multi_gpu else model_auxi_train.state_dict()
            torch.save(model_metadata, chk_path)

    eot_time_struct = time.localtime() # get the end-of-training struct_time
    eot_time_string = time.strftime("%m/%d/%Y, %H:%M:%S", eot_time_struct)
    print('\n[INFO] Network training successfully completed on {}'.format(eot_time_string))
    print('[INFO] Total training time was {:.1f} hrs'.format((time.time() - train_t0)/3600))

    # Save final model
    if args.save_model:
        supv_set_tag = 's'+args.subjects_train.replace(',', '').replace('S', '')+supv_sbj_subset
        semi_set_tag = '-v-s'+args.subjects_unlabeled.replace(',', '').replace('S', '') if semi_supervised else ''
        if args.timestamp=='time':
            time_log_tag = time.strftime("_%Y.%m.%d-%H.%M", eot_time_struct)
        elif args.timestamp=='date':
            time_log_tag = time.strftime("_%Y.%m.%d", eot_time_struct)
        else: time_log_tag = ''
        file_name = '{}{}_{}_{}_e{}{}{}.bin'.format(
            supv_set_tag, semi_set_tag, pose2d_tag, arc_tag, epoch, frm_rate_tag, time_log_tag)
        file_path = os.path.join(args.checkpoint, file_name)
        print('[INFO] Saving model to {}'.format(file_path))
        model_metadata = {
            'epoch': epoch, 'lr': lr,
            'model_pose': model_pose_train.module.state_dict()
                if training_with_multi_gpu else model_pose_train.state_dict(),
            'model_auxi': None,
        }
        if args.use_auxiliary_model:
            model_metadata['model_auxi'] = model_auxi_train.module.state_dict() \
                if training_with_multi_gpu else model_auxi_train.state_dict()
        torch.save(model_metadata, file_path)


# Evaluate
def evaluate(infer_gen, fbom_instance=None, fbom_qset_kpt_idxs=None, vpse3d_2_orient_bone_idxs=None,
             map_jnt_2_bone=None, return_predictions=False, for_auxiliary_model=False):
    global loss_3d_pose_per_kpt, loss_pose_scale_per_kpt, loss_procrustes_per_kpt, \
        loss_3d_vel_per_kpt, loss_orient_per_kpt, loss_orient_per_bone, n_total_poses
    loss_3d_pose, loss_procrustes, loss_pose_scale, loss_3d_vel = 0, 0, 0, 0
    loss_bone_orient, loss_joint_orient = 0, 0

    with torch.no_grad():
        if not for_auxiliary_model:
            model_pose.eval()
        else: model_auxi.eval()
        N = 0
        for _, batch_3d, batch_2d in infer_gen.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if run_on_available_gpu:
                inputs_2d = inputs_2d.cuda()

            # Positional model
            if not for_auxiliary_model:
                predicted_3d_pose = model_pose(inputs_2d)
            else: predicted_3d_pose = model_auxi(inputs_2d)

            # Test-time augmentation (if enabled)
            if infer_gen.augment_enabled():
                # Note: inference with test-time augmentation is NOT EXACT.
                # Undo flipping and take average with non-flipped version
                predicted_3d_pose[1, :, :, 0] *= -1
                if not for_auxiliary_model:
                    if args.test_time_augmentation==2: # favorite right side augmentation
                        # exchange original left-side estimations with flipped right-side estimations
                        predicted_3d_pose[0,:,joints_left] = predicted_3d_pose[1,:,joints_right]
                        # take the average of original and flipped central keypoints
                        predicted_3d_pose[0,:,joints_inbtw] = torch.mean(predicted_3d_pose[:,:,joints_inbtw], dim=0)
                        predicted_3d_pose = predicted_3d_pose[[0]]
                    else:
                        # error between gt. pose and the avg. of non-flip & flipped predicted pose
                        predicted_3d_pose[1,:,joints_left+joints_right] = predicted_3d_pose[1,:,joints_right+joints_left]
                        predicted_3d_pose = torch.mean(predicted_3d_pose, dim=0, keepdim=True)
                else: predicted_3d_pose = torch.mean(predicted_3d_pose, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pose.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            if run_on_available_gpu:
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, root_kpt_idx] = 0
            if infer_gen.augment_enabled():
                inputs_3d = inputs_3d[:1]
                # inputs_3d = inputs_3d[[1]] # del >> swap vs. no-swap eval (swap:[1]. no-swap:[0])
            n_poses = inputs_3d.shape[0]*inputs_3d.shape[1]

            # Compute orientation errors - before dropping 'Nse' if ignore_nose_kpt
            fboi_indexes = fbones_oi_indices if ignore_nose_kpt else None
            map_kpt2idx = KPT16_2_IDX if ignore_nose_kpt else KPT_2_IDX
            # MPBOE
            b_orient_err, orient_err_per_bone = mpboe(predicted_3d_pose, inputs_3d,
                fbom_instance, fbom_qset_kpt_idxs, vpse3d_2_orient_bone_idxs, fboi_indexes)
            loss_bone_orient += n_poses * b_orient_err.item()
            loss_orient_per_bone += n_poses * to_numpy(orient_err_per_bone)
            # J-MPBOE
            j_orient_err, orient_err_per_kpt = j_mpboe(predicted_3d_pose, inputs_3d, fbom_instance,
                fbom_qset_kpt_idxs, vpse3d_2_orient_bone_idxs, map_jnt_2_bone, map_kpt2idx, processor, fboi_indexes)
            loss_joint_orient += n_poses * j_orient_err.item()
            loss_orient_per_kpt += n_poses * to_numpy(orient_err_per_kpt)

            if ignore_nose_kpt:
                # activated for HRNet (data_2d_h36m_hr.npz) without nose kpt
                inputs_3d = inputs_3d[:,:,koi_indexes,:]
                predicted_3d_pose = predicted_3d_pose[:,:,koi_indexes,:]

            # Compute 3d pose and scale normalized 3d pose errors
            predicted_3d_pose -= predicted_3d_pose[:, :, [root_kpt_idx]] # align to pelvis
            error, error_per_kpt = mpjpe(predicted_3d_pose, inputs_3d, ret_per_kpt=True)
            loss_3d_pose += n_poses * error.item()
            loss_3d_pose_per_kpt += n_poses * to_numpy(error_per_kpt)

            norm_err, norm_err_per_kpt = n_mpjpe(predicted_3d_pose, inputs_3d)
            loss_pose_scale += n_poses * norm_err.item()
            loss_pose_scale_per_kpt += n_poses * to_numpy(norm_err_per_kpt)

            N += n_poses
            n_total_poses += n_poses

            # Compute procrustes aligned 3d pose error
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pose = predicted_3d_pose.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            procustes_err, procustes_err_per_kpt = p_mpjpe(predicted_3d_pose, inputs)
            loss_procrustes += n_poses * procustes_err
            loss_procrustes_per_kpt += n_poses * procustes_err_per_kpt

            # Compute velocity error
            velocity_err, velocity_err_per_kpt = mean_velocity_error(predicted_3d_pose, inputs)
            loss_3d_vel += n_poses * velocity_err
            loss_3d_vel_per_kpt += n_poses * velocity_err_per_kpt

    e1 = (loss_3d_pose / N)*1000
    e2 = (loss_procrustes / N)*1000
    e3 = (loss_pose_scale / N)*1000
    ev = (loss_3d_vel / N)*1000
    be = (loss_bone_orient / N)*1000
    je = (loss_joint_orient / N)*1000

    return e1, e2, e3, ev, be, je


if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('[INFO] this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator(None, None, [input_keypoints], pad=pad, causal_shift=causal_shift,
                             augment=bool(args.test_time_augmentation), kps_left=kps_left, kps_right=kps_right,
                             joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)
    if model_auxi is not None and ground_truth is None:
        prediction_auxi = evaluate(gen, return_predictions=True, for_auxiliary_model=True)
        prediction += prediction_auxi

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            ground_truth[:, 1:] += trajectory
            prediction += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        from common.visualization import render_animation
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)

elif args.predict_poses:
    assert (args.evaluate), 'args.predict_poses should be enabled with args.evaluate'
    # assumes model pose and model_auxi are already loaded
    model_pose.eval()
    model_auxi.eval()
    print('Creating custom H36M predicted 3D pose dataset using {} 2D poses...'.format(args.keypoints))
    pred_pose3d_per_sbj = {}

    for subject in dataset.subjects():
        if pred_pose3d_per_sbj.get(subject, None) is None: pred_pose3d_per_sbj[subject] = {}

        for action in dataset[subject].keys():
            if pred_pose3d_per_sbj[subject].get(action, None) is None: pred_pose3d_per_sbj[subject][action] = {}
            anim = dataset[subject][action]

            if 'positions' in anim:
                poses_2d = keypoints[subject][action] # c*(?,j,2)
                assert (len(poses_2d)==len(anim['cameras']))

                if pred_pose3d_per_sbj[subject].get(action, None) is None:
                    pred_pose3d_per_sbj[subject][action]['positions'] = anim['positions']
                pred_sbj_act_position_3d = []

                for cam_idx, cam in enumerate(anim['cameras']): # Iterate across cameras
                    if poses_2d[cam_idx].shape[1] == 16: # characteristic of data_2d_h36m_hr.npz HRNet
                        poses_2d[cam_idx] = add_estimated_nose_kpt(poses_2d[cam_idx])
                    generator = UnchunkedGenerator(None, None, [poses_2d[cam_idx]], pad=pad, causal_shift=causal_shift,
                                    augment=bool(args.test_time_augmentation), kps_left=kps_left, kps_right=kps_right,
                                    joints_left=joints_left, joints_right=joints_right)

                    with torch.no_grad():
                        gen_call_cnt = 0
                        for _, batch_3d, batch_2d in generator.next_epoch():
                            gen_call_cnt += 1
                            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                            if run_on_available_gpu:
                                inputs_2d = inputs_2d.cuda()

                            # Positional model
                            predicted_3d_pose = model_pose(inputs_2d)
                            predicted_3d_traj = model_auxi(inputs_2d)

                            # Test-time augmentation (if enabled)
                            if generator.augment_enabled():
                                # Note: inference with test-time augmentation is NOT EXACT.
                                # Undo flipping and take average with non-flipped version
                                predicted_3d_pose[1, :, :, 0] *= -1
                                if args.test_time_augmentation==2: # favorite right side augmentation
                                    # exchange original left-side estimations with flipped right-side estimations
                                    predicted_3d_pose[0,:,joints_left] = predicted_3d_pose[1,:,joints_right]
                                    # take the average of original and flipped central keypoints
                                    predicted_3d_pose[0,:,joints_inbtw] = torch.mean(predicted_3d_pose[:,:,joints_inbtw], dim=0)
                                    predicted_3d_pose = predicted_3d_pose[[0]]
                                else:
                                    # error between gt. pose and the avg. of non-flip & flipped predicted pose
                                    predicted_3d_pose[1,:,joints_left+joints_right] = predicted_3d_pose[1,:,joints_right+joints_left]
                                    predicted_3d_pose = torch.mean(predicted_3d_pose, dim=0, keepdim=True)
                                predicted_3d_traj = torch.mean(predicted_3d_traj, dim=0, keepdim=True)

                            assert (predicted_3d_pose.shape[:2]==predicted_3d_traj.shape[:2])
                            cam_pred_pose_3d = predicted_3d_pose[0] + predicted_3d_traj[0]
                            assert (cam_pred_pose_3d.shape == anim['positions_3d'][cam_idx].shape), \
                                '{} vs. {}'.format(cam_pred_pose_3d.shape, anim['positions_3d'][cam_idx].shape)
                            pred_sbj_act_position_3d.append(cam_pred_pose_3d.cpu().numpy())
                        assert (gen_call_cnt==1)

                assert (pred_pose3d_per_sbj[subject][action].get('positions_3d', None) is None)
                pred_pose3d_per_sbj[subject][action]['positions_3d'] = pred_sbj_act_position_3d

    wrt_dataset_path = args.dataset_home_dir + 'estimated_3d_h36m_{}_2d.npz'.format(args.keypoints)
    print('Saving...')
    np.savez_compressed(wrt_dataset_path, positions_3d=pred_pose3d_per_sbj, src_model_name=chk_filename)
    print('[INFO] Custom dataset saved to {}'.format(wrt_dataset_path))
    print('Done.')

else:
    print('Evaluating on Test-set {} with {} 2D pose inputs...'.format(
        args.subjects_test, 'ground-truth' if args.test_with_2dgt else args.keypoints.upper()+' detected'))
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))

    # Prepare Free-bone Orientation instance for MPBOE evaluation metric
    fbo_qset_kpt_idxs, xy_yx_axis_dirs, xy_yx_idxs, z_axis_ab_idxs, yx_axis_ab_idxs, \
    nr_fb_idxs, map_jnt_2_bone, vpse3d_2_orient_bone_idxs = fboa_rotamatrix_config(MPBOE_BONE_ALIGN_CONFIG,
        list(MPBOE_BONE_ALIGN_CONFIG.keys()), group_jnts=False, quintuple=True, drop_face_bone=ignore_nose_kpt)
    fbo_uvec = FreeBoneOrientation(args.batch_size, xy_yx_axis_dirs, z_axis_ab_idxs, yx_axis_ab_idxs, xy_yx_idxs,
                    nr_fb_idxs=nr_fb_idxs, quintuple=True, validate_ops=bool(args.evaluate), ret_mode=0, rot_tfm_mode=0)

    def fetch_actions(actions):
        in_poses_2d = []
        out_poses_3d = []
        for subject, action in actions:
            if args.test_with_2dgt:
                poses_2d = gt2d_keypoints[subject][action]
            else: poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                in_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert(len(poses_3d) == len(poses_2d)), 'Camera count mismatch'
            for i in range(len(poses_3d)): # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(in_poses_2d)):
                in_poses_2d[i] = in_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        return out_poses_3d, in_poses_2d

    def run_evaluation(actions, action_filter=None):
        global loss_3d_pose_per_kpt, loss_pose_scale_per_kpt, loss_procrustes_per_kpt, \
            loss_3d_vel_per_kpt, loss_orient_per_kpt, loss_orient_per_bone, n_total_poses
        n_jnts = 16 if ignore_nose_kpt else 17
        loss_3d_pose_per_kpt = np.zeros(n_jnts, dtype=np.float32)
        loss_pose_scale_per_kpt = np.zeros(n_jnts, dtype=np.float32)
        loss_procrustes_per_kpt = np.zeros(n_jnts, dtype=np.float32)
        loss_3d_vel_per_kpt = np.zeros(n_jnts, dtype=np.float32)
        loss_orient_per_kpt = np.zeros(n_jnts, dtype=np.float32)
        loss_orient_per_bone = np.zeros(n_jnts-1, dtype=np.float32)
        n_total_poses = 0

        errors_p1, errors_p2, errors_p3, errors_vel, errors_boe, errors_joe = [], [], [], [], [], []
        list_of_actions = list(actions.keys())
        list_of_actions.sort()
        for action_key in list_of_actions:
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act, pad=pad, causal_shift=causal_shift,
                                     augment=bool(args.test_time_augmentation), kps_left=kps_left, kps_right=kps_right,
                                     joints_left=joints_left, joints_right=joints_right)
            e1, e2, e3, ev, be, je = evaluate(gen, fbo_uvec, fbo_qset_kpt_idxs, vpse3d_2_orient_bone_idxs, map_jnt_2_bone)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)
            errors_boe.append(be)
            errors_joe.append(je)

        print('Test time augmentation: {} >> {}'.format(args.test_time_augmentation, bool(args.test_time_augmentation)))
        table0_rows = [
            np.concatenate((['  MPJPE'], errors_p1)),
            np.concatenate((['N-MPJPE'], errors_p3)),
            np.concatenate((['P-MPJPE'], errors_p2)),
            np.concatenate((['  MPBOE'], errors_boe)),
            np.concatenate((['J-MPBOE'], errors_joe)),
            np.concatenate((['  MPJVE'], errors_vel))]
        header0 = ['Metric (mm)'] + abbreviate_action_names(list_of_actions)
        table0 = tabulate(table0_rows, headers=header0, tablefmt='psql', floatfmt='.2f')
        print(table0)

        print('Protocol #1   (MPJPE) action-wise naive average: {:4.1f} mm'.format(np.mean(errors_p1)))
        print('Protocol #3 (N-MPJPE) action-wise naive average: {:4.1f} mm'.format(np.mean(errors_p3)))
        print('Protocol #2 (P-MPJPE) action-wise naive average: {:4.1f} mm'.format(np.mean(errors_p2)))
        print('Orient-B #1   (MPBOE) action-wise naive average: {:4.1f} mm'.format(np.mean(errors_boe)))
        print('Orient-J #2 (J-MPBOE) action-wise naive average: {:4.1f} mm'.format(np.mean(errors_joe)))
        print('Velocity      (MPJVE) action-wise naive average: {:4.2f} mm'.format(np.mean(errors_vel)))

        loss_3d_pose_per_kpt = (loss_3d_pose_per_kpt / n_total_poses) * 1000
        loss_procrustes_per_kpt = (loss_procrustes_per_kpt / n_total_poses) * 1000
        loss_pose_scale_per_kpt = (loss_pose_scale_per_kpt / n_total_poses) * 1000
        loss_3d_vel_per_kpt = (loss_3d_vel_per_kpt / n_total_poses) * 1000
        loss_orient_per_kpt = (loss_orient_per_kpt / n_total_poses) * 1000
        loss_orient_per_bone = (loss_orient_per_bone / n_total_poses) * 1000

        if ignore_nose_kpt: KPT_HEADER_ORDER.remove('Nse')
        map_kpt2idx = KPT16_2_IDX if ignore_nose_kpt else KPT_2_IDX
        reorder_kpt_idxs = [map_kpt2idx[kpt_id] for kpt_id in KPT_HEADER_ORDER]

        table1_rows = [
            np.concatenate((['  MPJPE'], loss_3d_pose_per_kpt[reorder_kpt_idxs], [np.mean(loss_3d_pose_per_kpt)])),
            np.concatenate((['N-MPJPE'], loss_pose_scale_per_kpt[reorder_kpt_idxs], [np.mean(loss_pose_scale_per_kpt)])),
            np.concatenate((['P-MPJPE'], loss_procrustes_per_kpt[reorder_kpt_idxs], [np.mean(loss_procrustes_per_kpt)])),
            np.concatenate((['J-MPBOE'], loss_orient_per_kpt[reorder_kpt_idxs], [np.mean(loss_orient_per_kpt)])),
            np.concatenate((['  MPJVE'], loss_3d_vel_per_kpt[reorder_kpt_idxs], [np.mean(loss_3d_vel_per_kpt)]))]
        header1 = ['Metric (mm)']+KPT_HEADER_ORDER+['True Avg']
        table1 = tabulate(table1_rows, headers=header1, tablefmt='psql', floatfmt='.2f')
        print(table1)

        if ignore_nose_kpt: BONE_HEADER_ORDER.remove('Face'), BONE_ALIGN_ORDER.remove('Face')
        reorder_bone_idxs = []
        for h_bone_id in BONE_HEADER_ORDER:
            for idx, a_bone_id in enumerate(BONE_ALIGN_ORDER):
                if h_bone_id==a_bone_id: reorder_bone_idxs.append(idx)
        header2 = ['Metric (mm)']+BONE_HEADER_ORDER+['Avg']
        table2_rows = [
            np.concatenate((['  MPBOE'], loss_orient_per_bone[reorder_bone_idxs], [np.mean(loss_orient_per_bone)]))]
        table2 = tabulate(table2_rows, headers=header2, tablefmt='psql', floatfmt='.2f')
        print(table2)
        print()


    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')