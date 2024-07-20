# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np

from common.arguments import parse_args
import torch
#torch.cuda.empty_cache() # Added by Lawrence 05/05/21, deactivated 07/05/2022
#torch.autograd.set_detect_anomaly(True)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import gc
import sys
import errno

from scipy.special import softmax
import matplotlib.pyplot as plt

from agents.pose_regs import *
from agents.rbo_transform_np import *
from agents.rbo_transform_tc import *
from agents.helper import *
from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random
from agents.visuals import plot_3d_pose, plot_2d_pose

np.set_printoptions(precision=4, linewidth=128, suppress=True)
torch.set_printoptions(precision=4)

args = parse_args()

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

#dataset_home_dir = '../../datasets/human36m/'

print('Loading dataset...')
dataset_path = args.dataset_home_dir + 'data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset(args.dataset_home_dir + 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
gt_keypoints = np.load(args.dataset_home_dir + 'data_2d_' + args.dataset + '_gt.npz', allow_pickle=True)
keypoints_metadata = gt_keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
del gt_keypoints
keypoints = np.load(args.dataset_home_dir + 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints = keypoints['positions_2d'].item()

# Set checkpoint directory (Added by Lawrence, 09/09/21)
log_func_modes = \
    (args.logli_func_mode, args.norm_logli_func_mode, args.reverse_logli_func_mode, args.shift_logli_func_mode)
supr_subjects = args.subjects_train.replace(",", ".")
supr_sbj_subset = '' if args.subset==1 else "."+str(args.subset).replace("0.", "")
supr_subset_tag = '{}{}'.format(supr_subjects, supr_sbj_subset)
frm_rate_tag = '' if args.downsample==1 else '_ds{}'.format(args.downsample)
if args.export_training_curves or args.log_losses_numpy or args.save_model:
    semi_supr_wgts = np.float32([float(x) for x in args.semi_supervised_loss_coef.split(',')])
    if np.all(semi_supr_wgts[:2]==1.) and np.all(semi_supr_wgts[2:]==0):
        sota_tag = '_{}sota'.format(args.dir_tag_prefix)
    else: sota_tag = '{}'.format(args.dir_tag_prefix)
    gen_priors_tag = '_all' if args.gen_pose_priors==0 else '_sub'
    logp_tag = '{0}{1}{2}{3}'.format(*log_func_modes)
    exp_dir = 'experiments/{}_2d_ds{}{}/{}{}/{}'.format(args.keypoints[:3], args.downsample, sota_tag,
                                                        supr_subset_tag, gen_priors_tag, logp_tag)
    try:
        os.makedirs(exp_dir) # Create experiment directory if it does not exist
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create experiment directory:', exp_dir)

# Prune 2d-poses for 1-1 correspondence with 3d-poses
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        # Short-Data: Skip 2D-poses without corresponding 3D-poses
        if 'positions_3d' not in dataset[subject][action]:
            print('[INFO] Missing corresponding 3D-poses for {}:"{}"'.format(subject, action))
            continue # TODO: disable? Opportunity to include more unlabelled data

        for cam_idx in range(len(keypoints[subject][action])):
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            kpt2d_length = keypoints[subject][action][cam_idx].shape[0]
            msg_tmp = 'GT 2D kpts frames of {}-{} at cam {} is less than {} 3D frames'
            assert kpt2d_length >= mocap_length, msg_tmp.format(subject, action, cam_idx, mocap_length)

            if kpt2d_length > mocap_length:
                # Shorten sequence TODO*NOTE: Another reason why less than available frames are used (more unlabelled data)
                print('[INFO] Excluding {} extra 2d-pose sequence of {}:"{}"'.format(kpt2d_length-mocap_length, subject, action))
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

# Normalize 2d pose pixel coordinates
for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]

semi_supervised = len(subjects_semi) > 0
if semi_supervised and not dataset.supports_semi_supervised():
    raise RuntimeError('Semi-supervised training is not implemented for this dataset')

# loss terms weights/coefficients and decay
#p3d_coef, trj_coef = \
#    torch.from_numpy(np.float32([float(x) for x in args.supervised_loss_coef.split(',')])).to(processor)
rp2d_coef, mble_coef, pto_coef, bse_coef, bpc_coef, jmc_coef, mce_wcx_coef, mce_ncx_coef = \
    torch.from_numpy(np.float32([float(x) for x in args.semi_supervised_loss_coef.split(',')])).to(processor)
#print('[INFO] Supervised branch loss coefficients: 3D-POSE:{:.5f} - 3D-TRAJ:{:.5f}'.format(p3d_coef, trj_coef))
print('[INFO] Weakly-Supervised branch loss coefficients:\n       2D-PROJ:{:.5f} - MBLE:{:.5f} - '
      'PTO:{:.5f} - SBE:{:.5f} - BPE:{:.5f} - JMC:{:.5f} - MCE_wcx:{:.5f} - MCE_ncx:{:.5f}'.
      format(rp2d_coef, mble_coef if args.mean_bone_length_term else 0, pto_coef if args.pelvis_placement_term else 0,
             bse_coef if args.bone_symmetry_term else 0, bpc_coef if args.bone_proportion_term else 0,
             jmc_coef if args.joint_mobility_term else 0, mce_wcx_coef, mce_ncx_coef))
use_traj_model = True if (args.subjects_unlabeled!='' and (args.projection_type>0 or args.mce_type>0)) else False
ignore_some_keypoints = False # Default setting unless estimating less than 17 joint skeletal pose
print('[INFO] semi_supervised:{} - use_traj_model:{}'.format(semi_supervised, use_traj_model))


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True, subset_type='Unkown', visualize_examples=False):
    global ignore_some_keypoints
    out_poses_3d = []
    out_poses_2d = []
    out_cams_intrinsic = [] # Added by Lawrence (03/14/22)
    out_cams_extrinsic = []
    info_table_log = {} # key->subject, value->actions_cnt_dict
    session_ids = [] # id: subject_action
    for subject in subjects:
        subject_cnt = 0
        actions_cnt = {}
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found: continue

            # get action name (ie. minus number suffix)
            char_idx = action.find(' ')
            action_name = action[0:char_idx] if char_idx>=0 else action

            sub_act_cnt = 0
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                if poses_2d[i].shape[1] == 16: # characteristic of data_2d_h36m_hr.npz HRNet
                    # Add nose keypoint as the midway position between neck and skull keypoint
                    ignore_some_keypoints = True
                    nck_idx, nse_idx, adj_skl_idx = KPT_2_IDX['Nck'], KPT_2_IDX['Nse'], KPT_2_IDX['Skl']-1
                    est_nse_pos = np.mean(poses_2d[i][:,[nck_idx, adj_skl_idx]], axis=1, keepdims=True) # (?,16,2)->(?,17,2)
                    poses_2d[i] = np.concatenate([poses_2d[i][:,:nse_idx], est_nse_pos, poses_2d[i][:,nse_idx:]], axis=1)
                out_poses_2d.append(poses_2d[i])
                sub_act_cnt += poses_2d[i].shape[0] # shape->(varying # of frames between 1-5k, joints:17, [x,y]:2)
                session_ids.append((subject, action_name))
                if visualize_examples:
                    j = np.random.randint(0, poses_2d[i].shape[0])
                    plot_2d_pose(poses_2d[i][j], '{} [{}] - Seq:{} Frm:{}'.format(subject, action, i, j), KPT_2_IDX)

            # log number of frames for action
            actions_cnt[action_name] = actions_cnt.get(action_name, 0) + sub_act_cnt
            subject_cnt += sub_act_cnt

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam: out_cams_intrinsic.append(cam['intrinsic'])
                    if 'extrinsic' in cam: out_cams_extrinsic.append(cam['extrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i]) # (?,17,3)

        actions_cnt['All-Acts'] = subject_cnt
        info_table_log[subject] = actions_cnt
    if subset>=1:
        info_table = display_subject_actions_frame_counts(info_table_log, sum_rows=len(subjects)>1)
        print('INFO: {} Data (With {:,} Subjects-&-Actions Sessions)\n{}'.
              format(subset_type, len(out_poses_2d), info_table))

    if len(out_cams_intrinsic) == 0: out_cams_intrinsic = None
    if len(out_cams_extrinsic) == 0: out_cams_extrinsic = None
    if len(out_poses_3d) == 0: out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        subject_id = ''
        subject_cnt = 0
        actions_cnt = {}
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
            # log reduced frame count
            sub_act_cnt = len(out_poses_2d[i])
            subject, action_name = session_ids[i]
            if i==0: subject_id = '{} {} ds={}'.format(subset, subject, stride)
            actions_cnt[action_name] = actions_cnt.get(action_name, 0) + sub_act_cnt
            subject_cnt += sub_act_cnt
        # display table
        actions_cnt['All-Acts'] = subject_cnt
        info_table_log[subject_id] = actions_cnt
        info_table = display_subject_actions_frame_counts(info_table_log, sum_rows=False)
        print('INFO: {} Data (With {:,} Subjects-&-Actions Sessions)\n{}'.
              format(subset_type, len(out_poses_2d), info_table))
    elif stride > 1:
        # Downsample as requested
        info_table_log = {} # key->subject, value->actions_cnt_dict
        subjects_frm_cnt = {} # key->subject, value->total-frames
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
            # log reduced frame cnt
            subject, action_name = session_ids[i]
            subject = '{} ds={}'.format(subject, stride)
            if info_table_log.get(subject)==None: info_table_log[subject] = {}
            sub_act_cnt = len(out_poses_2d[i])
            info_table_log[subject][action_name] = info_table_log[subject].get(action_name, 0) + sub_act_cnt
            subjects_frm_cnt[subject] = subjects_frm_cnt.get(subject, 0) + sub_act_cnt
        # display table
        for subject in subjects:
            subject = '{} ds={}'.format(subject, stride)
            info_table_log[subject]['All-Acts'] = subjects_frm_cnt[subject]
        info_table = display_subject_actions_frame_counts(info_table_log, sum_rows=len(subjects)>1)
        print('INFO: Down-sampled {} Data (With {:,} Subjects-&-Actions Sessions)\n{}'.
              format(subset_type, len(out_poses_2d), info_table))

    return out_cams_intrinsic, out_cams_extrinsic, out_poses_3d, out_poses_2d


# Extract and configure pose prior regularizer parameters
# ------------------------------------------------------------------------------------------------------------------
if args.gen_pose_priors>=0:
    ls_tag = args.log_likelihood_spread
    nbr_tag = args.n_bone_ratios
    cov_tag = args.pdf_covariance_type # best: 'noctr'
    jmc_grp_tag = 'grpjnt' if args.group_jmc else 'perjnt'
    bpc_grp_tag = 'grpprp' if args.group_bpc else 'perprp'
    frt_tag = '' if args.gen_pose_priors==0 else frm_rate_tag
    if args.gen_pose_priors==0: set_tag = 'S1.S5.S6.S7.S8'
    elif args.gen_pose_priors==1: set_tag = supr_subset_tag
    else: set_tag = 'S1.05' # when args.gen_pose_priors==2
    print('[INFO] generating PPR parameters from {} subset...'.format(set_tag))

    # Initialize necessary torch constants
    likelihood_eps = torch_t(args.likelihood_epsilon) # 1e-0
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
    print('[INFO] wout_cam_ext_mce_type:{} with_cam_ext_mce_type:{} apply_mv_pose_consistency:{}'.
          format(wout_cam_ext_mce_type, with_cam_ext_mce_type, apply_mv_pose_consistency))
    print('[INFO] Computing: 3D-Traj:{} - 2D-Proj:{} - MBLE:{} - PTO:{} - BSE:{} - BPC:{} - JMC:{} - MCE:{}'.
          format(use_traj_model, args.projection_type!=0, args.mean_bone_length_term, args.pelvis_placement_term,
                 args.bone_symmetry_term, args.bone_proportion_term, args.joint_mobility_term, apply_mv_pose_consistency))
    print('[INFO] Other weakly-supervised conditional checks: Bone_len:{} - Repose:{}'.format(
        args.bone_symmetry_term or args.bone_proportion_term, args.reposition_to_origin and (
                args.bone_symmetry_term or args.bone_proportion_term or args.joint_mobility_term)))

    # Extract and organize Pose Priors configuration parameters
    # BPC
    bpc_priors = pickle_load('./priors/{}/bone_priors_wxaug_{}{}_br_{}_{:.0e}.pickle'.
                             format(set_tag, nbr_tag, frt_tag, bpc_grp_tag, ls_tag))
    bone_prop_pair_idxs_A, bone_prop_pair_idxs_B = extract_bpc_config_params(bpc_priors, VIDEOPOSE3D_BONE_ID_2_IDX)
    bpc_variance, bpc_exponent_coefs, bpc_mean, bpc_max_likelihoods, bpc_likeli_argmax, bpc_logli_spread, \
    bpc_logli_mean, bpc_logli_std, bpc_logli_min, bpc_logli_span, bpc_ilmw_wgts, bpc_log_of_spread, \
    bpc_move_up_const, per_comp_bpc_wgts, bpc_loglihood_wgts = configure_bpc_likelihood(bpc_priors, log_func_modes, log_lihood_eps)
    # JMC
    rot_tag = '_quat' if args.rbo_ops_type=='quat' else '_rmtx'#''
    jmc_priors = pickle_load('./priors/{}/joint_priors{}_{}{}_wxaug_16_{}_{:.0e}.pickle'.
                             format(set_tag, frt_tag, cov_tag, rot_tag, jmc_grp_tag, ls_tag))
    assert (args.group_jmc==jmc_priors['group']), 'args.group_jmc:{} != {}'.format(args.group_jmc, jmc_priors['group'])
    mcv_cnt = args.multi_cam_views # multi-cam-views used in semi-supervised pose consistency loss
    #ret_mode = 0 if mcv_cnt>1 and wout_cam_ext_mce_type<=-2 else 1
    ret_mode = 1 if wout_cam_ext_mce_type==-1 else 0
    print('ret_mode', ret_mode)
    if args.rbo_ops_type=='quat':
        quad_kpt_idxs, axis1_quadrant, axis2_quadrant, plane_proj_multiplier, hflip_multiplier = \
            extract_rfboa_quaternion_config(jmc_priors['joint_align_config'], jmc_priors['joint_order'], args.group_jmc)
        orient_fb_uvec = FreeBoneOrientation(args.batch_size, quad_uvec_axes1=axis1_quadrant, quad_uvec_axes2=axis2_quadrant,
                                             plane_proj_mult=plane_proj_multiplier, hflip_multiplier=hflip_multiplier,
                                             validate_ops=False, ret_mode=ret_mode, rot_tfm_mode=1,)
    else:
        quad_kpt_idxs, xy_yx_axis_dirs, xy_yx_idxs, z_axis_ab_idxs, yx_axis_ab_idxs = extract_rfboa_rotamatrix_config(
            jmc_priors['joint_align_config'], jmc_priors['joint_order'], group_jnts=args.group_jmc)
        orient_fb_uvec = FreeBoneOrientation(args.batch_size, xy_yx_axis_dirs, z_axis_ab_idxs, yx_axis_ab_idxs,
                                             xy_yx_idxs, validate_ops=False, ret_mode=ret_mode, rot_tfm_mode=0)
    jmc_params, jmc_loglihood_wgts, per_comp_jmc_wgts = \
        configure_jmc_likelihood(jmc_priors, log_func_modes, log_lihood_eps, args.jmc_ranks)
# ------------------------------------------------------------------------------------------------------------------

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cams_in_valid, _, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter, subset_type='Test')
print('Test, ignore_some_keypoints:{}'.format(ignore_some_keypoints)) #del
if ignore_some_keypoints:
    assert(args.keypoints=='hr'), "This feature has only been tested for data_2d_h36m_hr.npz"
    kpts_oi_indices = list(range(0, 17)) # keypoints-of-interest-indices
    kpts_oi_indices.remove(KPT_2_IDX['Nse']) # remove nose kpt from consideration
    fbones_oi_indices = list(range(0, 16)) # free-bones-of-interest-indices
    head_fbone_idx = get_id_index(jmc_priors['joint_order'], 'Head')
    fbones_oi_indices.remove(head_fbone_idx) # remove head free-bone (link to nose kpt) from consideration

filter_widths = [int(x) for x in args.architecture.split(',')]

if not args.disable_optimizations and not args.dense and args.stride == 1:
    # Use optimized model for single-frame predictions
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                            dataset.skeleton().num_joints(), filter_widths=filter_widths, causal=args.causal,
                            dropout=args.dropout, channels=args.channels)
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                            dataset.skeleton().num_joints(), filter_widths=filter_widths, causal=args.causal,
                            dropout=args.dropout, channels=args.channels, dense=args.dense)
# Note: model_pos_train is the trained instance, whose weights are loaded into model_pos for evaluation
model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                          filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                          dense=args.dense)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if run_on_available_gpu:
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()
    # Edited by Lawrence for multi-gpu
    if torch.cuda.device_count()>1 and args.multi_gpu_training:
        print('INFO: Using {} GPUs for model_pos*'.format(torch.cuda.device_count()))
        model_pos = nn.DataParallel(model_pos)
        model_pos_train = nn.DataParallel(model_pos_train)

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

    if args.evaluate and 'model_traj' in checkpoint:
        # Load trajectory model if it contained in the checkpoint (e.g. for inference in the wild)
        model_traj = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                            filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)
        if run_on_available_gpu:
            model_traj = model_traj.cuda()
        model_traj.load_state_dict(checkpoint['model_traj'])
    else:
        model_traj = None

test_generator = UnchunkedGenerator(cams_in_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))


if not args.evaluate:
    print('3D Supervised Training') #del
    cams_in_train, _, poses_train, poses_train_2d = \
        fetch(subjects_train, action_filter, subset=args.subset, subset_type='3D Supervised Training')

    lr = args.learning_rate
    if semi_supervised:
        print('Semi-Supervised Training') #del
        cams_in_semi, cams_ex_semi, _, poses_semi_2d = \
            fetch(subjects_semi, action_filter, parse_3d_poses=False, subset_type='Semi-Supervised Training')
        if use_traj_model:
            if not args.disable_optimizations and not args.dense and args.stride == 1:
                # Use optimized model for single-frame predictions
                model_traj_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
            else:
                # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
                model_traj_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                        dense=args.dense)
            # Note: model_traj_train is the trained instance, whose weights are loaded into model_traj for evaluation
            model_traj = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)
            if run_on_available_gpu:
                model_traj = model_traj.cuda()
                model_traj_train = model_traj_train.cuda()
                # Edited by Lawrence for multi-gpu
                if torch.cuda.device_count()>1 and args.multi_gpu_training:
                    print('INFO: Using {} GPUs for model_traj*'.format(torch.cuda.device_count()))
                    model_traj = nn.DataParallel(model_traj)
                    model_traj_train = nn.DataParallel(model_traj_train)

            optimizer = optim.Adam(list(model_pos_train.parameters()) + list(model_traj_train.parameters()), lr=lr, amsgrad=True)
        else: optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

        losses_2d_train_unlabeled = []
        losses_2d_train_labeled_eval = []
        losses_2d_train_unlabeled_eval = []
        losses_2d_valid = []

        losses_traj_train = []
        losses_traj_train_eval = []
        losses_traj_valid = []

        # Added by Lawrence 03/29/21 - 09/09/21
        losses_mble_train_unlabeled = []
        losses_mce_wcx_train_unlabeled = []
        losses_mce_ncx_train_unlabeled = []
        losses_pto_train_unlabeled = []
        losses_bse_train_unlabeled = []
        losses_bpc_train_unlabeled = []
        losses_jmc_train_unlabeled = []

        mean_loss_sesup = []
        mean_weighted_loss_sesup = []
    else:
        optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)
        losses_pto_train_labeled = []
        losses_bse_train_labeled = []
        losses_bpc_train_labeled = []
        losses_jmc_train_labeled = []

    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []
    mean_loss_sup = []
    mean_weighted_loss_sup = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(args.batch_size//args.stride, cams_in_train, None, poses_train, poses_train_2d, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator(cams_in_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
    if semi_supervised:
        semi_generator = ChunkedGenerator(args.batch_size//args.stride, cams_in_semi, cams_ex_semi, None, poses_semi_2d, args.stride,
                                          pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation, random_seed=4321,
                                          multi_cams=mcv_cnt, mce_flip=wout_cam_ext_mce_type<0, kps_left=kps_left, kps_right=kps_right,
                                          joints_left=joints_left, joints_right=joints_right, endless=True)
        semi_generator_eval = UnchunkedGenerator(cams_in_semi, None, poses_semi_2d,
                                                 pad=pad, causal_shift=causal_shift, augment=False)
        print('INFO: Semi-supervision on {} frames'.format(semi_generator_eval.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        lr = checkpoint['lr']
        if semi_supervised:
            if use_traj_model:
                model_traj_train.load_state_dict(checkpoint['model_traj'])
                model_traj.load_state_dict(checkpoint['model_traj'])
            semi_generator.set_random_state(checkpoint['random_state_semi'])

    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')


    if args.gen_pose_priors<0:
        pose_reg_properties_extractor(train_generator, subjects_train, args, supr_subset_tag, frm_rate_tag)
        sys.exit(0)

    n_full_supervised_iters = 0
    n_semi_supervised_iters = 0
    last_warmup_epoch = args.warmup
    plot_log_freq = np.floor(args.epochs/5) # plots are logged 5 or 6 times during training
    #projection_func = project_to_2d_linear if args.linear_projection else project_to_2d #***
    if args.projection_type==1: projection_func = project_to_2d
    elif args.projection_type==2: projection_func = project_to_2d_linear

    zero_tsr = torch_t(0.)
    pos_one_tsr = torch_t(1.)
    neg_one_tsr = torch_t(-1.)
    pos_two_tsr = torch_t(2.)
    one_k_tsr = torch_t(1e+3)
    log_epsilon = torch_t(1e+0) # sigm:1e-6
    q1_q3_tsr = torch_t([0.25, 0.75])
    repose_pelvis_arr = np.ones((1,1,17,1), dtype=np.float32)
    repose_pelvis_arr[0,0,0,0] = 0.
    repose_pelvis_tsr = torch_t(repose_pelvis_arr)
    expected_pelvis_pos = torch.zeros((1,1,1,3), dtype=torch.float32, device=torch.device(processor))
    if semi_supervised and args.projection_type<0:
        # Note args.batch_size will be the same as semi_generator.batch_size
        projection_mtx_base = torch.zeros(
            (semi_generator.batch_size,1,17,4,4), dtype=torch.float32, device=torch.device(processor)) # (?,f,17,4,4)
        projection_mtx_base[:,:,:,[0,1,3],[0,1,3]] =  1
        homogenous_one_base = torch.ones(
            (semi_generator.batch_size,1,17,1), dtype=torch.float32, device=torch.device(processor)) # (?,f,17,1)
    else: projection_mtx_base, homogenous_one_base = None, None

    def project2d_reconstruction(predicted_3d_pos, predicted_traj, cam, target, swap_dims=False):
        if args.projection_type>0:
            reconstruction = project_to_2d(predicted_3d_pos + predicted_traj, cam)
        elif args.projection_type<0:
            if swap_dims:
                assert (predicted_3d_pos.shape[0]==1), '{}'.format(predicted_3d_pos.shape)
                target = torch.swapaxes(target, 0, 1)
                predicted_3d_pos = torch.swapaxes(predicted_3d_pos, 0, 1)
                reconstruction = tc_orthographic_projection(predicted_3d_pos, None, None)
            else:
                assert (predicted_3d_pos.shape[1]==1), '{}'.format(predicted_3d_pos.shape)
                reconstruction = tc_orthographic_projection(predicted_3d_pos, projection_mtx_base, homogenous_one_base)
            reconstruction = tc_scale_normalize(reconstruction, target) # scaled to match target_semi (?>,f,17,2)
            if args.projection_type==-2: # move poses to place pelvis at origin
                reconstruction = reconstruction - reconstruction[:,:,[0]]
                target = target - target[:,:,[0]]
        else: return target, target # no 2d-reprojection, so return target as reconstruction to get an error of 0
        if ignore_some_keypoints: # activated for HRNet (data_2d_h36m_hr.npz) without nose kpt
            reconstruction = reconstruction[:,:,kpts_oi_indices]
            target = target[:,:,kpts_oi_indices,:]
        return reconstruction, target

    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        # Added by Lawrence 03/29/2021 - 09/09/21
        epoch_loss_mble_train_unlabeled = 0
        epoch_loss_mce_wcx_train_unlabeled = 0
        epoch_loss_mce_ncx_train_unlabeled = 0
        epoch_loss_pto_train_unlabeled = 0
        epoch_loss_bse_train_unlabeled = 0
        epoch_loss_bpc_train_unlabeled = 0
        epoch_loss_jmc_train_unlabeled = 0
        epoch_loss_pto_train_labeled = 0
        epoch_loss_bse_train_labeled = 0
        epoch_loss_bpc_train_labeled = 0
        epoch_loss_jmc_train_labeled = 0

        N = 0
        N_semi = 0
        N_semi_mv_wcx = 0
        N_semi_mv_ncx = 0
        model_pos_train.train()

        if semi_supervised:
            # Semi-supervised scenario
            if use_traj_model: model_traj_train.train()
            # Begin iteration for 1 epoch -----------------------------------------------------------------------------
            for (_, batch_3d, batch_2d), (tbi_semi, nai_semi, hfi_semi, cam_ex_semi, cam_in_semi, _, batch_2d_semi) \
                    in zip(train_generator.next_epoch(), semi_generator.next_epoch()):

                # Fall back to supervised training for the first few epochs (to avoid instability)
                skip = epoch < args.warmup
                n_full_supervised_iters += 1
                cam_ex_semi = torch.from_numpy(cam_ex_semi.astype('float32'))
                cam_in_semi = torch.from_numpy(cam_in_semi.astype('float32'))
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                if run_on_available_gpu:
                    cam_ex_semi = cam_ex_semi.cuda()
                    cam_in_semi = cam_in_semi.cuda()
                    inputs_3d = inputs_3d.cuda()

                if use_traj_model: inputs_traj = inputs_3d[:, :, :1].clone()
                inputs_3d[:, :, 0] = 0 # Note, affects only 3D supervised branch

                # Split point between labeled and unlabeled samples in the batch
                split_idx = inputs_3d.shape[0]

                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_2d_semi = torch.from_numpy(batch_2d_semi.astype('float32'))
                if run_on_available_gpu:
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_semi = inputs_2d_semi.cuda()
                # Source of overfiting when warmup<epoch and all semi-supervised loss coefficients are 0
                inputs_2d_cat = torch.cat((inputs_2d, inputs_2d_semi), dim=0) if not skip else inputs_2d

                # # visualize orthographic projection
                # if pad > 0: target_2d = inputs_2d[:, pad:-pad, :, :2] # (?>,f,17,2)
                # else: target_2d = inputs_2d[:, :, :, :2] # (?>,f,17,2)
                # reproj_2d = tc_orthographic_projection(inputs_3d, projection_mtx_base, homogenous_one_base)
                # scaled_reproj_2d = tc_scale_normalize(reproj_2d, target_2d)
                # transl_target_2d = target_2d - target_2d[:,:,[0]]
                # for i in range(target_2d.shape[0]):
                #     plot_2d_pose([target_2d[i,0].cpu().numpy(), reproj_2d[i,0].cpu().numpy(),
                #                   scaled_reproj_2d[i,0].cpu().numpy(), transl_target_2d[i,0].cpu().numpy()],
                #                  ['Target-2D','Reproj-2D','Scaled Reproj-2D','Transl. Target-2D'], KPT_2_IDX, overlay=True)
                #     if (i+1)%10==0:
                #         cmd = input("continue? enter 'y' or 'n': ")
                #         if cmd=='n': sys.exit(0)

                optimizer.zero_grad()

                # Computes Supervised Losses: loss_3d_pos & loss_traj
                # with top subset of batch corresponding to x% of --subjects-train (eg. 10% of S1)
                # starting with 3D pose loss
                predicted_3d_pos_cat = model_pos_train(inputs_2d_cat)
                if ignore_some_keypoints: # for 16 keypoints (15 Bones) HRNet pose
                    loss_3d_pos = mpjpe(predicted_3d_pos_cat[:split_idx,:,kpts_oi_indices], inputs_3d[:,:,kpts_oi_indices])
                else: loss_3d_pos = mpjpe(predicted_3d_pos_cat[:split_idx], inputs_3d)
                # todo: undo line above and 2 lines below
                # inputs_3d = torch.cat((inputs_3d, predicted_3d_pos_cat[split_idx:].detach().clone()), dim=0)
                # loss_3d_pos = mpjpe(predicted_3d_pos_cat, inputs_3d, n_samples_oi=split_idx)
                epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                N += inputs_3d.shape[0]*inputs_3d.shape[1]
                loss_total = loss_3d_pos

                # Compute global trajectory
                if use_traj_model:
                    predicted_traj_cat = model_traj_train(inputs_2d_cat)
                    w = 1 / inputs_traj[:, :, :, 2] # Weight inversely proportional to depth
                    loss_traj = weighted_mpjpe(predicted_traj_cat[:split_idx], inputs_traj, w)
                    epoch_loss_traj_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_traj.item()
                    assert inputs_traj.shape[0]*inputs_traj.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]
                    loss_total += loss_traj

                if not skip: # Semi-Supervised
                    # Semi-supervised loss for unlabeled samples
                    # Computes Self-Supervised Losses: Reprojected 2D MPJPE loss and 3D PTO, BSE, BPC, JMC losses
                    # - 1st half of batch corresponding to --subjects-train (labelled subset with 3D gt. Eg. S5,S6)
                    # - 2nd half of batch corresponding to --subjects-unlabeled (subset without 3D gt. Eg. S7,S8)
                    # predicted_3d_pos_cat: (batch, predicted-frames-per-example, kpts, xyz-cord): (128, 1, 17, 3)
                    n_semi_supervised_iters += 1

                    assert (~apply_mv_pose_consistency or len(tbi_semi)>0)
                    assert (len(tbi_semi)==0 or len(nai_semi)+len(hfi_semi)>0)
                    assert (len(tbi_semi)!=0 or len(tbi_semi)+len(nai_semi)+len(hfi_semi)==0)

                    # multi-view pose and trajectory consistency loss
                    if apply_mv_pose_consistency:
                        # separate true-batch (without added multiview poses) from batch superset
                        all_semi_pred_3d_pos = predicted_3d_pos_cat[split_idx:] # (?",f,17,3)
                        semi_pred_3d_pos = all_semi_pred_3d_pos[tbi_semi] # (?>,f,17,3)
                        n_semi_samples, n_semi_frames = semi_pred_3d_pos.shape[:2]
                        if use_traj_model:
                            all_semi_pred_traj = predicted_traj_cat[split_idx:] # (?",f,1,3)
                            semi_pred_traj = all_semi_pred_traj[tbi_semi] # (?>,f,1,3)
                        inputs_2d_semi = inputs_2d_semi[tbi_semi] # (?>,f,17,3)
                        cam_in_semi = cam_in_semi[tbi_semi] # (?>,f,1,3)

                        mv_anchor_samples = (len(nai_semi)+len(hfi_semi))//mcv_cnt
                        #n_semi_mv_batch_frames = mv_anchor_samples * n_semi_frames
                        assert (mv_anchor_samples>0), 'apply_mv_pose_consistency --> {}>0'.format(mv_anchor_samples)

                        # for args.joint_mobility_term and wout_cam_ext_mce_type>=-3
                        #all_semi_pred_3d_pos_mm = all_semi_pred_3d_pos * one_k_tsr # (?",f,17,3) from meters --> millimeters
                        #all_semi_pred_3d_pos_mm *= repose_pelvis_tsr # (?",f,17,3) 0 at kpt index 0, 1 at other kpt indexes
                        all_semi_pred_quadruplet_kpts = all_semi_pred_3d_pos[:,:,quad_kpt_idxs,:] # (?",f,17,3)-->(?",f,16,4,3)

                        if wout_cam_ext_mce_type == -1:
                            all_semi_pred_fb_uvecs, all_semi_pred_fb_vecs = orient_fb_uvec(all_semi_pred_quadruplet_kpts)
                        else: all_semi_pred_fb_uvecs = orient_fb_uvec(all_semi_pred_quadruplet_kpts) # +-> (?",f,16,3)
                        semi_pred_fb_uvecs = all_semi_pred_fb_uvecs[tbi_semi] # (?>,f,12,3)
                        extracted_batch_fb_uvecs = True # todo: replace with apply_mv_pose_consistency as their truth table are equivalent

                        if with_cam_ext_mce_type>0:
                            wcx_mv_semi_pred_3d_pos = all_semi_pred_3d_pos[nai_semi] # (?'*,f,17,3) with-cam-ext
                            # Reposition to pelvis coordinate to origin. todo: deactivate to improve mpjpe?
                            #wcx_mv_semi_pred_3d_pos *= repose_pelvis_tsr # 0 at keypoint index 0, 1 at other keypoint indexes
                            # Transform poses to a common frame (aka world-frame)
                            wcx_mv_semi_pred_traj = all_semi_pred_traj[nai_semi] # (?'*,f,1,3)
                            wcx_mv_cam_ex_semi = torch.reshape(cam_ex_semi[nai_semi], (-1, 1, 1, 7)) # (?'*,7)->(?'*,1,1,7)
                            wcx_mv_cam_ex_semi = torch.tile(wcx_mv_cam_ex_semi, (1, 1, 17, 1)) # (?'*,f,17,3)
                            camfrm_wcx_mv_poses = wcx_mv_semi_pred_3d_pos + wcx_mv_semi_pred_traj # (?'*,f,17,3)
                            wldfrm_wcx_mv_poses = qrot(wcx_mv_cam_ex_semi[:,:,:,:4], camfrm_wcx_mv_poses) \
                                                  + wcx_mv_cam_ex_semi[:,:,:,4:] # (?'*,f,17,3)
                            wldfrm_wcx_mv_poses = wldfrm_wcx_mv_poses.view(-1, mcv_cnt, n_semi_frames, 17, 3) # (?*,m,f,17,3)

                            # # debug visuals
                            # # -------------------------------------------------------------------------------------------
                            # wcx_mv_inputs_3d_semi = inputs_3d_semi[nai_semi] # (?>>,f,17,3)
                            # imgfrm_wcx_mv_poses = torch.clone(wcx_mv_inputs_3d_semi).view(-1, mcv_cnt, n_semi_frames, 17, 3)) # (?>>,m,f,17,3)
                            # wcx_mv_inputs_traj_semi = inputs_traj_semi[nai_semi] # (?>>,f,1,3)
                            # wcx_mv_cam_ex_semi = torch.reshape(cam_ex_semi[nai_semi], (-1, 1, 1, 7)) # (?>>,7)->(?>>,1,1,7)
                            # wcx_mv_cam_ex_semi = torch.tile(wcx_mv_cam_ex_semi, (1, 1, 17, 1)) # (?>>,f,17,3)
                            # camfrm_wcx_mv_poses = wcx_mv_inputs_3d_semi + wcx_mv_inputs_traj_semi # (?>>,f,17,3)
                            # wldfrm_wcx_mv_poses = qrot(wcx_mv_cam_ex_semi[:,:,:,:4], camfrm_wcx_mv_poses) \
                            #                          + wcx_mv_cam_ex_semi[:,:,:,4:] # (?>>,f,17,3)
                            # wldfrm_wcx_mv_poses = torch.reshape(wldfrm_wcx_mv_poses, (-1, mcv_cnt, n_semi_frames, 17, 3)) # (?>>,m,f,17,3)
                            #
                            # n_rows, n_cols = 2, mcv_cnt
                            # for s in range(len(nai_semi)//mcv_cnt):
                            #     SP_FIG, SP_AXS = plt.subplots(n_rows, n_cols, subplot_kw={'projection':'3d'}, figsize=(3*n_cols, 4*n_rows))
                            #     SP_FIG.subplots_adjust(left=0.0, right=1.0, wspace=-0.0)
                            #     for i in range(mcv_cnt):
                            #         plot_3d_pose(to_numpy(imgfrm_wcx_mv_poses[s,i,0])*1000, SP_FIG, SP_AXS[0,i], 'Cam-View {}'.format(i),
                            #                      KPT_2_IDX, [-1]*4)
                            #         plot_3d_pose(to_numpy(wldfrm_wcx_mv_poses[s,i,0])*1000, SP_FIG, SP_AXS[1,i], 'Wld-View {}'.format(i),
                            #                      KPT_2_IDX, [-1]*4, display=i==1)
                            # # -------------------------------------------------------------------------------------------

                            # Translate poses to origin to reduce sensitivity ot trajectory model
                            wldfrm_wcx_mv_poses -= wldfrm_wcx_mv_poses[:,:,:,[0]] # (?*,m,f,17,3)

                            if with_cam_ext_mce_type==1:
                                # Anchor-to-Each-Positive: anchor pose is compared to each positive pose
                                each_ach_2_pos_errs = []
                                for view_idx in range(1, mcv_cnt):
                                    each_ach_2_pos_errs.append(torch.linalg.norm(
                                        wldfrm_wcx_mv_poses[:,0] - wldfrm_wcx_mv_poses[:,view_idx], dim=-1)) # (?*,f,17)
                                agg_kpt_wcx_mce_per_pose = torch.stack(each_ach_2_pos_errs, dim=3) # (m-1)*(?*,f,17)->(?*,f,17,m-1)

                            elif with_cam_ext_mce_type==2:
                                # Adjacent-Pairs-Comparison: pairs of adjacent multi-view poses are compared
                                each_adj_pair_errs = []
                                for view_idx in range(mcv_cnt-1):
                                    each_adj_pair_errs.append(torch.linalg.norm(
                                        wldfrm_wcx_mv_poses[:,view_idx] - wldfrm_wcx_mv_poses[:,view_idx+1], dim=-1)) # (?*,f,17)
                                agg_kpt_wcx_mce_per_pose = torch.stack(each_adj_pair_errs, dim=3) # (m-1)*(?*,f,17)->(?*,f,17,m-1)

                            n_semi_mv_wcx_batch_frames = agg_kpt_wcx_mce_per_pose.shape[0] * n_semi_frames
                            if ignore_some_keypoints: # for 16 keypoints (15 Bones) HRNet pose
                                loss_mce_wcx = torch.mean(agg_kpt_wcx_mce_per_pose[:,:,kpts_oi_indices])
                            else: loss_mce_wcx = torch.mean(agg_kpt_wcx_mce_per_pose)
                            epoch_loss_mce_wcx_train_unlabeled += n_semi_mv_wcx_batch_frames * loss_mce_wcx.item()
                            loss_total += mce_wcx_coef * loss_mce_wcx
                        else: n_semi_mv_wcx_batch_frames = 1 # To avoid division by zero

                        if wout_cam_ext_mce_type<0:
                            if with_cam_ext_mce_type>0: wout_cam_ext_indices = hfi_semi
                            else: wout_cam_ext_indices = nai_semi + hfi_semi
                            nex_mv_semi_pred_3d_pos = all_semi_pred_3d_pos[wout_cam_ext_indices] # (?'^,f,17,3)
                            # Reposition to pelvis coordinate to origin. todo: deactivate to see if it improves estimation
                            nex_mv_semi_pred_3d_pos *= repose_pelvis_tsr # 0 at keypoint index 0, 1 at other keypoint indexes

                            if wout_cam_ext_mce_type>=-3: # i.e. -1, -2 or -3
                                # Compare aligned free-bone vectors
                                # Anchor-to-Each-Positive: anchor pose is compared to each positive pose
                                if wout_cam_ext_mce_type == -1:
                                    nex_mv_free_bone_vecs = all_semi_pred_fb_vecs[wout_cam_ext_indices] # (?'^,f,16,3)
                                else: nex_mv_free_bone_vecs = all_semi_pred_fb_uvecs[wout_cam_ext_indices] # (?'^,f,16,3)

                                plvfrm_ncx_mv_fb_vecs = nex_mv_free_bone_vecs.view(-1, mcv_cnt, n_semi_frames, 16, 3) # (?^,m,f,16,3)
                                each_ach_2_pos_errs = []
                                for view_idx in range(1, mcv_cnt):
                                    each_ach_2_pos_errs.append(torch.linalg.norm(
                                        plvfrm_ncx_mv_fb_vecs[:,0] - plvfrm_ncx_mv_fb_vecs[:,view_idx], dim=-1)) # (?^,f,16)
                                agg_kpt_ncx_mce_per_pose = torch.stack(each_ach_2_pos_errs, dim=3) # (m-1)*(?^,f,16)->(?^,f,16,m-1)
                                if ignore_some_keypoints: # for 16 keypoints (15 Bones) HRNet pose
                                    agg_kpt_ncx_mce_per_pose = agg_kpt_ncx_mce_per_pose[:,:,fbones_oi_indices]

                                if wout_cam_ext_mce_type==-1:
                                    loss_mce_ncx = torch.mean(agg_kpt_ncx_mce_per_pose) # /one_k_tsr) # fb-vecs from mm -> meters

                                elif wout_cam_ext_mce_type<=-2:
                                    loss_mce_ncx = torch.mean(agg_kpt_ncx_mce_per_pose) # converting <=2 mm to meters is too small
                                    if wout_cam_ext_mce_type==-3:
                                        # also compare bone lengths of multi-view poses
                                        nex_mv_bone_dists = nex_mv_semi_pred_3d_pos[:,:,1:] - \
                                                            nex_mv_semi_pred_3d_pos[:,:,dataset.skeleton().parents()[1:]] # (?'^,f,16,3)
                                        nex_mv_bone_lens = torch.linalg.norm(nex_mv_bone_dists, dim=3)  # (?'^,f,16,3)->(?'^,f,16)
                                        per_ncx_mv_bone_lens = nex_mv_bone_lens.view(-1, mcv_cnt, n_semi_frames, 16) # (?^,m,f,16)
                                        each_ach_2_pos_dist = []
                                        for view_idx in range(1, mcv_cnt):
                                            each_ach_2_pos_dist.append(torch.abs(per_ncx_mv_bone_lens[:,0] -
                                                                                 per_ncx_mv_bone_lens[:,view_idx])) # (?^,f,16)
                                        agg_kpt_blen_dist_err = torch.stack(each_ach_2_pos_dist, dim=3) # (m-1)*(?^,f,16) -> (?^,f,16,m-1)
                                        loss_mce_ncx += torch.mean(agg_kpt_blen_dist_err)

                            elif wout_cam_ext_mce_type==-4:
                                # MPCE from Procrustes Rigid Aligned Pose
                                plvfrm_ncx_mv_poses = nex_mv_semi_pred_3d_pos.view(-1, mcv_cnt, n_semi_frames, 17, 3) # (?^,m,f,17,3)
                                each_ach_2_pos_errs = []
                                for view_idx in range(mcv_cnt-1):
                                    each_ach_2_pos_errs.append(torch_p_mpjpe(
                                        plvfrm_ncx_mv_poses[0, view_idx].view(-1, 17, 3),
                                        plvfrm_ncx_mv_poses[0, view_idx+1].view(-1, 17, 3)))
                                agg_kpt_ncx_mce_per_pose = torch.stack(each_ach_2_pos_errs, dim=2) # (m-1)*(?^,17) -> (?^,17,m-1)
                                if ignore_some_keypoints: # for 16 keypoints (15 Bones) HRNet pose
                                    loss_mce_ncx = torch.mean(agg_kpt_ncx_mce_per_pose[:,kpts_oi_indices])
                                else: loss_mce_ncx = torch.mean(agg_kpt_ncx_mce_per_pose)

                            n_semi_mv_ncx_batch_frames = agg_kpt_ncx_mce_per_pose.shape[0] * n_semi_frames
                            epoch_loss_mce_ncx_train_unlabeled += n_semi_mv_ncx_batch_frames * loss_mce_ncx.item()
                            loss_total += mce_ncx_coef * loss_mce_ncx
                        else: n_semi_mv_ncx_batch_frames = 1 # To avoid division by zero
                    else:
                        semi_pred_3d_pos = predicted_3d_pos_cat[split_idx:] # (?>,f,17,3)
                        n_semi_samples, n_semi_frames = semi_pred_3d_pos.shape[:2]
                        if use_traj_model: semi_pred_traj = predicted_traj_cat[split_idx:] # (?>,f,1,3)
                        extracted_batch_fb_uvecs = False
                        n_semi_mv_wcx_batch_frames = 1 # To avoid division by zero
                        n_semi_mv_ncx_batch_frames = 1 # To avoid division by zero
                    n_semi_batch_frames = n_semi_samples * n_semi_frames

                    # 2D MPJPE loss
                    if args.projection_type!=0:
                        #assert(False), 'args.projection_type:{}'.format(args.projection_type)
                        if pad > 0:
                            target_semi = inputs_2d_semi[:, pad:-pad, :, :2].contiguous() # (?>,f,17,2)
                        else: target_semi = inputs_2d_semi[:, :, :, :2].contiguous() # (?>,f,17,2)
                        if args.projection_type>0:
                            reconstruction_semi = projection_func(semi_pred_3d_pos+semi_pred_traj, cam_in_semi) # (?>,f,17,2)
                        else: # args.projection_type<0:
                            reconstruction_semi = tc_orthographic_projection(
                                semi_pred_3d_pos, projection_mtx_base, homogenous_one_base) # (?>,f,17,2)
                            reconstruction_semi = tc_scale_normalize(reconstruction_semi, target_semi) # scaled to match target_semi
                            if args.projection_type==-2: # move poses to place pelvis at origin
                                reconstruction_semi = reconstruction_semi - reconstruction_semi[:,:,[0]]
                                target_semi = target_semi - target_semi[:,:,[0]]
                        if ignore_some_keypoints: # for 16 keypoints (15 Bones) HRNet pose
                            reconstruction_semi = reconstruction_semi[:,:,kpts_oi_indices]
                            target_semi = target_semi[:,:,kpts_oi_indices]
                        loss_reconstruction = mpjpe(reconstruction_semi, target_semi) # On 2D poses
                        epoch_loss_2d_train_unlabeled += n_semi_batch_frames * loss_reconstruction.item()
                        loss_total += rp2d_coef * loss_reconstruction

                    # Bone Length Error (BLE) term to enforce kinematic constraints
                    if args.mean_bone_length_term:
                        dists = predicted_3d_pos_cat[:, :, 1:] - \
                                predicted_3d_pos_cat[:, :, dataset.skeleton().parents()[1:]] # -->(?,f,16,3)
                        avg_frm_bone_lens = torch.mean(torch.linalg.norm(dists, dim=3), dim=1) # (?,f,16,3)->(?,f,16)->(?,16)
                        penalty = torch.mean(torch.abs(torch.mean(avg_frm_bone_lens[:split_idx], dim=0) -
                                                       torch.mean(avg_frm_bone_lens[split_idx:], dim=0))) # (?,16)->(16,)->(1,)
                        epoch_loss_mble_train_unlabeled += n_semi_batch_frames * penalty.item()
                        loss_total += mble_coef * penalty

                    # 3D Pose Pelvis-to-Origin (PTO) loss term
                    if args.pelvis_placement_term:
                        loss_pto = mpjpe(semi_pred_3d_pos[:,:,[0]], expected_pelvis_pos, shapeMatch=False)
                        epoch_loss_pto_train_unlabeled += n_semi_batch_frames * loss_pto.item()
                        loss_total += pto_coef * loss_pto

                    if args.reposition_to_origin and \
                            (args.bone_symmetry_term or args.bone_proportion_term or args.joint_mobility_term):
                        # re-orient pelvis to origin, do so after computing PTO and before computing BPC & JMC
                        semi_pred_3d_pos *= repose_pelvis_tsr # 0 at keypoint index 0, 1 at other keypoint indexes

                    if args.bone_symmetry_term or args.bone_proportion_term:
                        semi_dists = semi_pred_3d_pos[:, :, BONE_CHILD_KPTS_IDXS] - \
                                     semi_pred_3d_pos[:, :, BONE_PARENT_KPTS_IDXS] # -->(?>,f,16,3)
                        semi_bone_lengths = torch.linalg.norm(semi_dists, dim=3)  # (?>,f,16,3) --> (?>,f,16)
                        #assert(torch.all(torch.ge(semi_bone_lengths, 0.))), "shouldn't be negative {}".format(semi_bone_lengths)

                    # 3D Bone Symmetry Error (BSE) Loss
                    if args.bone_symmetry_term:
                        semi_rgt_sym_bone_lengths = semi_bone_lengths[:, :, RGT_SYM_BONE_INDEXES] # (?>,f,6)
                        semi_lft_sym_bone_lengths = semi_bone_lengths[:, :, LFT_SYM_BONE_INDEXES] # (?>,f,6)
                        semi_sym_bones_difference = semi_rgt_sym_bone_lengths - semi_lft_sym_bone_lengths  # (?>,f,6)
                        loss_bse = torch.mean(torch.abs(semi_sym_bones_difference))  # (?>,f,6)->(1,)
                        epoch_loss_bse_train_unlabeled += n_semi_batch_frames * loss_bse.item()
                        loss_total += bse_coef * loss_bse

                    # 3D Bone Proportion Constraint (BPC) Regularizer
                    if args.bone_proportion_term:
                        semi_ratio_prop = guard_div(semi_bone_lengths[:,:,bone_prop_pair_idxs_A],
                                                    semi_bone_lengths[:,:,bone_prop_pair_idxs_B]) # (?>,f,15)
                        bpc_likelihood = bpc_likelihood_func(semi_ratio_prop, bpc_variance, bpc_exponent_coefs, bpc_mean)
                        bpc_clamped_likelihood = torch.clamp(bpc_likelihood + log_epsilon, min=1e-08)
                        bpc_actv_likelihood = -bpc_loglihood_wgts * torch.log(bpc_clamped_likelihood)
                        loss_bpc = torch.mean(bpc_actv_likelihood)
                        # loss_bpc = torch.mean(bpc_log_lihood[split_idx:]) # (?,f,15)->(?>,f,15)->(1,)
                        epoch_loss_bpc_train_unlabeled += n_semi_batch_frames * loss_bpc.item()
                        loss_total += bpc_coef * loss_bpc

                    # 3D Joint Mobility Constraint (JMC) or Bone Orientation Constraint (BOC) Regularizer
                    if args.joint_mobility_term:
                        # Adjust pose for JMC
                        if not extracted_batch_fb_uvecs:
                            #semi_pred_3d_pos_mm = semi_pred_3d_pos * one_k_tsr # (?>,f,17,3) from meters --> millimeters
                            #semi_pred_3d_pos_mm *= repose_pelvis_tsr # (?>,f,17,3) 0 at kpt index 0, 1 at other kpt indexes
                            semi_pred_quadruplet_kpts = semi_pred_3d_pos[:,:,quad_kpt_idxs,:] # (?>,f,17,3)-->(?>,f,16,4,3)
                            semi_pred_fb_uvecs = orient_fb_uvec(semi_pred_quadruplet_kpts) # (?>,f,16,3)

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
                        jmc_clamped_likelihood = torch.clamp(jmc_likelihood + log_epsilon, min=1e-08)
                        jmc_actv_likelihood = -jmc_loglihood_wgts * torch.log(jmc_clamped_likelihood)
                        loss_jmc = torch.mean(jmc_actv_likelihood)
                        epoch_loss_jmc_train_unlabeled += n_semi_batch_frames * loss_jmc.item()
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
            # End iteration for 1 epoch -------------------------------------------------------------------------------

            # log mean of certain metrics wrt. all batches & frames per epoch
            losses_traj_train.append(epoch_loss_traj_train / N)
            losses_2d_train_unlabeled.append(epoch_loss_2d_train_unlabeled / N_semi)
            losses_mble_train_unlabeled.append(epoch_loss_mble_train_unlabeled / N_semi)
            losses_mce_wcx_train_unlabeled.append(epoch_loss_mce_wcx_train_unlabeled / N_semi_mv_wcx)
            losses_mce_ncx_train_unlabeled.append(epoch_loss_mce_ncx_train_unlabeled / N_semi_mv_ncx)
            losses_pto_train_unlabeled.append(epoch_loss_pto_train_unlabeled / N_semi)
            losses_bse_train_unlabeled.append(epoch_loss_bse_train_unlabeled / N_semi)
            losses_bpc_train_unlabeled.append(epoch_loss_bpc_train_unlabeled / N_semi)
            losses_jmc_train_unlabeled.append(epoch_loss_jmc_train_unlabeled / N_semi)

        else: # Fully Supervised
            # Regular supervised scenario
            for _, batch_3d, batch_2d in train_generator.next_epoch():
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if run_on_available_gpu:
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                inputs_3d[:, :, 0] = 0

                optimizer.zero_grad()

                # Predict 3D poses
                predicted_3d_pos = model_pos_train(inputs_2d)
                loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                N += inputs_3d.shape[0]*inputs_3d.shape[1]

                loss_total = loss_3d_pos
                loss_total.backward()

                optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict())
            model_pos.eval()
            if semi_supervised and use_traj_model:
                model_traj.load_state_dict(model_traj_train.state_dict())
                model_traj.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            N = 0

            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if run_on_available_gpu:
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    if use_traj_model: inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # Predict 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    if ignore_some_keypoints:
                        # activated for HRNet (data_2d_h36m_hr.npz) without nose kpt
                        inputs_3d = inputs_3d[:,:,kpts_oi_indices,:]
                        predicted_3d_pos = predicted_3d_pos[:,:,kpts_oi_indices,:]
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                    if semi_supervised:
                        cam = torch.from_numpy(cam.astype('float32'))
                        if run_on_available_gpu:
                            cam = cam.cuda()

                        if use_traj_model:
                            predicted_traj = model_traj(inputs_2d)
                            loss_traj = mpjpe(predicted_traj, inputs_traj)
                            epoch_loss_traj_valid += inputs_traj.shape[0]*inputs_traj.shape[1] * loss_traj.item()
                            assert inputs_traj.shape[0]*inputs_traj.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]
                        else: predicted_traj = None

                        if pad > 0:
                            target = inputs_2d[:, pad:-pad, :, :2].contiguous()
                        else: target = inputs_2d[:, :, :, :2].contiguous()
                        reconstruction, target = project2d_reconstruction(predicted_3d_pos, predicted_traj, cam, target, swap_dims=True)
                        loss_reconstruction = mpjpe(reconstruction, target) # On 2D poses
                        epoch_loss_2d_valid += reconstruction.shape[0]*reconstruction.shape[1] * loss_reconstruction.item()
                        assert reconstruction.shape[0]*reconstruction.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)
                if semi_supervised:
                    losses_traj_valid.append(epoch_loss_traj_valid / N)
                    losses_2d_valid.append(epoch_loss_2d_valid / N)


                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if run_on_available_gpu:
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    if use_traj_model: inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                    if semi_supervised:
                        cam = torch.from_numpy(cam.astype('float32'))
                        if run_on_available_gpu:
                            cam = cam.cuda()
                        if use_traj_model:
                            predicted_traj = model_traj(inputs_2d)
                            loss_traj = mpjpe(predicted_traj, inputs_traj)
                            epoch_loss_traj_train_eval += inputs_traj.shape[0]*inputs_traj.shape[1] * loss_traj.item()
                            assert inputs_traj.shape[0]*inputs_traj.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]
                        else: predicted_traj = None

                        if pad > 0:
                            target = inputs_2d[:, pad:-pad, :, :2].contiguous()
                        else: target = inputs_2d[:, :, :, :2].contiguous()
                        reconstruction, target = project2d_reconstruction(predicted_3d_pos, predicted_traj, cam, target, swap_dims=True)
                        loss_reconstruction = mpjpe(reconstruction, target)
                        epoch_loss_2d_train_labeled_eval += reconstruction.shape[0]*reconstruction.shape[1] * loss_reconstruction.item()
                        assert reconstruction.shape[0]*reconstruction.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)
                if semi_supervised:
                    losses_traj_train_eval.append(epoch_loss_traj_train_eval / N)
                    losses_2d_train_labeled_eval.append(epoch_loss_2d_train_labeled_eval / N)

                # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                epoch_loss_2d_train_unlabeled_eval = 0
                N_semi = 0
                if semi_supervised:
                    for cam, _, batch_2d in semi_generator_eval.next_epoch():
                        cam = torch.from_numpy(cam.astype('float32'))
                        inputs_2d_semi = torch.from_numpy(batch_2d.astype('float32'))
                        if run_on_available_gpu:
                            cam = cam.cuda()
                            inputs_2d_semi = inputs_2d_semi.cuda()

                        predicted_3d_pos_semi = model_pos(inputs_2d_semi)
                        predicted_traj_semi = model_traj(inputs_2d_semi) if use_traj_model else None
                        if pad > 0:
                            target_semi = inputs_2d_semi[:, pad:-pad, :, :2].contiguous()
                        else: target_semi = inputs_2d_semi[:, :, :, :2].contiguous()
                        reconstruction_semi, target_semi = \
                            project2d_reconstruction(predicted_3d_pos_semi, predicted_traj_semi, cam, target_semi, swap_dims=True)
                        loss_reconstruction_semi = mpjpe(reconstruction_semi, target_semi)

                        epoch_loss_2d_train_unlabeled_eval += reconstruction_semi.shape[0]*reconstruction_semi.shape[1] \
                                                              * loss_reconstruction_semi.item()
                        N_semi += reconstruction_semi.shape[0]*reconstruction_semi.shape[1]
                    losses_2d_train_unlabeled_eval.append(epoch_loss_2d_train_unlabeled_eval / N_semi)

        elapsed = (time() - start_time)/60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (epoch+1, elapsed, lr, losses_3d_train[-1]*1000))
        else:
            if semi_supervised:
                mean_loss_sup.append(losses_3d_train[-1] + losses_traj_train[-1])
                mean_weighted_loss_sup.append(losses_3d_train[-1] + losses_traj_train[-1])
                mean_loss_sesup.append(losses_2d_train_unlabeled[-1] + losses_mble_train_unlabeled[-1] +
                                       losses_pto_train_unlabeled[-1] + losses_bse_train_unlabeled[-1] +
                                       losses_bpc_train_unlabeled[-1] + losses_jmc_train_unlabeled[-1] +
                                       losses_mce_wcx_train_unlabeled[-1] + losses_mce_ncx_train_unlabeled[-1])
                mean_weighted_loss_sesup.append(rp2d_coef*losses_2d_train_unlabeled[-1] + mble_coef*losses_mble_train_unlabeled[-1] +
                                                pto_coef*losses_pto_train_unlabeled[-1] + bse_coef*losses_bse_train_unlabeled[-1] +
                                                bpc_coef*losses_bpc_train_unlabeled[-1] + jmc_coef*losses_jmc_train_unlabeled[-1] +
                                    mce_wcx_coef*losses_mce_wcx_train_unlabeled[-1] + mce_ncx_coef*losses_mce_ncx_train_unlabeled[-1])
                print('[{:>3}] {:.2f} mins n_sup_iter:{:<6} n_uns_iter:{:<6} lr:{:.2e}\n      ' 
                      '3dps_sup_trn:{:>6.4f} 3dps_mm_trn:{:>6.2f} 3dps_mm_eva:{:>6.2f} 3dps_mm_val:{:>6.2f}\n      '
                      'traj_sup_trn:{:>6.4f} traj_mm_trn:{:>6.2f} traj_mm_eva:{:>6.2f} traj_mm_val:{:>6.2f}\n      '
                      'r2de_uns_trn:{:>6.4f} r2d_sup_eva:{:>6.4f} r2d_uns_eva:{:>6.4f} 2d_pose_val:{:>6.4f} mbl_uns_trn:{:>6.4f}\n      ' 
                      'mce_wcx_utrn:{:>6.4f} pto_uns_trn:{:>6.4f} bse_uns_trn:{:>6.4f} bpc_uns_trn:{:>6.3f} jmc_uns_trn:{:>6.3f}\n      '
                      'mce_ncx_utrn:{:>6.4f} mea_sup_trn:{:>6.4f} mea_uns_trn:{:>6.3f} m&w_sup_trn:{:>6.4f} m&w_uns_trn:{:>6.4f}'
                      .format(epoch+1, elapsed, n_full_supervised_iters, n_semi_supervised_iters, lr,
                              # 3d-pose
                              losses_3d_train[-1],
                              losses_3d_train[-1] * 1000,
                              losses_3d_train_eval[-1] * 1000,
                              losses_3d_valid[-1] * 1000,
                              # 3d-trajectory
                              losses_traj_train[-1],
                              losses_traj_train[-1] * 1000,
                              losses_traj_train_eval[-1] * 1000,
                              losses_traj_valid[-1] * 1000,
                              # 2d-pose (back-projected)
                              losses_2d_train_unlabeled[-1],
                              losses_2d_valid[-1],
                              losses_2d_train_labeled_eval[-1],
                              losses_2d_train_unlabeled_eval[-1],
                              # other self-supervised-branch loss
                              losses_mble_train_unlabeled[-1],
                              losses_mce_wcx_train_unlabeled[-1],
                              losses_pto_train_unlabeled[-1],
                              losses_bse_train_unlabeled[-1],
                              losses_bpc_train_unlabeled[-1],
                              losses_jmc_train_unlabeled[-1],
                              losses_mce_ncx_train_unlabeled[-1],
                              # loss summary
                              mean_loss_sup[-1],
                              mean_loss_sesup[-1],
                              mean_weighted_loss_sup[-1],
                              mean_weighted_loss_sesup[-1]))
                              #mean_weighted_loss_sup[-1] + mean_weighted_loss_sesup[-1]))
            else:
                print('[{:>3}] time:{:.2f} mins n_sup_iters:{:<6} lr:{:.7f}\n      '
                      '3dps_sup_trn:{:>6.4f} 3dps_mm_trn:{:>6.2f} 3dps_mm_eva:{:>6.2f} 3dps_mm_val:{:>6.2f}'
                    .format(epoch+1, elapsed, n_full_supervised_iters, lr,
                            # 3d-pose
                            losses_3d_train[-1],
                            losses_3d_train[-1] * 1000,
                            losses_3d_train_eval[-1] * 1000,
                            losses_3d_valid[-1] * 1000))

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        epoch += 1

        # Decay BatchNorm momentum (Edited by Lawrence 03/25/2021)
        momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        if torch.cuda.device_count()>1 and args.multi_gpu_training:
            model_pos_train.module.set_bn_momentum(momentum)
        else:
            model_pos_train.set_bn_momentum(momentum)
        if semi_supervised and use_traj_model:
            if torch.cuda.device_count()>1 and args.multi_gpu_training:
                model_traj_train.module.set_bn_momentum(momentum)
            else:
                model_traj_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_traj': model_traj_train.state_dict() if semi_supervised and use_traj_model else None,
                'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch>last_warmup_epoch and (epoch%plot_log_freq==0 or epoch==args.epochs):
            assert (False)
            print('\t logging plots and/or ndarrays after epoch {}'.format(epoch))
            if 'matplotlib' not in sys.modules:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(last_warmup_epoch, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[last_warmup_epoch:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[last_warmup_epoch:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[last_warmup_epoch:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epochs')
            plt.xlim((last_warmup_epoch, epoch))
            plt.title('3D-Pose Errors')
            plt.savefig(os.path.join(exp_dir, 'loss_3d.png'))

            if semi_supervised:
                plt.figure()
                plt.plot(epoch_x, losses_traj_train[last_warmup_epoch:], '--', color='C0')
                plt.plot(epoch_x, losses_traj_train_eval[last_warmup_epoch:], color='C0')
                plt.plot(epoch_x, losses_traj_valid[last_warmup_epoch:], color='C1')
                plt.legend(['traj. train', 'traj. train (eval)', 'traj. valid (eval)'])
                plt.ylabel('Mean distance (m)')
                plt.xlabel('Epochs')
                plt.xlim((last_warmup_epoch, epoch))
                plt.title('3D-Trajectory Errors')
                plt.savefig(os.path.join(exp_dir, 'loss_traj.png'))

                plt.figure()
                plt.plot(epoch_x, losses_2d_train_labeled_eval[last_warmup_epoch:], color='C0')
                plt.plot(epoch_x, losses_2d_train_unlabeled[last_warmup_epoch:], '--', color='C1')
                plt.plot(epoch_x, losses_2d_train_unlabeled_eval[last_warmup_epoch:], color='C1')
                plt.plot(epoch_x, losses_2d_valid[last_warmup_epoch:], color='C2')
                plt.legend(['2d train labeled (eval)', '2d train unlabeled', '2d train unlabeled (eval)', '2d valid (eval)'])
                plt.ylabel('MPJPE (2D)')
                plt.xlabel('Epochs')
                plt.xlim((last_warmup_epoch, epoch))
                plt.title('2D-Pose Errors')
                plt.savefig(os.path.join(exp_dir, 'loss_2d.png'))

                # Added by Lawrence 09/09/2021
                sps_fig, sps_axs = plt.subplots(nrows=1, ncols=2, num='SeSupervised', figsize=(20, 7))
                plt.figure('SeSupervised')
                sps_fig.suptitle('Self-Supervised Branch Loss Terms', fontweight='bold', size=12)
                sps_axs[0].clear()
                sps_axs[0].plot(epoch_x, losses_2d_train_unlabeled[last_warmup_epoch:], color='C0')
                sps_axs[0].plot(epoch_x, losses_mble_train_unlabeled[last_warmup_epoch:], '--', color='C3')
                sps_axs[0].plot(epoch_x, losses_bse_train_unlabeled[last_warmup_epoch:], color='C1')
                sps_axs[0].plot(epoch_x, losses_pto_train_unlabeled[last_warmup_epoch:], color='C5')
                sps_axs[0].legend(['2DP', 'BLE', 'BSE', 'PTO'])
                sps_axs[0].set(title='SOTA Loss Terms', xlabel='Epochs', ylabel='Avg. Weighted Loss')
                sps_axs[1].clear()
                sps_axs[1].plot(epoch_x, losses_bpc_train_unlabeled[last_warmup_epoch:], color='C2')
                sps_axs[1].plot(epoch_x, losses_jmc_train_unlabeled[last_warmup_epoch:], color='C4')
                sps_axs[1].legend(['BPC', 'JMC'])
                sps_axs[1].set(title='Proposed Loss Regularizer Terms', xlabel='Epochs', ylabel='Avg. Weighted Loss')
                plt.savefig(os.path.join(exp_dir, 'loss_sesup.png'))

                sps_fig, sps_axs = plt.subplots(nrows=1, ncols=2, num='SesupVsSup', figsize=(20, 7))
                plt.figure('SesupVsSup')
                sps_fig.suptitle('Supervised & Self-Supervised Branch Combined Losses', fontweight='bold', size=12)
                sps_axs[0].clear()
                sps_axs[0].plot(epoch_x, mean_loss_sup[last_warmup_epoch:], '--', color='C0')
                sps_axs[0].plot(epoch_x, mean_weighted_loss_sup[last_warmup_epoch:], color='C0')
                sps_axs[0].plot(epoch_x, mean_weighted_loss_sesup[last_warmup_epoch:], color='C1')
                sps_axs[0].legend(['Sup.', 'Wgt. Sup.', 'Wgt. Self-Sup.'])
                sps_axs[0].set(xlabel='Epochs', ylabel='Avg. Loss')
                sps_axs[1].clear()
                sps_axs[1].plot(epoch_x, mean_loss_sesup[last_warmup_epoch:], '--', color='C1')
                sps_axs[1].legend(['Self-Sup.'])
                sps_axs[1].set(xlabel='Epochs', ylabel='Avg. Loss')
                plt.savefig(os.path.join(exp_dir, 'loss_sup_sesup.png'))

            plt.close('all')

        # Save loss values (Added by Lawrence 09/09/21)
        if args.log_losses_numpy:
            if semi_supervised:
                per_epoch_avg_losses = \
                    np.stack([losses_traj_train,
                              losses_3d_train, losses_3d_train_eval, losses_3d_valid,
                              losses_2d_train_unlabeled, losses_mble_train_unlabeled,
                              losses_bse_train_unlabeled, losses_bpc_train_unlabeled,
                              losses_jmc_train_unlabeled, losses_pto_train_unlabeled])
            else:
                per_epoch_avg_losses = \
                    np.stack([losses_3d_train, losses_3d_train_eval, losses_3d_valid,
                              losses_bse_train_labeled, losses_bpc_train_labeled,
                              losses_jmc_train_labeled, losses_pto_train_labeled])
            np.save(os.path.join(exp_dir, 'per_epoch_avg_losses.npy'), per_epoch_avg_losses.astype(np.float32))

        # Save final model
        if args.save_model:
            chk_path = os.path.join(exp_dir, 'epoch_{}.bin'.format(epoch))
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        # Added by Lawrence 05/05/20 to clear cache after every batch
        #gc.collect() # TODO: DISABLE!! 07/05/2022
        #torch.cuda.empty_cache() # TODO: DISABLE!! 07/05/2022


# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        else:
            model_traj.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if run_on_available_gpu:
                inputs_2d = inputs_2d.cuda()

            # Positional model
            if not use_trajectory_model:
                predicted_3d_pos = model_pos(inputs_2d)
            else:
                predicted_3d_pos = model_traj(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # TODO Note: Eval with test-time augmentation is NOT EXACT, it's avg of non-flip & reverse of pred flip
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                if not use_trajectory_model:
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if run_on_available_gpu:
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            if ignore_some_keypoints:
                # activated for HRNet (data_2d_h36m_hr.npz) without nose kpt
                inputs_3d = inputs_3d[:,:,kpts_oi_indices,:]
                predicted_3d_pos = predicted_3d_pos[:,:,kpts_oi_indices,:]

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error   (MPJPE): {:>5.2f} mm'.format(e1))
    print('Protocol #2 Error (P-MPJPE): {:>5.2f} mm'.format(e2))
    print('Protocol #3 Error (N-MPJPE): {:>5.2f} mm'.format(e3))
    print('Velocity    Error   (MPJVE): {:>5.2f} mm'.format(ev))
    print('----------')

    return e1, e2, e3, ev


if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator(None, None, [input_keypoints], pad=pad, causal_shift=causal_shift,
                             augment=args.test_time_augmentation, kps_left=kps_left, kps_right=kps_right,
                             joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)
    if model_traj is not None and ground_truth is None:
        prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
        prediction += prediction_traj

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

else:
    print('Evaluating...')
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

    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)): # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_poses_3d, out_poses_2d

    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
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
                                     augment=args.test_time_augmentation, kps_left=kps_left, kps_right=kps_right,
                                     joints_left=joints_left, joints_right=joints_right)
            e1, e2, e3, ev = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        print('Protocol #1   (MPJPE) action-wise average: {:>5.2f} mm'.format(round(np.mean(errors_p1), 1)))
        print('Protocol #2 (P-MPJPE) action-wise average: {:>5.2f} mm'.format(round(np.mean(errors_p2), 1)))
        print('Protocol #3 (N-MPJPE) action-wise average: {:>5.2f} mm'.format(round(np.mean(errors_p3), 1)))
        print('Velocity      (MPJVE) action-wise average: {:>5.2f} mm'.format(round(np.mean(errors_vel), 2)))

    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')