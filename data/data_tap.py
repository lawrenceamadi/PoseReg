# -*- coding: utf-8 -*-
# @Time    : 3/1/2023 5:58 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : data_tap.py
# @Software: pose.reg


import numpy as np

from agents.helper import *
from common.utils import deterministic_random
from agents.visuals import plot_3d_pose, plot_2d_pose


def fetch(keypoints, gt2d_keypoints, dataset, subjects, action_filter=None,
          downsample=1, subset=1, parse_3d_poses=True, subset_type='Unkown', visualize_examples=False):
    in_poses_2d = []  # corresponding to estimated 2D poses
    gt_poses_2d = [] # corresponding to ground-truth 2D poses
    out_poses_3d = []
    out_cams_intrinsic = [] # Added by Lawrence (03/14/22)
    out_cams_extrinsic = []
    info_table_log = {} # key->subject, value->actions_cnt_dict
    session_ids = [] # id: subject_action
    ignore_nose_kpt = False

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
            poses_2d_est = keypoints[subject][action] # c*(?,j,2)
            poses_2d_gt = gt2d_keypoints[subject][action] # c*(?,j,2)
            sa_n_cam_views = len(poses_2d_est)
            assert (sa_n_cam_views==4), "We assume all subject's actions have 4 views not {}".format(sa_n_cam_views)

            for i in range(sa_n_cam_views): # Iterate across cameras
                assert (poses_2d_est[i].shape[0]==poses_2d_gt[i].shape[0]), '{} vs. {}'.format(poses_2d_est[i].shape, poses_2d_gt[i].shape)
                if poses_2d_est[i].shape[1] == 16: # characteristic of data_2d_h36m_hr.npz HRNet
                    ignore_nose_kpt = True
                    poses_2d_est[i] = add_estimated_nose_kpt(poses_2d_est[i])
                in_poses_2d.append(poses_2d_est[i])
                gt_poses_2d.append(poses_2d_gt[i])
                sub_act_cnt += poses_2d_est[i].shape[0] # shape->(varying # of frames between 1-5k, joints:17, [x,y]:2)
                session_ids.append((subject, action_name))
                if visualize_examples:
                    j = np.random.randint(0, poses_2d_est[i].shape[0])
                    plot_2d_pose(poses_2d_gt[i][[j]], 'GT. {} [{}] - Seq:{} Frm:{}'.format(subject, action, i, j), KPT_2_IDX)
                    plot_2d_pose(poses_2d_est[i][[j]], 'Est. {} [{}] - Seq:{} Frm:{}'.format(subject, action, i, j), KPT_2_IDX)

            # log number of frames for action
            actions_cnt[action_name] = actions_cnt.get(action_name, 0) + sub_act_cnt
            subject_cnt += sub_act_cnt

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert (len(cams) == sa_n_cam_views), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam: out_cams_intrinsic.append(cam['intrinsic'])
                    if 'extrinsic' in cam: out_cams_extrinsic.append(cam['extrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert (len(poses_3d) == sa_n_cam_views), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across multi-view poses
                    out_poses_3d.append(poses_3d[i]) # (?,j,3)

        actions_cnt['All-Acts'] = subject_cnt
        info_table_log[subject] = actions_cnt
    if subset>=1:
        info_table = display_subject_actions_frame_counts(info_table_log, sum_rows=len(subjects)>1)
        print('[INFO] {} Data (With {:,} Subjects-&-Actions Sessions)\n{}'.
              format(subset_type, len(in_poses_2d), info_table))

    if len(out_cams_intrinsic)==0: out_cams_intrinsic = None
    if len(out_cams_extrinsic)==0: out_cams_extrinsic = None
    if len(out_poses_3d)==0: out_poses_3d = None

    stride = downsample
    if subset < 1:
        subject_id = ''
        subject_cnt = 0
        actions_cnt = {}
        for i in range(len(in_poses_2d)):
            n_frames = int(round(len(in_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(in_poses_2d[i]) - n_frames + 1, str(len(in_poses_2d[i])))
            in_poses_2d[i] = in_poses_2d[i][start:start+n_frames:stride]
            gt_poses_2d[i] = gt_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
            # log reduced frame count
            sub_act_cnt = len(in_poses_2d[i])
            subject, action_name = session_ids[i]
            if i==0: subject_id = '{:.0%}{}\nds-{}'.format(subset, subject, stride)
            actions_cnt[action_name] = actions_cnt.get(action_name, 0) + sub_act_cnt
            subject_cnt += sub_act_cnt
        # display table
        actions_cnt['All-Acts'] = subject_cnt
        info_table_log[subject_id] = actions_cnt
        info_table = display_subject_actions_frame_counts(info_table_log, sum_rows=False, table_style='grid')
        print('[INFO] {} Data (With {:,} Subjects-&-Actions Sessions)\n{}'.
              format(subset_type, len(in_poses_2d), info_table))
    elif stride > 1:
        # Downsample as requested
        info_table_log = {} # key->subject, value->actions_cnt_dict
        subjects_frm_cnt = {} # key->subject, value->total-frames
        for i in range(len(in_poses_2d)):
            in_poses_2d[i] = in_poses_2d[i][::stride]
            gt_poses_2d[i] = gt_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
            # log reduced frame cnt
            subject, action_name = session_ids[i]
            subject = '{} ds={}'.format(subject, stride)
            if info_table_log.get(subject)==None: info_table_log[subject] = {}
            sub_act_cnt = len(in_poses_2d[i])
            info_table_log[subject][action_name] = info_table_log[subject].get(action_name, 0) + sub_act_cnt
            subjects_frm_cnt[subject] = subjects_frm_cnt.get(subject, 0) + sub_act_cnt
        # display table
        for subject in subjects:
            subject = '{} ds={}'.format(subject, stride)
            info_table_log[subject]['All-Acts'] = subjects_frm_cnt[subject]
        info_table = display_subject_actions_frame_counts(info_table_log, sum_rows=len(subjects)>1)
        print('[INFO] Down-sampled {} Data (With {:,} Subjects-&-Actions Sessions)\n{}'.
              format(subset_type, len(in_poses_2d), info_table))

    return out_cams_intrinsic, out_cams_extrinsic, out_poses_3d, gt_poses_2d, in_poses_2d, ignore_nose_kpt