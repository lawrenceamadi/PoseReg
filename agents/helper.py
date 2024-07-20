# -*- coding: utf-8 -*-
# @Time    : 9/18/2021 6:07 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : helper.py
# @Software: videopose3d

import sys
import torch
import pickle
import numpy as np
from tabulate import tabulate
from sklearn.covariance import empirical_covariance


use_gpu = True # set to 'False' to test on CPU only
run_on_available_gpu = use_gpu and torch.cuda.is_available()
if run_on_available_gpu:
    processor = 'cuda' #"cuda:0"
else: processor = 'cpu' #"cpu"


# Notes on keypoints
# See https://github.com/qxcv/pose-prediction/blob/master/H36M-NOTES.md for H3.6M keypoint indexes
# - Abdomen/Plv, Xiphoid-process/Spn, Base-of-neck/Nck, Face/Nse, Head/Skl
# Changed MHP->Plv, Skl->Nse, Hed->Skl, Vertebra->Thorax
#                                Plv RHp RKe RAk LHp LKe LAk Spn Nck Nse Skl LSh LEb LWr RSh REb RWr
# 17 movable H3.6M kpt indexes: [  0   1   2   3   6   7   8  12  13  14  15  17  18  19  25  26  27] common/skeleton.py/69
# corresponding output indexes: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16]
# dataset.skeleton().parents(): [ -1   0   1   2   0   4   5   0   7   8   9   8  11  12   8  14  15]
KPT_2_IDX = {'Plv':0, 'RHp':1, 'RKe':2, 'RAk':3, 'LHp':4, 'LKe':5, 'LAk':6, 'Spn':7, 'Nck':8,
             'Nse':9, 'Skl':10, 'LSh':11, 'LEb':12, 'LWr':13, 'RSh':14, 'REb':15, 'RWr':16}

BONE_KPT_PAIRS = { # format: key:BoneID, value:('Child-KptID','Parent-KptID')
    'LShoulder':('LSh','Nck'), 'LBicep':('LEb','LSh'), 'LForearm':('LWr','LEb'),
    'RShoulder':('RSh','Nck'), 'RBicep':('REb','RSh'), 'RForearm':('RWr','REb'),
    'Abdomen':('Spn','Plv'), 'Thorax':('Nck','Spn'), 'Head':('Skl','Nck'), 'UFace':('Nse','Skl'),
    'RHip':('RHp','Plv'), 'RThigh':('RKe','RHp'), 'RLeg':('RAk','RKe'),
    'LHip':('LHp','Plv'), 'LThigh':('LKe','LHp'), 'LLeg':('LAk','LKe'),
}

RGT_SYM_BONES = ['RHip','RThigh','RLeg','RShoulder','RBicep','RForearm']
LFT_SYM_BONES = ['LHip','LThigh','LLeg','LShoulder','LBicep','LForearm']


def videopose3d_bone_structure(n_bones=16, root_kpt='Plv'):
    bone_id_2_idx = {}
    bone_child_kpt_idxs = [0]*n_bones
    bone_parent_kpt_idxs = [0]*n_bones
    plv_kpt_idx = KPT_2_IDX[root_kpt]

    for bone_id, (child_kpt, parent_kpt) in BONE_KPT_PAIRS.items():
        assert (child_kpt!=root_kpt), 'Root-kpt (Pelvis) should only be a parent keypoint'

        child_kpt_idx = KPT_2_IDX[child_kpt]
        bone_idx = child_kpt_idx-1 if child_kpt_idx>plv_kpt_idx else child_kpt_idx
        bone_id_2_idx[bone_id] = bone_idx
        bone_child_kpt_idxs[bone_idx] = KPT_2_IDX[child_kpt]
        bone_parent_kpt_idxs[bone_idx] = KPT_2_IDX[parent_kpt]

    rgt_sym_bone_idxs = [bone_id_2_idx[bone_id] for bone_id in RGT_SYM_BONES]
    lft_sym_bone_idxs = [bone_id_2_idx[bone_id] for bone_id in LFT_SYM_BONES]

    # print('VIDEOPOSE3D_BONE_IDXS:\n{}'.format(bone_id_2_idx))
    # print('PARENT_KPT_IDXS: {}'.format(bone_parent_kpt_idxs))
    # print('CHILD_KPT_IDXS: {}'.format(bone_child_kpt_idxs))
    # print('RGT_SYM_BONE_IDXS: {}'.format(rgt_sym_bone_idxs))
    # print('LFT_SYM_BONE_IDXS: {}'.format(rgt_sym_bone_idxs))
    return bone_id_2_idx, bone_child_kpt_idxs, bone_parent_kpt_idxs, rgt_sym_bone_idxs, lft_sym_bone_idxs

VIDEOPOSE3D_BONE_ID_2_IDX, BONE_CHILD_KPTS_IDXS, BONE_PARENT_KPTS_IDXS, \
RGT_SYM_BONE_INDEXES, LFT_SYM_BONE_INDEXES = videopose3d_bone_structure()


RICHERS_BONE_PROPS = np.float32( # Richer's human proportions for men
#   [ 0.59,  1.88,  1.88,  0.59,  1.88,  1.88,  1.41,  0.94,  0.53,  0.58,  0.88,  1.41,  1.18,  0.88,  1.41,  1.18]
    [ 0.59,  1.94,  2.00,  0.59,  1.94,  2.00,  1.06,  1.18,  0.53,  0.47,  0.70,  1.23,  1.12,  0.70,  1.23,  1.12])
#    RHpPlv RKeRHp RAkRKe LHpPlv LKeLHp LAkLKe SpnPlv NckSpn NseNck SklNse LShNck LEbLSh LWrLEb RShNck REbRSh RWrREb
#    R-Hip  RThigh R-FLeg L-Hip  LThigh L-FLeg Abdomn Thorax  Neck   Head  LShoda LBicep L-FArm RShoda RBicep R-FArm
#      0       1      2     3       4      5      6      7      8      9     10     11     12     13     14     15

# TODO Note: after including new free-bones, quad_kpts[0] will no longer be unique but quad_kpts[3] should be
# JMC_JOINT_CONFIG = { # Set to Per-Jnt Default
#     # joint_id : [Pivot Axis Plane Free]   xy_yx_dir xy_idx yx_idx z_axb_idxs yx_axb_idxs uvec_xyz_ftrs
#     'Head'     :(['Skl','Nck','RSh','Nse'],  [-1,-1],    1,     0,     [0,1],     [1,2],      [0,1,1]),  # *
#     'Neck'     :(['Nck','Spn','LSh','Skl'],  [-1,-1],    1,     0,     [1,0],     [1,2],      [1,1,1]),  # *
#     'Spine'    :(['Spn','Nck','RSh','Plv'],  [ 1,-1],    1,     0,     [1,0],     [2,1],      [1,1,1]),  # TODO: Spn->Nck
#     'Pelvis'   :(['Plv','LHp','LSh','Spn'],  [ 1, 1],    0,     1,     [0,1],     [2,0],      [1,1,1]),  #
#     'RHip'     :(['RHp','Plv','RSh','RKe'],  [ 1, 1],    0,     1,     [0,1],     [2,0],      [1,1,1]),  # old:['RHp','LHp','RSh','RKe'] - new:['RHp','Plv','RSh','RKe']
#     #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
#     'LHip'     :(['LHp','Plv','LSh','LKe'],  [-1, 1],    0,     1,     [1,0],     [0,2],      [1,1,1]),  # old:['LHp','RHp','LSh','LKe'] - new:['LHp','Plv','LSh','LKe']
#     'RKnee'    :(['RKe','RHp','Plv','RAk'],  [ 1, 1],    1,     0,     [0,1],     [1,2],      [0,1,1]),  # * old:|[1,0], [2,1]|
#     #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
#     'LKnee'    :(['LKe','LHp','Plv','LAk'],  [ 1, 1],    1,     0,     [1,0],     [1,2],      [0,1,1]),  # * old:|[1,0], [2,1]|
#     'RShoulder':(['RSh','Nck','RHp','REb'],  [ 1, 1],    0,     1,     [1,0],     [2,0],      [1,1,1]),  # old:['RSh','LSh','RHp','REb'] - new:['RSh','Nck','RHp','REb']
#     #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
#     'LShoulder':(['LSh','Nck','LHp','LEb'],  [-1, 1],    0,     1,     [0,1],     [0,2],      [1,1,1]),  # old:['LSh','RSh','LHp','LEb'] - new:['LSh','Nck','LHp','LEb']
#     'RElbow'   :(['REb','RSh','RHp','RWr'],  [ 1, 1],    1,     0,     [0,1],     [1,2],      [0,1,1]),  # *
#     #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
#     'LElbow'   :(['LEb','LSh','LHp','LWr'],  [ 1, 1],    1,     0,     [1,0],     [1,2],      [0,1,1]),  # *
# }
JMC_RMTX_JOINT_CONFIG = { # Set to Per-Jnt Default
    # joint_id : [Pivot Axis Plane Free]   xy_yx_dir xy_idx yx_idx z_axb_idxs yx_axb_idxs uvec_xyz_ftrs
    'Head'     :(['Skl','Nck','RSh','Nse'],  [-1,-1],    1,     0,     [0,1],     [1,2]),  # *
    'Neck'     :(['Nck','Spn','LSh','Skl'],  [-1,-1],    1,     0,     [1,0],     [1,2]),  # *
    'Spine'    :(['Spn','Plv','LHp','Nck'],  [-1,-1],    1,     0,     [1,0],     [1,2]),  # prev:['Spn','Nck','RSh','Plv'], [ 1,-1],... [2,1]
    'Pelvis'   :(['Plv','LHp','Nck','Spn'],  [ 1, 1],    0,     1,     [0,1],     [2,0]),  # prev:['Plv','LHp','RSh','Spn']
    'RWaist'   :(['Plv','Spn','LHp','RHp'],  [ 1, 1],    1,     0,     [0,1],     [1,2]),  # new
    #                                      yx_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LWaist'   :(['Plv','Spn','RHp','LHp'],  [ 1,-1],    1,     0,     [1,0],     [2,1]),  # new
    'RHip'     :(['RHp','Plv','Spn','RKe'],  [ 1, 1],    0,     1,     [0,1],     [2,0]),  # prev:['RHp','Plv','RSh','RKe']
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LHip'     :(['LHp','Plv','Spn','LKe'],  [-1, 1],    0,     1,     [1,0],     [0,2]),  # prev:['LHp','Plv','LSh','LKe']
    'RKnee'    :(['RKe','RHp','Plv','RAk'],  [ 1, 1],    1,     0,     [0,1],     [1,2]),  # *
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LKnee'    :(['LKe','LHp','Plv','LAk'],  [ 1, 1],    1,     0,     [1,0],     [1,2]),  # *
    'RScapula' :(['Nck','Spn','LSh','RSh'],  [-1,-1],    1,     0,     [1,0],     [1,2]),  # new
    #                                      yx_axis_dir: -1(per-jnt), or 1(grp-jnt)
    'LScapula' :(['Nck','Spn','RSh','LSh'],  [-1,-1],    1,     0,     [0,1],     [1,2]),  # new
    'RShoulder':(['RSh','Nck','Spn','REb'],  [ 1, 1],    0,     1,     [1,0],     [2,0]),  # prev:['RSh','Nck','RHp','REb']
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LShoulder':(['LSh','Nck','Spn','LEb'],  [-1, 1],    0,     1,     [0,1],     [0,2]),  # prev:['LSh','Nck','LHp','LEb']
    'RElbow'   :(['REb','RSh','Nck','RWr'],  [ 1, 1],    1,     0,     [0,1],     [1,2]),  # prev:['REb','RSh','RHp','RWr']
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LElbow'   :(['LEb','LSh','Nck','LWr'],  [ 1, 1],    1,     0,     [1,0],     [1,2]),  # prev:['LEb','LSh','LHp','LWr']
}

JMC_QUAT_JOINT_CONFIG = { # Set to Per-Jnt Default
    # todo: quintuple for Head->Pelvis to align lft & rgt hip/shoulder
    # joint_id : [Pivot Axis Plane Free]    Quadrant axis 1 & 2 uvecs
    'Head'     :(['Skl','Nck','RSh','Nse'],  ([ 0,-1, 0], [-1, 0, 0])),  # -y by -x axes
    'Neck'     :(['Nck','Spn','LSh','Skl'],  ([ 0,-1, 0], [ 1, 0, 0])),  # -y by +x axes*
    'Spine'    :(['Spn','Plv','LHp','Nck'],  ([ 0,-1, 0], [ 1, 0, 0])),  # -y by +x axes*
    'Pelvis'   :(['Plv','LHp','Nck','Spn'],  ([ 1, 0, 0], [ 0, 1, 0])),  # +x by +y axes
    'RWaist'   :(['Plv','Spn','LHp','RHp'],  ([ 0, 1, 0], [ 1, 0, 0])),  # +y by +x axes
    #'RWaist'   :(['Plv','LHp','Spn','RHp'],  ([ 1, 0, 0], [ 0, 1, 0])),  # +y by +x axes
    #                                      yx_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LWaist'   :(['Plv','Spn','RHp','LHp'],  ([ 0, 1, 0], [-1, 0, 0])),  # +y by -x axes
    'RHip'     :(['RHp','Plv','Spn','RKe'],  ([ 1, 0, 0], [ 0, 1, 0])),  # +x by +y axes
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LHip'     :(['LHp','Plv','Spn','LKe'],  ([-1, 0, 0], [ 0, 1, 0])),  # -x by +y axes
    'RKnee'    :(['RKe','RHp','Plv','RAk'],  ([ 0, 1, 0], [ 1, 0, 0])),  # +y by +x axes
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LKnee'    :(['LKe','LHp','Plv','LAk'],  ([ 0, 1, 0], [-1, 0, 0])),  # +y by -x axes
    'RScapula' :(['Nck','Spn','LSh','RSh'],  ([ 0,-1, 0], [ 1, 0, 0])),  # -y by +x axes
    #                                      yx_axis_dir: -1(per-jnt), or 1(grp-jnt)
    'LScapula' :(['Nck','Spn','RSh','LSh'],  ([ 0,-1, 0], [-1, 0, 0])),  # -y by -x axes
    'RShoulder':(['RSh','Nck','Spn','REb'],  ([ 1, 0, 0], [ 0,-1, 0])),  # +x by -y axes
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LShoulder':(['LSh','Nck','Spn','LEb'],  ([-1, 0, 0], [ 0,-1, 0])),  # -x by -y axes
    'RElbow'   :(['REb','RSh','Nck','RWr'],  ([ 0, 1, 0], [ 1, 0, 0])),  # +y by +x axes
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LElbow'   :(['LEb','LSh','Nck','LWr'],  ([ 0, 1, 0], [-1, 0, 0])),  # +y by -x axes
}

SYM_JOINT_GROUP = ['Hip', 'Shoulder', 'Knee', 'Elbow', 'Scapula', 'Waist']
SYM_BONES = ['Shoulder','Bicep','Forearm','Hip','Thigh','Leg']
JOINT_2D_TAGS = [] #['Head', 'Elbow', 'RElbow', 'LElbow', 'Knee', 'RKnee', 'LKnee']
H36M_ACTIONS = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases',
                'Sitting','SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']

TOY_3D_POSE = np.float32([[    0,    0,    0], # Plv
                          [  160,    0,    0], # RHp
                          [  180,  360,   60], # RKe
                          [  160,  840,    0], # RAk
                          [ -160,    0,    0], # LHp
                          [ -180,  360,   60], # LKe
                          [ -180,  360,  480], # LAk*
                          [    0, -260,  -40], # Spn
                          [    0, -500,    0], # Nck
                          [    0, -600,   40], # Nse
                          [    0, -660,  -20], # Skl
                          [ -240, -480,    0], # LSh
                          [ -320, -240,  -80], # LEb
                          [ -220,   20,   80], # LWr
                          [  240, -480,    0], # RSh
                          [  240, -180,    0], # REb
                          [  240, -180, -320]] # RWr*
                         ).reshape((1,1,17,3)) / 1000


def get_keypoints_indexes(kpt_ids_list):
    kpt_indexes_list = []
    for idx, kpt_id in enumerate(kpt_ids_list):
        kpt_indexes_list.append(KPT_2_IDX[kpt_id])
    return kpt_indexes_list # now a list of corresponding kpts indexes

def get_id_index(ordered_ids_list, id):
    for idx, id_at_idx in enumerate(ordered_ids_list):
        if id_at_idx==id: return idx
    assert (False), '{} not in {}'.format(id, ordered_ids_list)

def select_bone_ratios(videopose3d_bone_2_idx):
    bone_ratio_pairs = ['UFace/Head','Head/Thorax','RShoulder/Thorax','LShoulder/Thorax','Abdomen/Thorax',
                        'RHip/Abdomen','LHip/Abdomen','RShoulder/RBicep','LShoulder/LBicep','RForearm/RBicep',
                        'LForearm/LBicep','RHip/RThigh','LHip/LThigh','RLeg/RThigh','LLeg/LThigh']
    bone_pair_idxs_A = []
    bone_pair_idxs_B = []

    for ratio_id in bone_ratio_pairs:
        boneA, boneB = ratio_id.split('/')
        bone_pair_idxs_A.append(videopose3d_bone_2_idx[boneA])
        bone_pair_idxs_B.append(videopose3d_bone_2_idx[boneB])

    return bone_ratio_pairs, bone_pair_idxs_A, bone_pair_idxs_B

def all_bone_ratio_combo(videopose3d_bone_2_idx):
    LEN_ORDERED_BONES = [ # listed in ascending order of bone lengths
        'UFace','RHip','LHip','RShoulder','LShoulder','Head','Abdomen','LForearm',
        'RForearm','Thorax','LBicep','RBicep','RLeg','LLeg','LThigh','RThigh']
    bone_ratio_pairs = []
    bone_pair_idxs_A = []
    bone_pair_idxs_B = []

    for boneA_idx in range(len(LEN_ORDERED_BONES)):
        boneA = LEN_ORDERED_BONES[boneA_idx]
        for boneB_idx in range(boneA_idx+1, len(LEN_ORDERED_BONES)):
            boneB = LEN_ORDERED_BONES[boneB_idx]
            if boneA[1:] == boneB[1:]: continue # skip symmetric bone pairs
            ratio_id1 = '{}/{}'.format(boneA, boneB)
            ratio_id2 = '{}/{}'.format(boneB, boneA)
            if ratio_id1 in bone_ratio_pairs: continue # no duplicate counting
            if ratio_id2 in bone_ratio_pairs: continue # no duplicate counting
            bone_ratio_pairs.append(ratio_id1)
            bone_pair_idxs_A.append(videopose3d_bone_2_idx[boneA])
            bone_pair_idxs_B.append(videopose3d_bone_2_idx[boneB])
    print('[INFO] Counted {} unique bone ratio pairs'.format(len(bone_ratio_pairs)))
    return bone_ratio_pairs, bone_pair_idxs_A, bone_pair_idxs_B

def display_subject_actions_frame_counts(info_table_log, sum_rows=True, empty_cell=0):
    columns = H36M_ACTIONS + ['All-Acts']
    all_subject_cnt = {}
    all_rows = []
    for subject, actions_cnt in info_table_log.items():
        row_entry = [subject]
        for column_name in columns:
            act_cnt = actions_cnt.get(column_name, empty_cell)
            row_entry.append("{:,}".format(act_cnt))
            all_subject_cnt[column_name] = all_subject_cnt.get(column_name, 0) + act_cnt
        all_rows.append(row_entry)

    # sum all actions/columns across all subjects
    if sum_rows:
        row_entry = ['All-Subs']
        for column_name in columns:
            act_cnt = all_subject_cnt.get(column_name, empty_cell)
            row_entry.append("{:,}".format(act_cnt))
        all_rows.append(row_entry)

    col_header = ['Subjects'] + [col_name[0:8] for col_name in columns]
    info_table = tabulate(all_rows, headers=col_header, tablefmt='psql', showindex=False, stralign="right")
    return info_table

def pickle_write(py_object, filepath):
    with open(filepath, 'wb') as file_handle:
        pickle.dump(py_object, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(filepath):
    with open(filepath, 'rb') as file_handle:
        return pickle.load(file_handle)

def is_symmetric(M):
    return np.all(np.abs(M - M.T)<=1e-6)

def is_positive_definite(M):
    # https://en.wikipedia.org/wiki/Definite_matrix
    n, m = M.shape
    assert (n==m), 'must be a square matrix not {}'.format(M.shape)
    assert (is_symmetric(M)), 'should be symmetric:\n {}'.format(M)
    pos_def_zMz = [] #np.zeros((n,), dtype=np.float32)
    for i in range(n):
        z = M[:,[i]] # (3,1) or (2,1)
        #if not np.all(z==0): # skip zero vectors
        #   pos_def_zMz[i] = np.matmul(np.matmul(z.T, M), z)
        if not np.all(np.isclose(z, 0)): # skip zero vectors
            pos_def_zMz.append(np.matmul(np.matmul(z.T, M), z))
    #return np.all(pos_def_zMz>=0)
    return np.all(np.asarray(pos_def_zMz)>0)

def non_max_supress_points(points_coord, points_ftrvec=None, d=0.01): # k: [0.05, 0.001]
    '''Non-maximum suppression'''
    agg_points_coord, agg_points_ftrvec, agg_counts = [], [], []
    agg_counts = list()
    while len(points_coord)>0:
        diff = points_coord - points_coord[[0]]
        dist = np.linalg.norm(diff, axis=1)
        assert(np.all(dist>=0)), 'dist should not be negative {}'.format(dist)
        #assert(dist.ndim==1), 'dist.ndim:{}'.format(dist.ndim)
        neighbor_pnts_idxs = np.argwhere(dist<=d).flatten()
        #neighbor_pnts = points_coord[neighbor_pnts_idxs]
        #np.mean(neighbor_pnts, axis=0)
        #assert(neighbor_pnts.shape[0]==len(neighbor_pnts_idxs)), 'neighbor_pnts.shape:{}'.format(neighbor_pnts.shape)
        agg_counts.append(len(neighbor_pnts_idxs))
        aggregate_pnt_coord = points_coord[0]
        agg_points_coord.append(aggregate_pnt_coord)
        points_coord = np.delete(points_coord, neighbor_pnts_idxs, axis=0)
        if points_ftrvec is not None:
            aggregate_pnt_ftrvec = points_ftrvec[0]
            agg_points_ftrvec.append(aggregate_pnt_ftrvec)
            points_ftrvec = np.delete(points_ftrvec, neighbor_pnts_idxs, axis=0)
    agg_points_wgt = np.float32(agg_counts) / np.max(agg_counts)
    if points_ftrvec is None: return np.float32(agg_points_coord), agg_points_wgt
    return np.float32(agg_points_coord), np.float32(agg_points_ftrvec), agg_points_wgt

def points_on_circumference(orient_type=1, step=5, radius=1., eps=1e-05):
    # Free-bone 2D data-points generated from closed spherical domain ([0:'scs', 1:'ccs'])
    theta_degrees = np.arange(0, 360, step) # interval: [0, 360)
    points_coords, points_orient = [], []
    for theta_deg in theta_degrees:
        theta = np.radians(theta_deg)
        x = radius*np.cos(theta)
        y = radius*np.sin(theta)
        points_coords.append([x, y])
        if orient_type==0:
            azimuth_theta_deg = theta_deg - (360 * np.floor(theta_deg/(180+eps))) # [0,360)->(-pi,pi]
            points_orient.append(np.radians(azimuth_theta_deg))
    if orient_type==0:
        return np.float32(points_coords), np.float32(points_orient)
    return np.float32(points_coords), np.float32(points_coords)

def scs_to_ccs(theta, phi, radius=1.):
    x = radius*np.cos(theta)*np.sin(phi)
    y = radius*np.sin(theta)*np.sin(phi)
    z = radius*np.cos(phi)
    return x, y, z


def points_on_spherical_surface(orient_type=1, step=5, radius=1., eps=1e-05):
    # Free-bone 3D data-points generated from closed spherical domain ([0:'scs', 1:'ccs'])
    phi_degrees = np.arange(0, 180+step, step) # interval: [0, 180]
    theta_degrees = np.arange(-180+step, 180, step) # interval: (-180, 180] equivalent to [0, 360)
    points_coords, points_orient = [], []
    for phi_deg in phi_degrees:
        phi = np.radians(phi_deg) # elevation angle
        for theta_deg in theta_degrees:
            theta = np.radians(theta_deg) # azimuth angle
            x = radius*np.cos(theta)*np.sin(phi)
            y = radius*np.sin(theta)*np.sin(phi)
            z = radius*np.cos(phi)
            points_coords.append([x, y, z])
            if orient_type==0: points_orient.append([theta, phi])
    if orient_type==0:
        return np.float32(points_coords), np.float32(points_orient)
    return np.float32(points_coords), np.float32(points_coords)


def get_value_with_max_loglihood(values, values_loglihood):
    # return values[np.argmax(values_loglihood)]
    max_loglihood = np.max(values_loglihood)
    max_loglihood_indices = []
    for i in range(values_loglihood.shape[0]):
        if values_loglihood[i]>=max_loglihood: max_loglihood_indices.append(i)
    max_loglihood_values = values[max_loglihood_indices]
    return np.mean(max_loglihood_values, axis=0) # note, mean may not be a unit vector on surface of sphere


def log_stretch(values, spread=0.001):
    '''
    Computes a vertically translated (upward) and strech of natural logarithm
    Args:
        values: ndarray of values
        spread: from interval (0,1], preferably [1e-5,1]. Controls the spread (y-range)
    Returns: vertically shifted natural logarithm of values within >=0
    '''
    return np.log(values+spread) - np.log(spread)


def probability_density_func(xdata, mu, sigma):
    # Computes multivariate probability density function of xdata->free_limb_unit_vec->(N,k)
    # given mu->mean->(1,), sigma->variance->(1,), and k->#_of_features/components
    # https://en.wikipedia.org/wiki/Probability_density_function (equation in "Families of densities")
    # or https://www.sciencedirect.com/topics/earth-and-planetary-sciences/normal-density-functions
    exponent_coef = (2*np.pi*sigma)**(-1/2) # (1,)
    mean_centered = xdata - mu # (N,)
    exponent = (-1/2)*(mean_centered**2/sigma) # (N,)
    likelihoods = exponent_coef * np.exp(exponent) # (N,)
    assert (np.all(likelihoods>=0)), "likelihoods:\n{}".format(likelihoods)
    return likelihoods # (N,)


def mvpdf_exponent_coefficient(sigma, k=None, vs_with_k=False, append_dims=False):
    '''
    Args:
        sigma: (...,j,k,k) covariance matrices
        k: None or int, the dimension of covariance matrix
        vs_with_k:
        append_dims:
    Returns:
    '''
    # (...,j) --> (1,) or (1,1,j)
    if vs_with_k:
        exp_coefs = (2*np.pi)**(-k/2) * np.linalg.det(sigma)**(-1/2) # (...,j)
    else: exp_coefs = np.linalg.det(2*np.pi * sigma)**(-1/2) # (...,j)
    # assert(np.isclose(exponent_coef0, exponent_coef, atol=1e-05)), "{} != {}".format(exponent_coef0, exponent_coef)
    if append_dims:
        return exp_coefs.reshape(exp_coefs.shape+(1,1)) # (1,1,j)->(1,1,j,1,1)
    return exp_coefs


def multivariate_probability_density_func(xdata, mu, sigma, sigma_inv=None, vs_with_k=False):
    # Computes multivariate probability density function of xdata->free_limb_unit_vec->(N,k)
    # given mu->mean_per_axis->(3,1), sigma->covariance_matrix->(3,3), and k->#_of_features/components
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # (search "PDF" in table to the right and/or "Non-degenerate case")
    xdata = np.expand_dims(xdata, axis=2) # (N,k)->(N,k,1)
    mu = np.expand_dims(mu, axis=0) # (k,1)->(1,k,1)
    sigma = np.expand_dims(sigma, axis=0) # (k,k)->(1,k,k)
    exponent_coef = mvpdf_exponent_coefficient(sigma, sigma.shape[0], vs_with_k)
    mean_centered = xdata - mu # (N,k,1)
    mean_centered_T = np.transpose(mean_centered, (0,2,1)) # (N,1,k)
    if sigma_inv is None:
        inv_covariance = np.linalg.inv(sigma) # (1,k,k)
    else: inv_covariance = np.expand_dims(sigma_inv, axis=0) # (k,k)->(1,k,k)
    exponent = (-1/2)*np.matmul(np.matmul(mean_centered_T, inv_covariance), mean_centered) # (N,1,1)
    likelihoods = exponent_coef * np.exp(exponent) # (N,1,1)
    assert (np.all(likelihoods>=0)), "likelihoods:\n{}".format(likelihoods)
    return likelihoods[:,0,0] # (N,1,1)->(N,)


def likelihood_func(x_features, mu_mean, sigma_variance):
    if mu_mean.shape[0]==1:
        return probability_density_func(x_features, mu_mean, sigma_variance)
    return multivariate_probability_density_func(x_features, mu_mean, sigma_variance)

