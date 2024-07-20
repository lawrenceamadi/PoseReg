# -*- coding: utf-8 -*-
# @Time    : 9/18/2021 6:07 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : helper.py
# @Software: pose.reg

import sys
import copy
import torch
import pickle
import numpy as np
from tabulate import tabulate


use_gpu = True # set to 'False' to test on CPU only
run_on_available_gpu = use_gpu and torch.cuda.is_available()
if run_on_available_gpu:
    processor = 'cuda' # 'cuda:0'
else: processor = 'cpu'


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
KPT16_2_IDX = {'Plv':0, 'RHp':1, 'RKe':2, 'RAk':3, 'LHp':4, 'LKe':5, 'LAk':6, 'Spn':7, 'Nck':8,
               'Skl':9, 'LSh':10, 'LEb':11, 'LWr':12, 'RSh':13, 'REb':14, 'RWr':15} # version without Nse

BONE_KPT_PAIRS = { # format: key:BoneID, value:('Child-KptID','Parent-KptID')
    'LShoulder':('LSh','Nck'), 'LBicep':('LEb','LSh'), 'LForearm':('LWr','LEb'),
    'RShoulder':('RSh','Nck'), 'RBicep':('REb','RSh'), 'RForearm':('RWr','REb'),
    'Abdomen':('Spn','Plv'), 'Thorax':('Nck','Spn'), 'Head':('Skl','Nck'), 'UFace':('Nse','Skl'),
    'RHip':('RHp','Plv'), 'RThigh':('RKe','RHp'), 'RLeg':('RAk','RKe'),
    'LHip':('LHp','Plv'), 'LThigh':('LKe','LHp'), 'LLeg':('LAk','LKe'),
}

RGT_SYM_BONES = ['RHip','RThigh','RLeg','RShoulder','RBicep','RForearm']
LFT_SYM_BONES = ['LHip','LThigh','LLeg','LShoulder','LBicep','LForearm']
CENTERD_BONES = ['UFace','Head','Thorax','Abdomen']

KPT_HEADER_ORDER = ['RWr','LWr','REb','LEb','RSh','LSh','Nse','Skl','Nck','Spn','Plv','RHp','LHp','RKe','LKe','RAk','LAk']
BONE_HEADER_ORDER = \
    ['RArm','LArm','RBcp','LBcp','RShd','LShd','Face','Head','Thrx','Abdm','RHip','LHip','RThg','LThg','RLeg','LLeg']
BONE_ALIGN_ORDER = \
    ['Face','Head','Thrx','Abdm','RHip','LHip','RThg','LThg','RLeg','LLeg','RShd','LShd','RBcp','LBcp','RArm','LArm']


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
    centerd_bone_idxs = [bone_id_2_idx[bone_id] for bone_id in CENTERD_BONES]

    return bone_id_2_idx, bone_child_kpt_idxs, bone_parent_kpt_idxs, \
           rgt_sym_bone_idxs, lft_sym_bone_idxs, centerd_bone_idxs

VPOSE3D_BONE_ID_2_IDX, BONE_CHILD_KPTS_IDXS, BONE_PARENT_KPTS_IDXS, \
RGT_SYM_BONE_INDEXES, LFT_SYM_BONE_INDEXES, CENTERD_BONE_INDEXES = videopose3d_bone_structure()


# RICHERS_BONE_PROPS = np.float32( # Richer's human proportions for men
#     [ 0.59,  1.94,  2.00,  0.59,  1.94,  2.00,  1.06,  1.18,  0.53,  0.47,  0.70,  1.23,  1.12,  0.70,  1.23,  1.12])
# #    RHpPlv RKeRHp RAkRKe LHpPlv LKeLHp LAkLKe SpnPlv NckSpn NseNck SklNse LShNck LEbLSh LWrLEb RShNck REbRSh RWrREb
# #    R-Hip  RThigh R-FLeg L-Hip  LThigh L-FLeg Abdomn Thorax  Neck   Head  LShoda LBicep L-FArm RShoda RBicep R-FArm
# #      0       1      2     3       4      5      6      7      8      9     10     11     12     13     14     15

QUINTUPLET_KPT_WGT = {0:('Pivot',0.03), 1:('Axis',0.01), 2:('Plane1',0.005), 3:('Plane2',0.005), 4:('Free',0.95)}
QUADRUPLET_KPT_WGT = {0:('Pivot',0.03), 1:('Axis',0.01), 2:('Plane',0.01), 3:('Free',0.95)} # Free->Vector


# Note: quad_kpts[0] (Pivot-kpt) per bone is not unique but quad_kpts[3] (Free-Kpt) is unique

# MPBOE_BONE_ALIGN_CONFIG_v0 = { # Set to Per-Jnt Default
#     # joint_id : [Pivot Axis|-&-Plane-Pair Free]  xy_yx_dir xy_idx yx_idx z_axb_idxs yx_axb_idxs
#     'UFace'    :(['Skl','Nck','RSh','LSh','Nse'],  [-1,-1],    1,     0,     [0,1],     [1,2]), # Head>>Cranium
#     'Head'     :(['Nck','Spn','LSh','RSh','Skl'],  [-1,-1],    1,     0,     [1,0],     [1,2]), # Neck>>Vertebra **!!**
#     'Thorax'   :(['Spn','Plv','LHp','RHp','Nck'],  [-1,-1],    1,     0,     [1,0],     [1,2]), # Spine>>Sternum **!!**
#     'Abdomen'  :(['Plv','Nck','RHp','LHp','Spn'],  [ 1,-1],    1,     0,     [1,0],     [2,1]), # Pelvis>>Spine
#     # 4th quintuple kpt below are just placeholders (duplicates of the 3rd kpt) and not relevant
#     'RHip'     :(['Plv','Spn','LHp','LHp','RHp'],  [ 1, 1],    1,     0,     [0,1],     [1,2]), # RWaist>>
#     #                                     yx_axis_dir: -1(per-jnt) or 1(grp-jnt)
#     'LHip'     :(['Plv','Spn','RHp','RHp','LHp'],  [ 1,-1],    1,     0,     [1,0],     [2,1]), # LWaist>> *JMC
#     'RThigh'   :(['RHp','Plv','Spn','Spn','RKe'],  [ 1, 1],    0,     1,     [0,1],     [2,0]), # RHip>>R.Femur
#     #                                     xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
#     'LThigh'   :(['LHp','Plv','Spn','Spn','LKe'],  [-1, 1],    0,     1,     [1,0],     [0,2]), # LHip>>L.Femur
#     'RLeg'     :(['RKe','RHp','Plv','Plv','RAk'],  [ 1, 1],    1,     0,     [0,1],     [1,2]), # RKnee>>R.Tibia
#     #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
#     'LLeg'     :(['LKe','LHp','Plv','Plv','LAk'],  [ 1, 1],    1,     0,     [1,0],     [1,2]), # LLeg>>L.Tibia
#     'RShoulder':(['Nck','Spn','LSh','LSh','RSh'],  [-1,-1],    1,     0,     [1,0],     [1,2]), # RScapula>>R.Clavicle **!!**
#     #                                     yx_axis_dir: -1(per-jnt), or 1(grp-jnt)
#     'LShoulder':(['Nck','Spn','RSh','RSh','LSh'],  [-1,-1],    1,     0,     [0,1],     [1,2]), # LScapula>>L.Clavicle
#     'RBicep'   :(['RSh','Nck','Spn','Spn','REb'],  [ 1, 1],    0,     1,     [1,0],     [2,0]), # RShoulder>>R.Humerus
#     #                                     xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
#     'LBicep'   :(['LSh','Nck','Spn','Spn','LEb'],  [-1, 1],    0,     1,     [0,1],     [0,2]), # LShoulder>>L.Humerus
#     'RForearm' :(['REb','RSh','Nck','Nck','RWr'],  [ 1, 1],    1,     0,     [0,1],     [1,2]), # RElbow>>R.Radius
#     #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
#     'LForearm' :(['LEb','LSh','Nck','Nck','LWr'],  [ 1, 1],    1,     0,     [1,0],     [1,2]), # LElbow>>L.Radius
# }

JMC_RMTX_JOINT_CONFIG = { # Set to Per-Jnt Default
    # joint_id : [Pivot Axis|-&-Plane-Pair Free]  xy_yx_dir xy_idx yx_idx z_axb_idxs yx_axb_idxs | [*]-->change
    'UFace'    :(['Skl','Nck','RSh','LSh','Nse'],  [-1,-1],    1,     0,     [0,1],     [1,2]), # Head>>Cranium
    'Head'     :(['Nck','Spn','LSh','RSh','Skl'],  [-1, 1],    1,     0,     [1,0],     [2,1]), # Neck>>Vertebra **!!**
    'Thorax'   :(['Spn','Plv','LHp','RHp','Nck'],  [-1, 1],    1,     0,     [1,0],     [2,1]), # Spine>>Sternum **!!**
    'Abdomen'  :(['Plv','Nck','RHp','LHp','Spn'],  [ 1,-1],    1,     0,     [1,0],     [2,1]), # Pelvis>>Spine
    # 4th quintuple kpt below are just placeholders (duplicates of the 3rd kpt) and not relevant
    'RHip'     :(['Plv','Spn','LHp','LHp','RHp'],  [ 1, 1],    1,     0,     [0,1],     [1,2]), # >>RWaist
    #                                      yx_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LHip'     :(['Plv','Spn','RHp','RHp','LHp'],  [ 1,-1],    1,     0,     [1,0],     [2,1]), # >>LWaist
    'RThigh'   :(['RHp','Plv','Spn','Spn','RKe'],  [ 1, 1],    0,     1,     [0,1],     [2,0]), # RHip>>R.Femur
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LThigh'   :(['LHp','Plv','Spn','Spn','LKe'],  [-1, 1],    0,     1,     [1,0],     [0,2]), # LHip>>L.Femur
    'RLeg'     :(['RKe','RHp','Plv','Plv','RAk'],  [ 1, 1],    1,     0,     [0,1],     [1,2]), # RKnee>>R.Tibia
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LLeg'     :(['LKe','LHp','Plv','Plv','LAk'],  [ 1, 1],    1,     0,     [1,0],     [1,2]), # LLeg>>L.Tibia
    'RShoulder':(['Nck','Spn','LSh','LSh','RSh'],  [-1, 1],    1,     0,     [1,0],     [2,1]), # RScapula>>R.Clavicle **!!**
    #                                      yx_axis_dir: -1(per-jnt), or 1(grp-jnt)
    'LShoulder':(['Nck','Spn','RSh','RSh','LSh'],  [-1,-1],    1,     0,     [0,1],     [1,2]), # LScapula>>L.Clavicle
    'RBicep'   :(['RSh','Nck','Spn','Spn','REb'],  [ 1, 1],    0,     1,     [1,0],     [2,0]), # RShoulder>>R.Humerus
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LBicep'   :(['LSh','Nck','Spn','Spn','LEb'],  [-1, 1],    0,     1,     [0,1],     [0,2]), # LShoulder>>L.Humerus
    'RForearm' :(['REb','RSh','Nck','Nck','RWr'],  [ 1, 1],    1,     0,     [0,1],     [1,2]), # RElbow>>R.Radius
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LForearm' :(['LEb','LSh','Nck','Nck','LWr'],  [ 1,-1],    1,     0,     [1,0],     [2,1]), # LElbow>>L.Radius
}
MPBOE_BONE_ALIGN_CONFIG = copy.deepcopy(JMC_RMTX_JOINT_CONFIG)

JMC_QUAT_JOINT_CONFIG = { # Set to Per-Jnt Default
    # todo: quintuple for Head->Pelvis to align lft & rgt hip/shoulder
    # joint_id : [Pivot Axis Plane1 Plane2 Free] Quadrant axis1 & axis2 uvecs
    'UFace'    :(['Skl','Nck','LSh','RSh','Nse'],  ([ 0,-1, 0], [ 1, 0, 0])),  # -y by +x axes * new change
    'Head'     :(['Nck','Spn','LSh','RSh','Skl'],  ([ 0,-1, 0], [ 1, 0, 0])),  # -y by +x axes*
    'Thorax'   :(['Spn','Plv','LHp','RHp','Nck'],  ([ 0,-1, 0], [ 1, 0, 0])),  # -y by +x axes*
    'Abdomen'  :(['Plv','Nck','LHp','RHp','Spn'],  ([ 0, 1, 0], [ 1, 0, 0])),  # +y by +x axes * new change
    # 4th quintuple kpt below are just placeholders (duplicates of the 3rd kpt) and not relevant
    'RHip'     :(['Plv','Spn','LHp','LHp','RHp'],  ([ 0, 1, 0], [ 1, 0, 0])),  # +y by +x axes
    #                                      yx_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LHip'     :(['Plv','Spn','RHp','RHp','LHp'],  ([ 0, 1, 0], [-1, 0, 0])),  # +y by -x axes
    'RThigh'   :(['RHp','Plv','Spn','Spn','RKe'],  ([ 1, 0, 0], [ 0, 1, 0])),  # +x by +y axes
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LThigh'   :(['LHp','Plv','Spn','Spn','LKe'],  ([-1, 0, 0], [ 0, 1, 0])),  # -x by +y axes
    'RLeg'     :(['RKe','RHp','Plv','Plv','RAk'],  ([ 0, 1, 0], [ 1, 0, 0])),  # +y by +x axes
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LLeg'     :(['LKe','LHp','Plv','Plv','LAk'],  ([ 0, 1, 0], [-1, 0, 0])),  # +y by -x axes
    'RShoulder':(['Nck','Spn','LSh','LSh','RSh'],  ([ 0,-1, 0], [ 1, 0, 0])),  # -y by +x axes
    #                                      yx_axis_dir: -1(per-jnt), or 1(grp-jnt)
    'LShoulder':(['Nck','Spn','RSh','RSh','LSh'],  ([ 0,-1, 0], [-1, 0, 0])),  # -y by -x axes
    'RBicep'   :(['RSh','Nck','Spn','Spn','REb'],  ([ 1, 0, 0], [ 0,-1, 0])),  # +x by -y axes
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LBicep'   :(['LSh','Nck','Spn','Spn','LEb'],  ([-1, 0, 0], [ 0,-1, 0])),  # -x by -y axes
    'RForearm' :(['REb','RSh','Nck','Nck','RWr'],  ([ 0, 1, 0], [ 1, 0, 0])),  # +y by +x axes
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LForearm' :(['LEb','LSh','Nck','Nck','LWr'],  ([ 0, 1, 0], [-1, 0, 0])),  # +y by -x axes
}

BONE_ALIGN_CONFIG = { # Appropriate for building 3D pose from priors
    # joint_id : [Pivot Axis|-&-Plane-Pair Free]  xy_yx_dir xy_idx yx_idx z_axb_idxs yx_axb_idxs | [*]-->change
    'UFace'    :(['Skl','Nck','RSh','LSh','Nse'],  [-1,-1],    1,     0,     [0,1],     [1,2]), # Head>>Cranium
    'Head'     :(['Nck','Spn','LSh','RSh','Skl'],  [-1, 1],    1,     0,     [1,0],     [2,1]), # Neck>>Vertebra **!!**
    'Thorax'   :(['Spn','Plv','LHp','RHp','Nck'],  [-1, 1],    1,     0,     [1,0],     [2,1]), # Spine>>Sternum **!!**
    'Abdomen'  :(['Plv','Nck','RHp','LHp','Spn'],  [ 1,-1],    1,     0,     [1,0],     [2,1]), # Pelvis>>Spine
    # 4th quintuple kpt below are just placeholders (duplicates of the 3rd kpt) and not relevant
    'RHip'     :(['Plv','Spn','LHp','LHp','RHp'],  [ 1, 1],    1,     0,     [0,1],     [1,2]), # >>RWaist
    #                                      yx_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LHip'     :(['Plv','Spn','RHp','RHp','LHp'],  [ 1,-1],    1,     0,     [1,0],     [2,1]), # >>LWaist
    'RThigh'   :(['RHp','Plv','Spn','Spn','RKe'],  [ 1, 1],    0,     1,     [0,1],     [2,0]), # RHip>>R.Femur
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LThigh'   :(['LHp','Plv','Spn','Spn','LKe'],  [-1, 1],    0,     1,     [1,0],     [0,2]), # LHip>>L.Femur
    'RLeg'     :(['RKe','RHp','Plv','Plv','RAk'],  [ 1, 1],    1,     0,     [0,1],     [1,2]), # RKnee>>R.Tibia
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LLeg'     :(['LKe','LHp','Plv','Plv','LAk'],  [ 1, 1],    1,     0,     [1,0],     [1,2]), # LLeg>>L.Tibia
    'RShoulder':(['Nck','Plv','RHp','RHp','RSh'],  [-1, 1],    1,     0,     [0,1],     [2,1]), # RScapula>>R.Clavicle **!!**
    #                                      yx_axis_dir: -1(per-jnt), or 1(grp-jnt)
    'LShoulder':(['Nck','Plv','LHp','LHp','LSh'],  [-1,-1],    1,     0,     [1,0],     [1,2]), # LScapula>>L.Clavicle
    'RBicep'   :(['RSh','Nck','Spn','Spn','REb'],  [ 1, 1],    0,     1,     [1,0],     [2,0]), # RShoulder>>R.Humerus
    #                                      xy_axis_dir: -1(per-jnt) or 1(grp-jnt)
    'LBicep'   :(['LSh','Nck','Spn','Spn','LEb'],  [-1, 1],    0,     1,     [0,1],     [0,2]), # LShoulder>>L.Humerus
    'RForearm' :(['REb','RSh','Nck','Nck','RWr'],  [ 1, 1],    1,     0,     [0,1],     [1,2]), # RElbow>>R.Radius
    #                                      yx_axis_dir: 1(per-jnt) or -1(grp-jnt)
    'LForearm' :(['LEb','LSh','Nck','Nck','LWr'],  [ 1,-1],    1,     0,     [1,0],     [2,1]), # LElbow>>L.Radius
}


SYM_JOINT_GROUP = ['Hip', 'Thigh', 'Leg', 'Shoulder', 'Bicep', 'Forearm']
H36M_ACTIONS = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases',
                'Sitting','SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']

# TOY_3D_POSE = np.float32([[    0,    0,    0], # Plv
#                           [  160,    0,    0], # RHp
#                           [  180,  360,   60], # RKe
#                           [  160,  840,    0], # RAk
#                           [ -160,    0,    0], # LHp
#                           [ -180,  360,   60], # LKe
#                           [ -180,  360,  480], # LAk*
#                           [    0, -260,  -40], # Spn
#                           [    0, -500,    0], # Nck
#                           [    0, -600,   40], # Nse
#                           [    0, -660,  -20], # Skl
#                           [ -240, -480,    0], # LSh
#                           [ -320, -240,  -80], # LEb
#                           [ -220,   20,   80], # LWr
#                           [  240, -480,    0], # RSh
#                           [  240, -180,    0], # REb
#                           [  240, -180, -320]] # RWr*
#                          ).reshape((1,1,17,3)) / 1000

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

def add_estimated_nose_kpt(poses_2d):
    # Add nose keypoint as the midway position between neck and skull keypoint
    nck_idx, nse_idx, adj_skl_idx = KPT_2_IDX['Nck'], KPT_2_IDX['Nse'], KPT16_2_IDX['Skl']
    est_nse_pose = np.mean(poses_2d[:,[nck_idx, adj_skl_idx]], axis=1, keepdims=True) # (?,b,2)->(?,j,2)
    return np.concatenate([poses_2d[:,:nse_idx], est_nse_pose, poses_2d[:,nse_idx:]], axis=1)

def abbreviate_action_names(actions, n_chars=5):
    actions_abb = []
    for act_name in actions:
        if act_name=='SittingDown': act_name = 'SitDown'
        actions_abb.append(act_name[0:n_chars]) # [0:8]>>[0:5]
    return actions_abb

def display_subject_actions_frame_counts(info_table_log, sum_rows=True, empty_cell=0, table_style='psql'):
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
        row_entry = ['All-S'] # All subject sets
        for column_name in columns:
            act_cnt = all_subject_cnt.get(column_name, empty_cell)
            row_entry.append("{:,}".format(act_cnt))
        all_rows.append(row_entry)

    col_header = ['Set'] + abbreviate_action_names(columns)
    info_table = tabulate(all_rows, headers=col_header, tablefmt=table_style, showindex=False, stralign="right")
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
        if not np.all(np.isclose(z, 0)): # skip zero vectors
            pos_def_zMz.append(np.matmul(np.matmul(z.T, M), z))
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


def mvpdf_exponent_coefficient(sigma, k=None, vrs_with_k=True, expand_dims=False):
    '''
    Args:
        sigma: (...,j,k,k) covariance matrices
        k: None or int, the dimension of covariance matrix
        vrs_with_k: whether to use the exponent function version with 'k' or not
        expand_dims: expand trailing dimensions
    Returns:
    '''
    # (...,j) --> (1,) or (1,1,j)
    if vrs_with_k:
        exp_coefs = (2*np.pi)**(-k/2) * np.linalg.det(sigma)**(-1/2) # (...,j) - Option-1
        # exp_coefs_ = np.linalg.det(2*np.pi * sigma)**(-1/2) # (...,j) (not equal to option-2)
        # assert(np.isclose(exp_coefs, exp_coefs_, atol=1e-05)), "{} != {}".format(exp_coefs, exp_coefs_)
    else: exp_coefs = np.linalg.det(2*np.pi * sigma)**(-1/2) # (...,j) - Option-2

    if expand_dims:
        return exp_coefs.reshape(exp_coefs.shape+(1,1)) # (1,1,j)->(1,1,j,1,1)
    return exp_coefs


def multivariate_probability_density_func(xdata, mu, sigma, sigma_inv=None, vrs_with_k=True):
    # Computes multivariate probability density function of xdata->free_limb_unit_vec->(N,k)
    # given mu->mean_per_axis->(3,1), sigma->covariance_matrix->(3,3), and k->#_of_features/components
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # (search "PDF" in table to the right and/or "Non-degenerate case")
    xdata = np.expand_dims(xdata, axis=2) # (N,k)->(N,k,1)
    mu = np.expand_dims(mu, axis=0) # (k,1)->(1,k,1)
    sigma = np.expand_dims(sigma, axis=0) # (k,k)->(1,k,k)
    exponent_coef = mvpdf_exponent_coefficient(sigma, sigma.shape[0], vrs_with_k)
    mean_centered = xdata - mu # (N,k,1)
    mean_centered_T = np.transpose(mean_centered, (0,2,1)) # (N,1,k)
    if sigma_inv is None:
        inv_covariance = np.linalg.inv(sigma) # (1,k,k)
    else: inv_covariance = np.expand_dims(sigma_inv, axis=0) # (k,k)->(1,k,k)
    exponent = (-1/2)*np.matmul(np.matmul(mean_centered_T, inv_covariance), mean_centered) # (N,1,1)
    likelihoods = exponent_coef * np.exp(exponent) # (N,1,1)
    assert (np.all(likelihoods>=0)), "likelihoods:\n{}".format(likelihoods)
    return likelihoods[:,0,0] # (N,1,1)->(N,)
