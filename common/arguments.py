# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset') # h36m or humaneva
    parser.add_argument('-dhd', '--dataset-home-dir', default='../../datasets/human36m/', type=str, metavar='NAME', help='location')
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='supervised training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-frequency', default=1000, type=int, metavar='N',  # was default=10
                        help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')

    # Model arguments
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', default=60, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('-drop', '--dropout', default=0.25, type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='disable train-time flipping')
    parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
                        help='disable test-time flipping')
    parser.add_argument('-arc', '--architecture', default='3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')
    parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N', help='number of channels in convolution layers')

    # Experimental
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--warmup', default=1, type=int, metavar='N', help='warm-up epochs for semi-supervision')
    parser.add_argument('--no-eval', action='store_true', help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')
    parser.add_argument('--disable-optimizations', action='store_true', help='disable optimized model for single-frame predictions')
    #parser.add_argument('--linear-projection', action='store_true', help='use only linear coefficients for semi-supervised projection')
    #parser.add_argument('--no-proj', action='store_true', help='disable projection for semi-supervised setting')
    parser.add_argument('-proj-t', '--projection-type', default=1, type=int, help='2D-Reprojection function for semi-supervised setting; '
                        '-2:orthographic projection with repose (no cam-int), -1:orthographic projection (no cam-int), 0: no projection, '
                        '1: non-linear projection with all cam-intrinsics, 2:linear projection with only linear camera coefficients')

    # added by Lawrence 03/25/2021
    parser.add_argument('-hyper', '--tune-hyper-params', action='store_true', dest='tune_hyper_parameters',
                        help='enable hyper parameter tuning training experiment')
    parser.add_argument('-no-repose', '--reposition-pose', action='store_false', dest='reposition_to_origin',
                        help='disable repositioning of pose CoM/pelvis to origin before computing bse,bpc,jmc in self-supervised-branch')
    parser.add_argument('-reg-sup', '--regularized_supervised', action='store_true', dest='supervision_with_pose_regularizers',
                        help='when enabled, pose regularizers loss terms are computed on the fully-supervised branch')
    parser.add_argument('-no-mble', '--no-bone-length', action='store_false', dest='mean_bone_length_term',
                        help='do not compute bone length error in self-supervised-branch for semi-supervised settings')
    parser.add_argument('-pto', '--pelvis-placement', action='store_true', dest='pelvis_placement_term',
                        help='enable pose placement error term in unsupervised-branch of semi-supervised settings')
    parser.add_argument('-bse', '--bone-symmetry', action='store_true', dest='bone_symmetry_term',
                        help='enable symmetric pair bone length error in unsupervised-branch of semi-supervised settings')
    parser.add_argument('-bpc', '--bone-proportion', action='store_true', dest='bone_proportion_term',
                        help='enable bone length proportions error term in unsupervised-branch of semi-supervised settings')
    parser.add_argument('-jmc', '--joint-mobility', action='store_true', dest='joint_mobility_term',
                        help='enable joint mobility error term in unsupervised-branch of semi-supervised settings')
    # parser.add_argument('-bse-t', '--bse-prior-type', default=0, type=int,
    #                     help='bse prior type; 0:mean-absolute-error , 1:mean-square-error')
    # parser.add_argument('-bpc-t', '--bpc-prior-type', default=2, type=int,
    #                     help='bpc prior type; 0:boneLen/sum(all), 1:bone-pairs.boneA/boneB, 2:bone-pairs.log-likelihood')
    parser.add_argument('-jmc-r', '--jmc-ranks', default=1, type=int, help='max number of jmc ranks, from 1 to j')
    # parser.add_argument('-jmc-t', '--jmc-prior-type', default=1, type=int,
    #                     help='jmc prior type computed from free-limb unit vector; 0:euclidean-distance, 1:log-likelihood')
    parser.add_argument('-rbo-t', '--rbo_ops_type', default='rmtx', type=str,
                        help='relative-bone-orientation operation used in jmc: "rmtx":rotation-matrix or "quat":quaternion')
    parser.add_argument('-pdf-cov', '--pdf_covariance_type', default='noctr', type=str,
                        help='tag for the type covariance estimation used to generate prior params')
    parser.add_argument('-mcv', '--multi_cam_views', default=3, type=int,
                        help='number of views in semi-supervised branch. mcv==1:single-view, 1<mcv<=4: multi-view')
    parser.add_argument('-mce-t', '--mce_type', default=-1.1, type=float,
                        help='mce formula; -4:PA-MPCE, -3:Fb-UVec-BLen, -2:Fb-UVec, -1:Fb-Vec, 0:Disable, 1:Ach-2-Each-Pos, 2:Adjacent-Pairs'
                             'if negative with decimal, whole number is for no-cam-ext and decimal number for with-cam-ext')
    parser.add_argument('-per-prop', '--per-bone-proportion', action='store_false', dest='group_bpc',
                        help='disable bpc grouping of symmetric bone ratios or proportion pair')
    parser.add_argument('-per-jnt', '--per-joint-transform', action='store_false', dest='group_jmc',
                        help='disable jmc grouping of symmetric joints or enable per joint transformation for jmc')
    parser.add_argument('-n-ratios', '--n-bone-ratios', default=15, type=int,
                        help='number of bone ratios used for bone-proportion-error')
    #parser.add_argument('-sup-coef', '--supervised-loss-coef', default='1.,1.', type=str, metavar='SUP_COEF',
    #                    help='supervised branch loss coefficients/weights separated by comma; p3d,trj')
    parser.add_argument('-semi-sup-coef', '--semi-supervised-loss-coef', default='1.,1.,0.,0.,0.,0.,0.,0.', type=str, metavar='SEMI_SUP_COEF',
                        help='semi-supervised branch loss coefficients/weights separated by comma; rp2d,mble,pto,bse,bpc,jmc')
    parser.add_argument('--multi-gpu', action='store_true', dest='multi_gpu_training',
                        help='enable data-distributed training on multi-gpus in a node, if and when available')
    parser.add_argument('-gen-pp', '--gen_pose_priors', default=1, type=int,
                        help='compute train-set pose priors properties. -3:numpy_op-gen-&-save, -2:torch_op-gen-&-save'
                             '-1:torch_op-runtime-gen-from-subset, 0:use-superset-pre-gen-priors, 1:use-subset-pre-gen-priors')
    parser.add_argument('-log-f', '--logli-func-mode', default=1, type=int,
                        help='modified log-likelihood function. '
                             '0:disable/vanilla-log-likelihood, 1:stretch-log-likelihood, 2:log-of-inverse-likelihood')
    parser.add_argument('-norm-log-f', '--norm-logli-func-mode', default=0, type=int,
                        help='normalize bpc & jmc log-likelihhod. '
                             '0:disable/no-normalization, 1:nmsl, 2:nmml, 3:nilm')
    parser.add_argument('-rev-log-f', '--reverse-logli-func-mode', default=1, type=int,
                        help='function to reverse or invert the curve of log-likelihood. '
                             '0:disable/identity, 1:(-)log-likelihood, 2:inverse-log-likelihood')
    parser.add_argument('-sft-log-f', '--shift-logli-func-mode', default=0, type=int,
                        help='vertical shift modification that translates the resulting curve upwards. '
                             '0:disable/identity, 1:log-likelihood(+)move_up_constant')
    parser.add_argument('-l-sprd', '--log-likelihood-spread', default=1., type=float,
                        help='stretch-log-likelihood func. spread constant to be used when generating priors')
    parser.add_argument('-l-eps', '--likelihood-epsilon', default=1., type=float,
                        help='constant 1e-5<=x<=1e-0 that is used to compute inverse of likelihood, preventing nans')
    parser.add_argument('-ll-eps', '--log-likelihood-epsilon', default=1e-5, type=float,
                        help='constant 1e-5<=x<=1e-0 that is used to compute inverse of log-likelihood, preventing nans')
    parser.add_argument('-tag', '--dir-tag-prefix', default='', type=str, metavar='CKPT_DIR',
                        help='prefix of tag included in checkpoint directory name')
    parser.add_argument('--log-losses-numpy', action='store_true', help='save value of all losses as a .npy array')
    parser.add_argument('--save-model', action='store_true', help='save model after last epoch')
    # end

    # Visualization
    parser.add_argument('--viz-subject', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
    parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')

    parser.set_defaults(mean_bone_length_term=True)
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True) # TODO: Turn-off for original H36M when not comparing to VideoPose3D
    # added by Lawrence 03/25/2021
    parser.set_defaults(group_bpc=True)
    parser.set_defaults(group_jmc=True)
    parser.set_defaults(tune_hyper_parameters=False)
    parser.set_defaults(supervision_with_pose_regularizers=False)
    parser.set_defaults(reposition_to_origin=True)
    parser.set_defaults(pelvis_placement_term=False)
    parser.set_defaults(bone_symmetry_term=False)
    parser.set_defaults(bone_proportion_term=False)
    parser.set_defaults(joint_mobility_term=False)
    parser.set_defaults(multi_gpu_training=False)

    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()

    # Enforce restrictions
    assert (1<=args.multi_cam_views<=4), 'For H36M, multi_cam_views cannot be less than 1 or greater than 4'
    assert ((args.mce_type!=0 or args.multi_cam_views==1) and (args.multi_cam_views!=1 or args.mce_type==0))
    # semi_coefs = np.float32([float(x) for x in args.semi_supervised_loss_coef.split(',')])
    # assert (~(args.subjects_unlabeled!='' and np.sum(semi_coefs)==0) or args.warmup>=args.epochs)
    # no_pose_reg = ~(args.pelvis_placement_term or args.bone_symmetry_term or
    #                 args.bone_proportion_term or args.joint_mobility_term)
    # assert (np.sum(semi_coefs)!=0 or (args.projection_type==0 and args.mce_type==0 and no_pose_reg))

    return args