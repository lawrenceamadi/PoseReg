# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Extensive modification of VideoPose3D source code
# by researchers at the Visual Computing Lab @ IIT

from itertools import zip_longest
import numpy as np


class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cams_intrinsic -- list of cameras' intrinsic, one element for each video (optional, used for semi-supervised training)
    cams_extrinsic -- list of cameras' extrinsic, one element for each video (optional, partial use in semi-supervised branch)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    multi_cams -- number of additional views of sequence to include in batch (0,1,2,or 3, partial use in semi-supervised branch)
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cams_intrinsic, cams_extrinsic, poses_3d, poses_2d, gt_poses_2d, chunk_length,
                 pad=0, causal_shift=0, shuffle=True, random_seed=1234, augment=False, rot_ang=0., multi_cams=1, mce_flip=False,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None, endless=False):
        
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cams_intrinsic is None or len(cams_intrinsic) == len(poses_2d)
        assert cams_extrinsic is None or len(cams_extrinsic) == len(poses_2d)
        self.ws_mode = endless #poses_3d is None # weakly supervised mode
        assert (not self.ws_mode or gt_poses_2d is not None), 'ws_mode:{}-->{}'.format(self.ws_mode, gt_poses_2d is not None)
        assert (multi_cams==1 or self.ws_mode), 'multi_cams:{}>1 --> ws_mode:{} ?'.format(multi_cams, self.ws_mode)
        self.append_multi_views = self.ws_mode and multi_cams>1
        assert (not self.append_multi_views or batch_size%multi_cams==0)

        self.cams_intrinsic = cams_intrinsic
        self.cams_extrinsic = cams_extrinsic
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.gt_poses_2d = gt_poses_2d

        self.augment = augment
        self.rot_aug = rot_ang>0
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        # multi-view/cam configuration
        self.multi_cams = multi_cams
        self.adj_batch_sz = batch_size * multi_cams
        self.flip_aug_for_posture_mce = mce_flip

        if self.ws_mode:
            self.pairs = self.compile_ws_sample_tuples(chunk_length, random_seed)
        else: self.pairs = self.compile_sample_tuples(chunk_length)
        self.chunk_length = chunk_length
        self.chunk_length_2dp = chunk_length + 2*pad

        # Initialize buffers
        if cams_intrinsic is not None:
            self.batch_cam_in = np.empty((self.adj_batch_sz, cams_intrinsic[0].shape[-1]))
        if cams_extrinsic is not None:
            self.batch_cam_ex = np.empty((self.adj_batch_sz, cams_extrinsic[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((self.adj_batch_sz, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1])) # (?,f,j,3)
        self.batch_2d = np.empty((self.adj_batch_sz, self.chunk_length_2dp, poses_2d[0].shape[-2], poses_2d[0].shape[-1])) # (?,f',j,2)
        self.with_gt2d = gt_poses_2d is not None
        if self.with_gt2d:
            self.batch_gt2d = np.empty((self.adj_batch_sz, self.chunk_length_2dp, gt_poses_2d[0].shape[-2], gt_poses_2d[0].shape[-1]))

        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.num_samples() / self.adj_batch_sz))
        self.random_epoch = np.random.RandomState(random_seed)

        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.data_epochs = 0
        self.endless = endless
        self.state = None

        self.homg_2dps = np.ones((self.chunk_length_2dp, poses_2d[0].shape[-2], 3), dtype=np.float32) # (f',j,3)
        self.rot_mtx_2x3 = np.zeros((self.chunk_length_2dp, 2, 3), dtype=np.float32) # (f',2,3)
        self.rot_mtx_3x3 = np.zeros((1, 3, 3), dtype=np.float32) # (f',3,3)
        self.rot_mtx_3x3[0, 2, 2] = 1
        self.rot_ang_lb = np.deg2rad(0)#-rot_ang)
        self.rot_ang_ub = np.deg2rad(rot_ang) + 1e-5
        self.rot_angle_rs = np.random.RandomState(random_seed)

    def steps_per_data_epoch(self):
        return self.num_batches

    def num_samples(self):
        if self.ws_mode:
            return len(self.pairs) * self.multi_cams
        else: return len(self.pairs)
    
    def random_state(self):
        return self.random_epoch
    
    def set_random_state(self, random):
        self.random_epoch = random
        
    def augment_enabled(self):
        return self.augment
    
    def compile_sample_tuples(self, chunk_length):
        # Build lineage info
        # Note: every 4 consecutive seq_idx are poses of the same activity sequence but from the 4 different cameras
        # for example sequence indexes: 0,1,2,3 - 4,5,6,7 - 16,17,18,19 ...
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(self.poses_2d)):
            assert (self.poses_3d is None or self.poses_3d[i].shape[0] == self.poses_2d[i].shape[0])
            n_chunks = (self.poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - self.poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds)-1, False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds)-1), bounds[:-1], bounds[1:], augment_vector)
            if self.augment:
                pairs += zip(np.repeat(i, len(bounds)-1), bounds[:-1], bounds[1:], ~augment_vector)
        return pairs

    def compile_ws_sample_tuples(self, chunk_length, random_seed):
        # Build lineage info for weakly supervised branch
        # Note: every 4 consecutive seq_idx are poses of the same activity sequence but from the 4 different cameras
        # for example sequence indexes: 0,1,2,3 - 4,5,6,7 - 16,17,18,19 ...
        self.cam_views_rs = np.random.RandomState(random_seed)
        pairs = [] # ([seq_idx_c1,...,seq_idx_ci], start_frame, end_frame, flip) tuples
        for cam_seq_idx in range(0, len(self.poses_2d), 4):
            assert (self.poses_3d is None or self.poses_3d[cam_seq_idx].shape[0] == self.poses_2d[cam_seq_idx].shape[0])
            n_chunks = (self.poses_2d[cam_seq_idx].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - self.poses_2d[cam_seq_idx].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            n_samples_from_seq = len(bounds) - 1
            augment_vector = np.full(n_samples_from_seq, False, dtype=bool)
            # rot_aug_vector = np.full(n_samples_from_seq, False, dtype=bool)
            # generate shuffled camera/view order for each sequence
            cam_view_seq_indexes = np.arange(cam_seq_idx, cam_seq_idx+4)
            seq_cam_views = []
            for seq_clip_s_idx in range(n_samples_from_seq):
                seq_cam_views.append(self.cam_views_rs.permutation(cam_view_seq_indexes))
            seq_cam_views = np.asarray(seq_cam_views)
            # add to sample tuple list
            for idx in range(4//self.multi_cams):
                s_i = idx * self.multi_cams
                e_i = (idx+1) * self.multi_cams
                pairs.extend(zip(seq_cam_views[:, s_i:e_i], bounds[:-1], bounds[1:], augment_vector))
                if self.augment:
                    pairs.extend(zip(seq_cam_views[:, s_i:e_i], bounds[:-1], bounds[1:], ~augment_vector))
        return np.asarray(pairs, dtype=object)

    def refresh_mv_groupings(self):
        # generate new permutations for multiview groups
        # self.pairs: ([seq_idx_c1,...,seq_idx_ci], start_frame, end_frame, flip) tuples
        s_idx = 0
        for cam_seq_idx in range(0, len(self.poses_2d), 4):
            n_chunks = (self.poses_2d[cam_seq_idx].shape[0] + self.chunk_length - 1) // self.chunk_length
            offset = (n_chunks * self.chunk_length - self.poses_2d[cam_seq_idx].shape[0]) // 2
            bounds = np.arange(n_chunks+1) * self.chunk_length - offset
            n_samples_from_seq = len(bounds) - 1
            if self.augment: n_samples_from_seq *= 2
            # generate shuffled camera/view order for each sequence
            cam_view_seq_indexes = np.arange(cam_seq_idx, cam_seq_idx+4)
            for seq_clip_s_idx in range(n_samples_from_seq):
                seq_clip_cam_views = self.cam_views_rs.permutation(cam_view_seq_indexes)
                # add to sample tuple list
                for idx in range(4//self.multi_cams):
                    s_i = idx * self.multi_cams
                    e_i = (idx+1) * self.multi_cams
                    smp_idx = s_idx + (n_samples_from_seq * idx) + seq_clip_s_idx
                    self.pairs[smp_idx][0] = seq_clip_cam_views[s_i:e_i]
            s_idx += n_samples_from_seq * (4//self.multi_cams)
        #assert(len(self.pairs)==s_idx), '{:,} vs. {:,}'.format(len(self.pairs), s_idx)
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                if self.ws_mode and self.data_epochs>0: self.refresh_mv_groupings()
                pairs = self.random_epoch.permutation(self.pairs) # (?,4)
            else: pairs = self.pairs # (?,4)
            return 0, pairs
        else: return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            batch_idx = 0
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                batch_idx = b_i
                # chunks = pairs[b_i*self.build_batch : (b_i+1)*self.build_batch]
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                unaltered_mv_indices = []
                augmented_mv_indices = []

                idx = 0
                for chk_idx, (seq_idx, start_3d, end_3d, augment) in enumerate(chunks):
                    matching_seq_set = seq_idx if self.ws_mode else [seq_idx]
                    if self.append_multi_views:
                        # log batch indexes of flipped or non-flipped multiview samples
                        if self.flip_aug_for_posture_mce and augment:
                            augmented_mv_indices.extend([*range(idx, idx+self.multi_cams)])
                        else: unaltered_mv_indices.extend([*range(idx, idx+self.multi_cams)])
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift

                    for seq_i in matching_seq_set:
                        # 2D poses
                        seq_2d = self.poses_2d[seq_i]
                        if self.with_gt2d: seq_gt2d = self.gt_poses_2d[seq_i]
                        low_2d = max(start_2d, 0)
                        high_2d = min(end_2d, seq_2d.shape[0])
                        pad_left_2d = low_2d - start_2d
                        pad_right_2d = end_2d - high_2d
                        if pad_left_2d != 0 or pad_right_2d != 0:
                            self.batch_2d[idx] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                            if self.with_gt2d:
                                self.batch_gt2d[idx] = np.pad(seq_gt2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_2d[idx] = seq_2d[low_2d:high_2d]
                            if self.with_gt2d: self.batch_gt2d[idx] = seq_gt2d[low_2d:high_2d]

                        if augment:
                            # Flip 2D keypoints
                            self.batch_2d[idx, :, :, 0] *= -1
                            self.batch_2d[idx, :, self.kps_left + self.kps_right] = self.batch_2d[idx, :, self.kps_right + self.kps_left]
                            if self.with_gt2d:
                                self.batch_gt2d[idx, :, :, 0] *= -1
                                self.batch_gt2d[idx, :, self.kps_left + self.kps_right] = self.batch_gt2d[idx, :, self.kps_right + self.kps_left]

                            if self.rot_aug:
                                # Rotation Augmentation
                                if self.batch_2d[idx, self.pad, 3, 1] > self.batch_2d[idx, self.pad, 6, 1]: # R vs. L Ankle
                                    ankle_kpt_idx, rot_angle_sign = 3, 1 # rotate about RAk kpt, clockwise
                                else: ankle_kpt_idx, rot_angle_sign = 6, -1 # rotate about LAk kpt, counter-clockwise

                                rot_angle_rad = self.rot_angle_rs.uniform(self.rot_ang_lb, self.rot_ang_ub) * rot_angle_sign
                                plv_x_pts = np.copy(self.batch_2d[idx, :, 0:1, 0]) # (?,f',j,2) -> (f',1), NOT [idx, :, [0], 0]
                                cos_ang = np.cos(rot_angle_rad)
                                sin_ang = np.sin(rot_angle_rad)
                                self.rot_mtx_2x3[:, 0, 0] = cos_ang
                                self.rot_mtx_2x3[:, 1, 1] = cos_ang
                                self.rot_mtx_2x3[:, 1, 0] = sin_ang
                                self.rot_mtx_2x3[:, 0, 1] = -sin_ang
                                x_pts = self.batch_2d[idx, :, ankle_kpt_idx, 0] # rotate about ankle kpt # (f')
                                y_pts = self.batch_2d[idx, :, ankle_kpt_idx, 1] # rotate about ankle kpt # (f')
                                self.rot_mtx_2x3[:, 0, 2] = x_pts - x_pts*cos_ang + y_pts*sin_ang
                                self.rot_mtx_2x3[:, 1, 2] = y_pts - x_pts*sin_ang - y_pts*cos_ang
                                self.homg_2dps[:, :, :2] = self.batch_2d[idx]
                                self.batch_2d[idx] = np.transpose(
                                    np.matmul(self.rot_mtx_2x3, np.transpose(self.homg_2dps, axes=(0,2,1))), axes=(0,2,1))
                                self.batch_2d[idx, :, :, 0] += plv_x_pts - self.batch_2d[idx, :, 0:1, 0]  # --> a + b - a = b

                                if self.with_gt2d:
                                    plv_x_pts = np.copy(self.batch_gt2d[idx, :, 0:1, 0]) # (f',1)
                                    x_pts = self.batch_gt2d[idx, :, ankle_kpt_idx, 0] # rotate about ankle kpt # (f')
                                    y_pts = self.batch_gt2d[idx, :, ankle_kpt_idx, 1] # rotate about ankle kpt # (f')
                                    self.rot_mtx_2x3[:, 0, 2] = x_pts - x_pts*cos_ang + y_pts*sin_ang
                                    self.rot_mtx_2x3[:, 1, 2] = y_pts - x_pts*sin_ang - y_pts*cos_ang
                                    self.homg_2dps[:, :, :2] = self.batch_gt2d[idx]
                                    self.batch_gt2d[idx] = np.transpose(
                                        np.matmul(self.rot_mtx_2x3, np.transpose(self.homg_2dps, axes=(0,2,1))), axes=(0,2,1))
                                    self.batch_gt2d[idx, :, :, 0] += plv_x_pts - self.batch_gt2d[idx, :, 0:1, 0] # --> a + b - a = b

                        # 3D poses
                        if self.poses_3d is not None:
                            seq_3d = self.poses_3d[seq_i]
                            low_3d = max(start_3d, 0)
                            high_3d = min(end_3d, seq_3d.shape[0])
                            pad_left_3d = low_3d - start_3d
                            pad_right_3d = end_3d - high_3d
                            if pad_left_3d != 0 or pad_right_3d != 0:
                                self.batch_3d[idx] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                            else: self.batch_3d[idx] = seq_3d[low_3d:high_3d]

                            if augment:
                                # Flip 3D joints
                                self.batch_3d[idx, :, :, 0] *= -1 # (?,f,j,3)
                                self.batch_3d[idx, :, self.joints_left + self.joints_right] = \
                                        self.batch_3d[idx, :, self.joints_right + self.joints_left]

                                if self.rot_aug:
                                    self.rot_mtx_3x3[0,:2,:2] = self.rot_mtx_2x3[0,:2,:2]
                                    self.batch_3d[idx, :, 1:] = np.transpose(
                                        np.matmul(self.rot_mtx_3x3, np.transpose(self.batch_3d[idx], axes=(0,2,1))), axes=(0,2,1))[:,1:]

                        # Cameras Intrinsic Params
                        if self.cams_intrinsic is not None:
                            self.batch_cam_in[idx] = self.cams_intrinsic[seq_i]
                            if augment:
                                # Flip horizontal distortion coefficients
                                self.batch_cam_in[idx, 2] *= -1
                                self.batch_cam_in[idx, 7] *= -1

                        # Cameras Extrinsic Params
                        if self.cams_extrinsic is not None:
                            self.batch_cam_ex[idx] = self.cams_extrinsic[seq_i]

                        idx += 1

                if self.endless: self.state = (b_i + 1, pairs)

                if self.poses_3d is None and self.cams_intrinsic is None:
                    yield None, None, self.batch_2d[:len(chunks)]
                elif self.poses_3d is not None and self.cams_intrinsic is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                elif self.ws_mode: # weakly-supervised training mode?
                    yield unaltered_mv_indices, augmented_mv_indices, \
                          self.batch_cam_ex[:idx], self.batch_cam_in[:idx], self.batch_2d[:idx], self.batch_gt2d[:idx]
                else:
                    yield self.batch_cam_in[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]

            assert (batch_idx*self.adj_batch_sz <= self.num_samples() and self.num_samples() <= (batch_idx+1)*self.adj_batch_sz), \
                'Is {} <= {} <= {} ?'.format(batch_idx*self.adj_batch_sz, self.num_samples(), (batch_idx+1)*self.adj_batch_sz)
            self.data_epochs += 1
            if self.endless:
                self.state = None
            else: enabled = False
            

class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cams_intrinsic -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cams_intrinsic, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cams_intrinsic is None or len(cams_intrinsic) == len(poses_2d)

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cams_intrinsic = [] if cams_intrinsic is None else cams_intrinsic
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cams_intrinsic, self.poses_3d, self.poses_2d):
            batch_cam_in = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0) # (1,frames,j,3)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam_in is not None:
                    batch_cam_in = np.concatenate((batch_cam_in, batch_cam_in), axis=0)
                    batch_cam_in[1, 2] *= -1
                    batch_cam_in[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0) # (2,frames,j,3)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam_in, batch_3d, batch_2d