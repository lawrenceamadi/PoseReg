# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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
    def __init__(self, batch_size, cams_intrinsic, cams_extrinsic, poses_3d, poses_2d, chunk_length,
                 pad=0, causal_shift=0, shuffle=True, random_seed=1234, augment=False, multi_cams=1, mce_flip=False,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None, endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cams_intrinsic is None or len(cams_intrinsic) == len(poses_2d)
        assert cams_extrinsic is None or len(cams_extrinsic) == len(poses_2d)
    
        # Build lineage info
        # Note: every 4 consecutive seq_idx are poses of the same activity sequence but from the 4 different cameras
        # for example sequence indexes: 0,1,2,3 - 4,5,6,7 - 16,17,18,19 ...
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            #print('chunk_length:{}, n_chunks:{}, offset:{}, bounds:{}'.format(chunk_length, n_chunks, offset, bounds)) #del
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # multi-view/cam configuration
        self.multi_cams = multi_cams
        self.mce_for_flip_aug = mce_flip
        self.append_multi_views = poses_3d is None and multi_cams>1
        extra_batch_size = batch_size * multi_cams

        # Initialize buffers
        if cams_intrinsic is not None:
            self.batch_cam_in = np.empty((extra_batch_size, cams_intrinsic[0].shape[-1]))
        if cams_extrinsic is not None:
            self.batch_cam_ex = np.empty((extra_batch_size, cams_extrinsic[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((extra_batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_2d = np.empty((extra_batch_size, chunk_length + 2*pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random_epoch = np.random.RandomState(random_seed)
        self.random_views = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cams_intrinsic = cams_intrinsic
        self.cams_extrinsic = cams_extrinsic
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random_epoch
    
    def set_random_state(self, random):
        self.random_epoch = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random_epoch.permutation(self.pairs) # (?,4)
            else:
                pairs = self.pairs # (?,4)
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                #multi_view_indices = []
                #mv_anchor_indices = [] # index of multi-view anchor in true batch
                true_batch_indices = [] # fixed size can be implemented with a buffer like self.batch_2d
                nonflip_mv_indices = []
                flipped_mv_indices = []

                idx = 0
                for chk_idx, (seq_idx, start_3d, end_3d, flip) in enumerate(chunks):
                #for (seq_idx, start_3d, end_3d, flip) in chunks:
                    #*print("seq_idx:{}, start_3d:{}, end_3d:{}, flip:{}".format(seq_idx, start_3d, end_3d, flip))
                    matching_seq_set = [seq_idx]
                    if self.append_multi_views:
                        true_batch_indices.append(idx)
                        # choose s matching sequence from other camera viewpoints
                        if self.mce_for_flip_aug or flip==0: # ie. generating for semi-supervised branch without flipping
                            matching_cams_seq = [*range(seq_idx//4 * 4, seq_idx//4 * 4 + 4)]
                            matching_cams_seq.remove(seq_idx)
                            matching_cams_seq = self.random_views.permutation(matching_cams_seq)
                            matching_seq_set.extend(matching_cams_seq[:self.multi_cams-1])
                            if flip==0:
                                nonflip_mv_indices.extend([*range(idx, idx+self.multi_cams)])
                            else: flipped_mv_indices.extend([*range(idx, idx+self.multi_cams)])
                            #multi_view_indices.extend([*range(idx, idx+self.multi_cams)])
                            #*print("seq_idx:{} - matching_cams_seq:{}".format(seq_idx, matching_cams_seq))`
                            #*print("seq_idx:{} - matching_seq_set:{}".format(seq_idx, matching_seq_set))
                            #*print("shuffled matching_cams_seq:{}".format(matching_cams_seq))
                            #*print("multi_cams:{} - matching_cams_seq[:multi_cams]:{}".format(self.multi_cams, matching_cams_seq[:self.multi_cams]))
                            #*print("multi_view_indices:{}".format(multi_view_indices))
                            #*print("-----------------------")
                            #mv_anchor_indices.append(chk_idx)
                        #*else: flipped_set_indices.append(idx)

                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift

                    for seq_i in matching_seq_set:
                        # 2D poses
                        seq_2d = self.poses_2d[seq_i]
                        low_2d = max(start_2d, 0)
                        high_2d = min(end_2d, seq_2d.shape[0])
                        pad_left_2d = low_2d - start_2d
                        pad_right_2d = end_2d - high_2d
                        if pad_left_2d != 0 or pad_right_2d != 0:
                            self.batch_2d[idx] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_2d[idx] = seq_2d[low_2d:high_2d]

                        # todo: if append_multi_views and flip, for other positive views, toss a coin p_flip, to determine whether or not to flip
                        if flip:
                            # Flip 2D keypoints
                            self.batch_2d[idx, :, :, 0] *= -1
                            self.batch_2d[idx, :, self.kps_left + self.kps_right] = self.batch_2d[idx, :, self.kps_right + self.kps_left]

                        # 3D poses
                        if self.poses_3d is not None:
                            seq_3d = self.poses_3d[seq_i]
                            low_3d = max(start_3d, 0)
                            high_3d = min(end_3d, seq_3d.shape[0])
                            pad_left_3d = low_3d - start_3d
                            pad_right_3d = end_3d - high_3d
                            if pad_left_3d != 0 or pad_right_3d != 0:
                                self.batch_3d[idx] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                            else:
                                self.batch_3d[idx] = seq_3d[low_3d:high_3d]

                            if flip:
                                # Flip 3D joints
                                self.batch_3d[idx, :, :, 0] *= -1
                                self.batch_3d[idx, :, self.joints_left + self.joints_right] = \
                                        self.batch_3d[idx, :, self.joints_right + self.joints_left]

                        # Cameras Intrinsic Params
                        if self.cams_intrinsic is not None:
                            self.batch_cam_in[idx] = self.cams_intrinsic[seq_i]
                            if flip:
                                # Flip horizontal distortion coefficients
                                self.batch_cam_in[idx, 2] *= -1
                                self.batch_cam_in[idx, 7] *= -1

                        # Cameras Extrinsic Params
                        if self.cams_extrinsic is not None:
                            self.batch_cam_ex[idx] = self.cams_extrinsic[seq_i]

                        idx += 1

                # # debug visuals
                # #--------------------------------------------------------------------------------------------
                # if self.append_multi_views:
                #     #*print("flipped_set_indices:{} {:.1f}%".
                #     #*      format(flipped_set_indices, len(flipped_set_indices)/len(true_batch_indices)*100))
                #     print("-----------------------")
                #     import torch
                #     from common.quaternion import qrot
                #     from agents.visuals import plot_3d_pose
                #     from agents.helper import KPT_2_IDX
                #     from agents.jme_torch import torch_t
                #     from matplotlib import pyplot as plt
                #     from mpl_toolkits import mplot3d
                #
                #     # multi-view pose and trajectory consistency loss
                #     mv_3d_pos = torch_t(self.batch_3d[multi_view_indices]) # (?>>,f,17,3)
                #     #mv_semi_predicted_traj = all_semi_predicted_traj[mvi_semi] # (?>>,f,1,3)
                #     cam_ex = torch_t(self.batch_cam_ex[multi_view_indices])
                #     mv_cam_ex_semi = torch.reshape(cam_ex, (-1, 1, 1, 7)) # (?>>,7)->(?>>,1,1,7)
                #     mv_cam_ex_semi = torch.tile(mv_cam_ex_semi, (1, 1, 17, 1)) # (?>>,f,17,3)
                #     #camfrm_predicted_poses = mv_semi_predicted_3d_pos + mv_semi_predicted_traj # (?>>,f,17,3)
                #     camfrm_poses = torch.clone(mv_3d_pos)
                #     camfrm_poses[:,:,1:,:] += camfrm_poses[:,:,[0],:]
                #     wldfrm_poses = qrot(mv_cam_ex_semi[:,:,:,:4], camfrm_poses) + mv_cam_ex_semi[:,:,:,4:] # (?>>,f,17,3)
                #
                #     same_wld_poses = torch.reshape(wldfrm_poses, (-1, 2, 1, 17, 3)) # (?>>,m,f,17,3)
                #     pose_set_1 = same_wld_poses.detach().cpu().clone().numpy() * 1000
                #     pose_set_1 -= pose_set_1[:,:,:,[0],:]
                #
                #     same_cam_poses = torch.reshape(mv_3d_pos, (-1, 2, 1, 17, 3)) # (?>>,m,f,17,3)
                #     pose_set_0 = same_cam_poses.detach().cpu().clone().numpy() * 1000
                #     pose_set_0[:,:,:,0,:] = 0
                #
                #     n_rows, n_cols = 2, self.multi_cams
                #     for s in range(len(multi_view_indices)//(self.multi_cams)):
                #         SP_FIG, SP_AXS = plt.subplots(n_rows, n_cols, subplot_kw={'projection':'3d'}, figsize=(3*n_cols, 4*n_rows))
                #         SP_FIG.subplots_adjust(left=0.0, right=1.0, wspace=-0.0)
                #         for i in range(self.multi_cams):
                #             plot_3d_pose(pose_set_0[s,i,0], SP_FIG, SP_AXS[0,i], 'Cam-View {}'.format(i),
                #                          KPT_2_IDX, [-1]*4)
                #             plot_3d_pose(pose_set_1[s,i,0], SP_FIG, SP_AXS[1,i], 'Wld-View {}'.format(i),
                #                          KPT_2_IDX, [-1]*4, display=i==1)
                # #--------------------------------------------------------------------------------------------

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cams_intrinsic is None:
                    yield None, None, self.batch_2d[:len(chunks)]
                elif self.poses_3d is not None and self.cams_intrinsic is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                #del*elif self.poses_3d is not None and self.append_multi_views:
                elif self.poses_3d is None: # semi-supervised training mode?
                    #yield self.batch_cam_in[:len(chunks)], None, self.batch_2d[:len(chunks)]
                    assert (len(chunks)<=self.batch_size)
                    assert (len(nonflip_mv_indices)%(self.multi_cams) == 0)
                    assert (len(flipped_mv_indices)%(self.multi_cams) == 0)
                    assert (~self.append_multi_views or (len(true_batch_indices)==len(chunks)))
                    assert (~self.append_multi_views or (len(flipped_mv_indices)+len(nonflip_mv_indices)==idx))
                    assert ((len(flipped_mv_indices)+len(nonflip_mv_indices))/self.multi_cams==len(true_batch_indices))
                    yield true_batch_indices, nonflip_mv_indices, flipped_mv_indices, \
                      self.batch_cam_ex[:idx], self.batch_cam_in[:idx], None, self.batch_2d[:idx]
                    #del*yield true_batch_indices, multi_view_indices, self.batch_cam_ex[:idx], \
                    #del*      self.batch_cam_in[:idx], self.batch_3d[:idx], self.batch_2d[:idx]
                else:
                    yield self.batch_cam_in[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
            
            if self.endless:
                self.state = None
            else:
                enabled = False
            

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
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
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
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam_in, batch_3d, batch_2d