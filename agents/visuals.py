# -*- coding: utf-8 -*-
# @Time    : 5/12/2021 5:32 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : visuals.py
# @Software: pose.reg

import sys
import matplotlib
from sklearn.covariance import empirical_covariance
from sklearn.covariance import MinCovDet, EmpiricalCovariance

sys.path.append('../')
from agents.pose_regs import *

# Added by Lawrence (04/14/2021) for cross OS compatibility
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg','Agg']
for gui in gui_env:
    try:
        matplotlib.use(gui, warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue

from mpl_toolkits import mplot3d # <--- This is important for 3d plotting
from matplotlib import animation



def inlier_subset(data_pts, pts_distance):
    q1, q3 = np.quantile(pts_distance, [0.25, 0.75])  # compute Q1 & Q3
    iqr = q3 - q1  # compute inter-quartile-range
    outlier_error_thresh = q3 + 1.5*iqr
    is_inlier = pts_distance <= outlier_error_thresh
    n_inliers = np.sum(is_inlier.astype(np.int32))
    inlier_indexes = np.squeeze(np.argwhere(is_inlier))
    cluster_inlier_pts = data_pts[inlier_indexes]
    return cluster_inlier_pts, n_inliers/len(cluster_inlier_pts), inlier_indexes


def plot_2d_pose(points2d, sp_tag, kpt_2_idx, fig_title='Normalized 2D Pose',
                 axis_len=0.75, display=True, overlay=False, thickness=2.0):
    n_ps = len(points2d)
    n_sf = len(points2d)+1 if overlay else len(points2d)
    n_lp = len(points2d)+2 if overlay else len(points2d)
    if overlay: sp_tag += ['Overlay']
    sp_fig, sp_axes = plt.subplots(nrows=1, ncols=n_sf, figsize=(n_sf*6, 5))
    sp_fig.suptitle(fig_title, y=1., fontweight='bold', size=12)

    for i in range(n_lp):
        if n_sf==1: sp_axs = sp_axes
        else:
            if i>=n_ps: sp_axs = sp_axes[n_ps]
            else: sp_axs = sp_axes[i]

        if i<n_sf:
            sp_axs.clear()
            sp_axs.set_title(sp_tag[i], fontweight='bold', size=10)
            sp_axs.set(xlim=(-axis_len,axis_len), ylim=(-axis_len,axis_len))
            sp_axs.set_yticklabels([]), sp_axs.set_xticklabels([]) # turn off tick labels
            # Draw x,y axis across frame(cartesian coordinate) origin
            sp_axs.plot([-axis_len,0], [0,0], 'red',  linestyle='-.', linewidth=0.6) # - x-axis
            sp_axs.plot([0, axis_len], [0,0], 'red',  linestyle='--', linewidth=1.2) # + x-axis
            sp_axs.plot([0,0], [-axis_len,0], 'blue', linestyle='-.', linewidth=0.6) # - y-axis
            sp_axs.plot([0,0], [0, axis_len], 'blue', linestyle='--', linewidth=1.2) # + y-axis

        j = i
        if i==n_ps: j, color, thickness = i-1, 'royalblue', 3.0 # i-1 or 3
        elif i==n_sf: j, color, thickness = i-3, 'black', 1.5 # i-3 or 2
        xKpts = points2d[j][:, 0]
        yKpts = points2d[j][:, 1]

        # Draw skeletal limbs (lines between keypoints)
        for limb, kptPair in BONE_KPT_PAIRS.items():
            kptA, kptB = kptPair
            # color based on right, left, or center body part
            if i<n_ps:
                if kptA[0]=='R' or kptB[0]=='R': color = 'olive'
                elif kptA[0]=='L' or kptB[0]=='L': color = 'purple'
                else: color = 'black'
            kptAIdx, kptBIdx = kpt_2_idx[kptA], kpt_2_idx[kptB]
            x_pts = [xKpts[kptAIdx], xKpts[kptBIdx]]
            y_pts = [yKpts[kptAIdx], yKpts[kptBIdx]]
            sp_axs.plot(x_pts, y_pts, color, linewidth=thickness)

    if display:
        plt.show(block=True)
        plt.close(sp_fig)


def plot_3d_pose(points3d, sp_fig, sp_axs, sp_tag, kpt_2_idx, jnt_quad_kpts, axis_len=0.75,
                 t_tag='', azim_deg=30, elev_deg=30, clear=True, uni_color=False, display=False):
    # draw y-component on z-axis and z-component on y-axis (xyz min:-890 max:913)
    if sp_fig is None and sp_axs is None:
        sp_fig, sp_axs = plt.subplots(1, 1, subplot_kw={'projection':'3d'}, figsize=(5,5))
    xKpts = points3d[:, 0]
    yKpts = points3d[:, 1]
    zKpts = points3d[:, 2]
    fig_title = '{}3D Pose Transformation for Each Bone Alignment'.format(t_tag)
    sp_fig.suptitle(fig_title, y=1., fontweight='bold', size=12)
    if clear: sp_axs.clear()
    sp_axs.set_title(sp_tag, fontweight='bold', size=10)
    sp_axs.set(#xlabel='x', ylabel='y', zlabel='z', # components order x-y-z
               xlim=(-axis_len,axis_len), zlim=(-axis_len,axis_len), ylim=(-axis_len,axis_len))
    sp_axs.set_yticklabels([]), sp_axs.set_xticklabels([]), sp_axs.set_zticklabels([]) # turn off tick labels
    sp_axs.view_init(azim=azim_deg, elev=elev_deg, vertical_axis='y') # Defaults: elev=30 azim=-60 roll=0 vertical_axis='z'
    sp_axs.set_box_aspect(aspect=(1,1,1))

    # Draw x,y,z axis across frame(cartesian coordinate) origin
    sp_axs.plot([-axis_len,0], [0,0], [0,0], 'red',    zdir='z', linestyle='-.', linewidth=0.6) # - x-axis
    sp_axs.plot([0, axis_len], [0,0], [0,0], 'red',    zdir='z', linestyle='--', linewidth=1.2) # + x-axis
    sp_axs.plot([0,0], [-axis_len,0], [0,0], 'blue',   zdir='z', linestyle='-.', linewidth=0.6) # - y-axis
    sp_axs.plot([0,0], [0, axis_len], [0,0], 'blue',   zdir='z', linestyle='--', linewidth=1.2) # + y-axis
    sp_axs.plot([0,0], [0,0], [-axis_len,0], 'orange', zdir='z', linestyle='-.', linewidth=0.6) # - z-axis
    sp_axs.plot([0,0], [0,0], [0, axis_len], 'orange', zdir='z', linestyle='--', linewidth=1.2) # + z-axis

    # Draw skeletal limbs (lines between keypoints)
    for limb, kptPair in BONE_KPT_PAIRS.items():
        kptA, kptB = kptPair
        # color based on right, left, or center body part
        if uni_color: color = 'midnightblue'
        elif kptA[0]=='R' or kptB[0]=='R': color = 'olive'
        elif kptA[0]=='L' or kptB[0]=='L': color = 'purple'
        else: color = 'black'
        thickness = 3.0 if uni_color else 1.5
        kptAIdx, kptBIdx = kpt_2_idx[kptA], kpt_2_idx[kptB]
        x_pts = [xKpts[kptAIdx], xKpts[kptBIdx]]
        y_pts = [yKpts[kptAIdx], yKpts[kptBIdx]]
        z_pts = [zKpts[kptAIdx], zKpts[kptBIdx]]
        sp_axs.plot(x_pts, y_pts, z_pts, color, zdir='z', linewidth=thickness)

    # Use Dots to identify quad-kpts
    if np.all(np.array(jnt_quad_kpts)>=0):
        sp_axs.scatter3D(xKpts[jnt_quad_kpts], yKpts[jnt_quad_kpts], zKpts[jnt_quad_kpts], alpha=1.0, s=20., c='black')
        sp_axs.scatter3D(xKpts[jnt_quad_kpts], yKpts[jnt_quad_kpts], zKpts[jnt_quad_kpts], alpha=1.0, s=10.,
                         c=['tab:gray', 'tab:blue', 'tab:pink', 'tab:green'])

    if display:
        plt.show(block=True)
        plt.close(sp_tag)


def render_joint_mobility(joint_name, data_points, point_color_wgts, title, prop_tag, tags, wgt_range, dtg,
                          sp_axs, sp_fig, fig_dir, axis_lim=(-1,1), z_lim=None, alpha=1.0, view_azim=30,
                          change_view=False, plot_axis=True, activate_fig=False, save_fig=True, display=False):

    if activate_fig: plt.figure(prop_tag)
    sp_fig.suptitle('{} with Non-max Suppression (d={})'.format(title, dtg), y=1., fontweight='bold', size=10)

    sp_axs.clear()
    sp_axs.set_title(joint_name, fontweight='bold', size=10)
    z_axis_lim = axis_lim if z_lim is None else z_lim
    sp_axs.set(xlim=axis_lim, zlim=z_axis_lim, ylim=axis_lim) #, xlabel='x', ylabel='y', zlabel='z')
    sp_axs.set_yticklabels([]), sp_axs.set_xticklabels([]), sp_axs.set_zticklabels([])
    # if change_view: sp_axs.view_init(elev=None, azim=-120) # Defaults: elev=30, azim=-60
    vc = -1 if change_view else 1
    sp_axs.view_init(azim=view_azim*vc, vertical_axis='y') # Defaults: elev=30, azim=-60, roll=0, vertical_axis='z'
    sp_axs.set_box_aspect(aspect=(1,1,1))

    # Draw axis
    if plot_axis:
        sp_axs.plot([-1,0], [0,0], [0,0], 'red',    zdir='z', linestyle='-.', linewidth=0.6)  # - x-axis
        sp_axs.plot([0, 1], [0,0], [0,0], 'red',    zdir='z', linestyle='--', linewidth=1.2)  # + x-axis
        sp_axs.plot([0,0], [-1,0], [0,0], 'blue',   zdir='z', linestyle='-.', linewidth=0.6)  # - y-axis
        sp_axs.plot([0,0], [0, 1], [0,0], 'blue',   zdir='z', linestyle='--', linewidth=1.2)  # + y-axis
        sp_axs.plot([0,0], [0,0], [-1,0], 'orange', zdir='z', linestyle='-.', linewidth=0.6)  # - z-axis
        sp_axs.plot([0,0], [0,0], [0, 1], 'orange', zdir='z', linestyle='--', linewidth=1.2)  # + z-axis

    # Sort data-points so that the least weighted points are plotted first
    if point_color_wgts is not None:
        ascending_order_indices = np.argsort(point_color_wgts)
        data_points = data_points[ascending_order_indices]
        point_color_wgts = point_color_wgts[ascending_order_indices]

    # Draw endpoint position of free_limb unit vector
    if data_points.shape[1]==3:
        x, y, z = data_points[:,0], data_points[:,1], data_points[:,2]
    else: x, y, z = 0, data_points[:,0], data_points[:,1]
    sp_axs.scatter3D(x, y, z, c=point_color_wgts, vmin=wgt_range[0], vmax=wgt_range[1], cmap='viridis', alpha=alpha)

    if save_fig:
        op_tag, grp_tag, wgt_tag, dup_tag = tags
        plt.savefig(os.path.join(fig_dir, '{}_{}_{}{}{}{}.png'.format(grp_tag, prop_tag, wgt_tag, op_tag, dup_tag, dtg)))
    if display:
        plt.show(block=True)
        plt.close(prop_tag)


def get_tail_vlines(x_points, data_points, vline_ymin):
    lft_x_min_idx = np.argmax(np.where(data_points>=vline_ymin, 1, 0))
    rgt_x_min_idx = np.argmax(np.where(data_points[lft_x_min_idx:]>vline_ymin, 0, 1)) + lft_x_min_idx
    return x_points[[lft_x_min_idx, rgt_x_min_idx]]

def twin_axis_line_plots(prop_tag, x_points, meta_points, title, prefix, fig_tag, sp_fig, sp_axs,
                         fig_dir, x_range, y_ranges, activate_fig=False, display=False, twin_y=True):
    '''line plots for bone ratios range for likelihood and log-likelihood'''
    if activate_fig: plt.figure(fig_tag)
    sp_fig.suptitle(title, y=1., fontweight='bold', size=10)
    y_points, vline_ymins = meta_points
    y1_vline_min, y2_vline_min = vline_ymins
    y1_points, y2_points = y_points
    y1_range, y2_range  = y_ranges

    sp_axs.clear()
    sp_axs.set_title(prop_tag, fontweight='bold', size=9)
    sp_axs.set_xticks(np.arange(x_range[0], x_range[1]+1e-5, 0.2))

    sp_axs.set(ylim=y1_range)
    sp_axs.plot(x_points, y1_points, color="tab:red", alpha=0.9)
    x_vlines = get_tail_vlines(x_points, y1_points, y1_vline_min)
    sp_axs.vlines(x=x_vlines, ymin=y1_range[0], ymax=y1_range[1],
                  colors="tab:red", linestyle='-.', linewidth=0.6, alpha=0.9)

    x_vlines = get_tail_vlines(x_points, y2_points, y2_vline_min)
    if twin_y:
        sp_axs2 = sp_axs.twinx()
        sp_axs2.set(ylim=y2_range)
        sp_axs2.plot(x_points, y2_points, color="tab:blue", alpha=0.9)
        sp_axs2.vlines(x=x_vlines, ymin=y2_range[0], ymax=y2_range[1],
                       colors="tab:blue", linestyle='-.', linewidth=0.6, alpha=0.9)
    else:
        sp_axs.plot(x_points, y2_points, color="tab:blue", alpha=0.9)
        sp_axs.vlines(x=x_vlines, ymin=y1_range[0], ymax=y1_range[1],
                      colors="tab:blue", linestyle='-.', linewidth=0.6, alpha=0.9)

    sp_fig.tight_layout(pad=5.0)
    plt.savefig(os.path.join(fig_dir, '{}_{}_pvr.png'.format(prefix, fig_tag)))
    if display:
        plt.show(block=True)
        plt.close(fig_tag)

def line_plots(prop_tag, x_points, data_points, vline_ymin, title, prefix, fig_tag, sp_fig, sp_axs,
               fig_dir, x_range, y_range, activate_fig=False, display=False):
    '''line plots for bone ratios range for likelihood and log-likelihood'''
    if activate_fig: plt.figure(fig_tag)
    sp_fig.suptitle(title, y=1., fontweight='bold', size=10)

    sp_axs.clear()
    sp_axs.set_title(prop_tag, fontweight='bold', size=9)
    sp_axs.set_xticks(np.arange(x_range[0], x_range[1]+1e-5, 0.2))
    sp_axs.set(ylim=y_range)

    sp_axs.plot(x_points, data_points, color="tab:blue")
    x_vlines = get_tail_vlines(x_points, data_points, vline_ymin)
    sp_axs.vlines(x=x_vlines, ymin=y_range[0], ymax=y_range[1],
                  color="tab:blue", linestyle='-.', linewidth=0.6)
    sp_fig.tight_layout(pad=5.0)
    plt.savefig(os.path.join(fig_dir, '{}_{}_pvr.png'.format(prefix, fig_tag)))
    if display:
        plt.show(block=True)
        plt.close(fig_tag)


def get_1d_priors(samples, log_spread):
    mean_mu = np.mean(samples) # (1,)
    variance_sigma = np.var(samples) # (1,)

    # computing data point weights as the likelihood via the probability density function
    likelihood = probability_density_func(samples, mean_mu, variance_sigma)
    likelihood_min, likelihood_max = np.min(likelihood), np.max(likelihood)
    log_lihood = log_stretch(likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
    log_lihood_min, log_lihood_max = np.min(log_lihood), np.max(log_lihood)
    log_lihood_mean, log_lihood_std, log_lihood_var = np.mean(log_lihood), np.std(log_lihood), np.var(log_lihood)

    return mean_mu, variance_sigma, likelihood_min, likelihood_max, \
           log_lihood_min, log_lihood_max, log_lihood_mean, log_lihood_std


def sv_pdf_value_likelihood_viz(subset, proportion_file, n_ratios, blen_std, property=('bpe','Bones-Ratio'),
                                frt_tag='', op_tag='_tch', augment=True, grp_props=False, log_spread=1.,
                                dom_prop_step=0.0001, plot_property_range=True, fmt=".7f"):
    # single-variate probability-density-function value derived likelihood (BSE & BPE)
    # Computes the probability density for the single-variate 'normal' distribution of bone proportions
    # from mean (1,) and variance (1,) per bone ratios.
    print('{} for {}\n'.format(property[1], property[0]))
    properties_logged = {}
    bpe_priors_params = {'br_mean_variance':{}, 'br_logli_metadata':{}, 'summary_log':''}
    aug_tag = 'wxaug' if augment else 'nouag'
    std_tag = 'blstd{}'.format(str(blen_std).replace("0.", "."))
    grp_tag = 'grpprp' if grp_props and n_ratios!=114 else 'perprp'
    subset_dir = os.path.join('../priors', subset)
    bone_prior_property_file = proportion_file.format(aug_tag, std_tag, n_ratios, frt_tag, op_tag)
    bone_prior_params_file = 'br_prior_params_{}_{}_{}{}_{}_{:.0e}.pickle'\
                                .format(std_tag, aug_tag, n_ratios, frt_tag, grp_tag, log_spread)
    bone_prior_figure_dir = 'bp_{}{}{}_ls{:.0e}'.format(n_ratios, frt_tag, std_tag, log_spread)

    if log_spread==1.:
        meta_range, meta_subticks = [(0., 1.2), (0, 20), (0, 3.5), (-8,2)], [2, 5, 5, 5] # [(0.2,1.2),,(0,11),(0,11)], []
    else: meta_range, meta_subticks = [(0., 1.4), (0, 28), (0, 11), (-8,2)], [2, 5, 5, 5]
    prop_range = meta_range[0]
    dom_proportion = np.arange(prop_range[0], prop_range[1]+dom_prop_step, dom_prop_step)

    bpe_prior_src = pickle_load(os.path.join(subset_dir, 'properties', bone_prior_property_file))
    ordered_bone_prop_tags = bpe_prior_src['ratio_order']
    bpe_priors_params['ratio_order'] = ordered_bone_prop_tags
    bpe_priors_params['induced_blen_std'] = bpe_prior_src['induced_blen_std']
    bpe_priors_params['keypoint_indexes'] = bpe_prior_src['keypoint_indexes']
    bpe_priors_params['bone_kpt_pairs'] = bpe_prior_src['bone_kpt_pairs']
    bpe_priors_params['rgt_sym_bones'] = bpe_prior_src['rgt_sym_bones']
    bpe_priors_params['lft_sym_bones'] = bpe_prior_src['lft_sym_bones']
    proportion_np = bpe_prior_src['bone_ratios'] # (n,1,13)
    log_global_max = np.zeros((2, 6), dtype=np.float32)
    log_global_min = np.full((2, 6), dtype=np.float32, fill_value=np.inf)

    if grp_props and n_ratios!=114:
        component_tags = []
        for prop_tag in ordered_bone_prop_tags:
            bone_pair = prop_tag.split('/')
            for p_idx in range(len(bone_pair)):
                if bone_pair[p_idx].find('R')==0:
                    bone_pair[p_idx] = bone_pair[p_idx].replace("R", "{0}", 1)
                elif bone_pair[p_idx].find('L')==0:
                    bone_pair[p_idx] = bone_pair[p_idx].replace("L", "{0}", 1)
            component_tags.append('{}/{}'.format(bone_pair[0], bone_pair[1]))
        component_tags = list(set(component_tags))
    else: component_tags = ordered_bone_prop_tags

    SPS_FIG, SPS_AXS, FIG_IDS = [None]*5, [None]*5, ['pdfl','lkli','msnl','mmsn','ilmw']
    n_rows, n_cols = prop_fig_rowvcol[len(component_tags)]
    for i in range(len(FIG_IDS)):
        SPS_FIG[i], SPS_AXS[i] = \
            plt.subplots(nrows=n_rows, ncols=n_cols, num=FIG_IDS[i], figsize=(3*n_cols, 3*n_rows))
        plt.subplots_adjust(left=0.03, bottom=None, right=0.97, top=None, wspace=0.3, hspace=0.5)

    for comp_tag in component_tags:
        print('Computing and plotting {}..'.format(comp_tag))
        # combine symmetric proportions in grp-prop mode
        prefixes = ['R', 'L'] if comp_tag.find('{0}')>=0 else ['']
        symm_prop_list = []
        if len(prefixes)>1:
            for prefix in prefixes:
                prop_id = comp_tag.format(prefix)
                r_idx = get_id_index(ordered_bone_prop_tags, prop_id)
                symm_prop_list.append(proportion_np[:, 0, r_idx])
            proportion_x = np.concatenate(symm_prop_list, axis=0)
        else:
            r_idx = get_id_index(ordered_bone_prop_tags, comp_tag)
            proportion_x = proportion_np[:, 0, r_idx]
        proportion_min, proportion_max = np.min(proportion_x), np.max(proportion_x)

        # BPE prior parameters are generated from the combination of subset and domain bone proportions
        mean_mu = np.mean(proportion_x) # (1,)
        variance_sigma = np.var(proportion_x) # (1,)
        print('subset:{} - domain:{} - domain/subset:{:.2f}%'.format(proportion_x.shape,
              dom_proportion.shape, (dom_proportion.shape[0]/proportion_x.shape[0])*100))

        # computing data point weights as the likelihood via the probability density function
        likelihood = probability_density_func(proportion_x, mean_mu, variance_sigma)
        likelihood_min, likelihood_max = np.min(likelihood), np.max(likelihood)
        likelihood_mean, likelihood_var = np.mean(likelihood), np.var(likelihood)
        log_lihood = log_stretch(likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
        log_lihood_min, log_lihood_max = np.min(log_lihood), np.max(log_lihood)
        log_lihood_mean, log_lihood_std, log_lihood_var = np.mean(log_lihood), np.std(log_lihood), np.var(log_lihood)
        inv_logli_mean_wgt = 1/np.ceil(log_lihood_mean)
        ilmw_logli = log_lihood * inv_logli_mean_wgt
        ilmw_logli_min, ilmw_logli_max = np.min(ilmw_logli), np.max(ilmw_logli)
        ilmw_logli_mean, ilmw_logli_var = np.mean(ilmw_logli), np.var(ilmw_logli)
        msnl_logli = (log_lihood - log_lihood_mean) / log_lihood_std
        msnl_logli_min, msnl_logli_max = np.min(msnl_logli), np.max(msnl_logli)
        msnl_logli_mean, msnl_logli_var = np.mean(msnl_logli), np.var(msnl_logli)

        # Bone-Ratio likelihood generated from closed dom
        dom_proportion_min, dom_proportion_max = np.min(dom_proportion), np.max(dom_proportion)
        dom_likelihood = probability_density_func(dom_proportion, mean_mu, variance_sigma)
        dom_likelihood_min, dom_likelihood_max = np.min(dom_likelihood), np.max(dom_likelihood)
        dom_log_lihood = log_stretch(dom_likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
        dom_log_lihood_min, dom_log_lihood_max = np.min(dom_log_lihood), np.max(dom_log_lihood)
        dom_log_lihood_mean, dom_log_lihood_var = np.mean(dom_log_lihood), np.var(dom_log_lihood)
        maximum_likelihood = max(likelihood_max, dom_likelihood_max)
        minimum_log_lihood = min(log_lihood_min, dom_log_lihood_min)
        maximum_log_lihood = max(log_lihood_max, dom_log_lihood_max)
        if dom_likelihood_max>=likelihood_max:
            boneprop_likelihood_argmax = get_value_with_max_loglihood(dom_proportion, dom_likelihood)
        else: boneprop_likelihood_argmax = get_value_with_max_loglihood(proportion_x, likelihood)

        mmsn_logli = (log_lihood - minimum_log_lihood) / (maximum_log_lihood - minimum_log_lihood)
        mmsn_logli_min, mmsn_logli_max = np.min(mmsn_logli), np.max(mmsn_logli)
        mmsn_logli_mean, mmsn_logli_var = np.mean(mmsn_logli), np.var(mmsn_logli)
        dom_mmsn_logli = (dom_log_lihood - minimum_log_lihood) / (maximum_log_lihood - minimum_log_lihood)
        dom_mmsn_logli_min, dom_mmsn_logli_max = np.min(dom_mmsn_logli), np.max(dom_mmsn_logli)
        dom_msnl_logli = (dom_log_lihood - log_lihood_mean) / log_lihood_std
        dom_msnl_logli_min, dom_msnl_logli_max = np.min(dom_msnl_logli), np.max(dom_msnl_logli)
        dom_ilmw_logli = dom_log_lihood * inv_logli_mean_wgt
        dom_ilmw_logli_min, dom_ilmw_logli_max = np.min(dom_ilmw_logli), np.max(dom_ilmw_logli)

        properties_logged[comp_tag] = (likelihood_min, log_lihood_min, msnl_logli_min, mmsn_logli_min, ilmw_logli_min,
                                       dom_likelihood, dom_log_lihood, dom_msnl_logli, dom_mmsn_logli, dom_ilmw_logli)
        meta_max = np.asarray(
            [[proportion_max, likelihood_max, log_lihood_max, msnl_logli_max, mmsn_logli_max, ilmw_logli_max],
             [dom_proportion_max, dom_likelihood_max, dom_log_lihood_max, dom_msnl_logli_max, dom_mmsn_logli_max, dom_ilmw_logli_max]])
        log_global_max = np.where(log_global_max<meta_max, meta_max, log_global_max)
        meta_min = np.asarray(
            [[proportion_min, likelihood_min, log_lihood_min, msnl_logli_min, mmsn_logli_min, ilmw_logli_min],
             [dom_proportion_min, dom_likelihood_min, dom_log_lihood_min, dom_msnl_logli_min, dom_mmsn_logli_min, dom_ilmw_logli_min]])
        log_global_min = np.where(log_global_min>meta_min, meta_min, log_global_min)

        # note pose prior parameters (free-bone mean & covariance, and log-likelihood mean)
        for prefix in prefixes:
            prop_id = comp_tag.format(prefix)
            bpe_priors_params['br_mean_variance'][prop_id] = (mean_mu, variance_sigma)
            bpe_priors_params['br_logli_metadata'][prop_id] = \
                (maximum_likelihood, boneprop_likelihood_argmax, log_spread, log_lihood_mean,
                 log_lihood_std, minimum_log_lihood, maximum_log_lihood, inv_logli_mean_wgt)

        info_table = \
            tabulate([[property[1], proportion_min, proportion_max, proportion_max-proportion_min, mean_mu, variance_sigma],
                      ['Likelihood', likelihood_min, likelihood_max, likelihood_max-likelihood_min, likelihood_mean, likelihood_var],
                      ['Log-Lihood', log_lihood_min, log_lihood_max, log_lihood_max-log_lihood_min, log_lihood_mean, log_lihood_var],
                      ['Domain Log-Lihood', dom_log_lihood_min, dom_log_lihood_max, dom_log_lihood_max-dom_log_lihood_min, dom_log_lihood_mean, dom_log_lihood_var],
                      ['NMSL Log-Lihood', msnl_logli_min, msnl_logli_max, msnl_logli_max-msnl_logli_min, msnl_logli_mean, msnl_logli_var],
                      ['MMSN Log-Lihood', mmsn_logli_min, mmsn_logli_max, mmsn_logli_max-mmsn_logli_min, mmsn_logli_mean, mmsn_logli_var],
                      ['ILMW Log-Lihood', ilmw_logli_min, ilmw_logli_max, ilmw_logli_max-ilmw_logli_min, ilmw_logli_mean, ilmw_logli_var]],
                     headers=[comp_tag, 'Min', 'Max', 'Range', 'Mean', 'Variance'], tablefmt='psql', floatfmt=fmt)
        print(info_table)
        bpe_priors_params['summary_log'] += info_table+'\n'

    # record priors
    pickle_write(bpe_priors_params, os.path.join(subset_dir, bone_prior_params_file))

    fig_dir = os.path.join('../images', subset, bone_prior_figure_dir)
    os.makedirs(fig_dir, exist_ok=True)

    if plot_property_range:
        first_line_plot_idx = 0 #4 if plot_hists else 0
        for c_idx, comp_tag in enumerate(component_tags):
            sp_comp_tag = comp_tag.format('')
            likelihood_min, log_lihood_min, msnl_logli_min, mmsn_logli_min, ilmw_logli_min, \
            dom_likelihood, dom_log_lihood, dom_msnl_logli, dom_mmsn_logli, dom_ilmw_logli = properties_logged[comp_tag]
            norm_meta_data = [((dom_log_lihood, dom_likelihood), (log_lihood_min, likelihood_min)),
                              ((dom_log_lihood, dom_msnl_logli), (log_lihood_min, msnl_logli_min)),
                              ((dom_log_lihood, dom_mmsn_logli), (log_lihood_min, mmsn_logli_min)),
                              ((dom_log_lihood, dom_ilmw_logli), (log_lihood_min, ilmw_logli_min))]
            row_idx, col_idx = c_idx//n_cols, c_idx%n_cols

            fig_idx = first_line_plot_idx
            title = 'Likelihood of Bone-Ratio Range [{} - LogSpred:{}]'.format(subset, log_spread)
            sps_axs_i = SPS_AXS[fig_idx][row_idx,col_idx] if n_rows>1 else SPS_AXS[fig_idx][col_idx]
            line_plots(sp_comp_tag, dom_proportion, dom_likelihood, likelihood_min, title, grp_tag, FIG_IDS[fig_idx],
                       SPS_FIG[fig_idx], sps_axs_i, fig_dir, meta_range[0], meta_range[1], activate_fig=True)
            for t_idx in range(len(norm_meta_data)):
                fig_idx = first_line_plot_idx+t_idx+1
                title = 'LogLihood (Red-Left) and {}-Loglihood (Blue-Right) Values for Bone-Ratio' \
                        ' [{} - LogSpred:{}]'.format(FIG_IDS[fig_idx].upper(), subset, log_spread)
                sps_axs_i = SPS_AXS[fig_idx][row_idx,col_idx] if n_rows>1 else SPS_AXS[fig_idx][col_idx]
                y_range = meta_range[2:] if t_idx>0 else meta_range[1:3]
                twin_axis_line_plots(sp_comp_tag, dom_proportion, norm_meta_data[t_idx], title, grp_tag, FIG_IDS[fig_idx],
                                     SPS_FIG[fig_idx], sps_axs_i, fig_dir, meta_range[0], y_range, activate_fig=True,
                                     twin_y=t_idx==1, display=(c_idx==len(component_tags)-1 and t_idx==3))

    for idx, set_name in enumerate(['All-Global', 'All-Global Domain']):
        summary_table = tabulate([[property[1], log_global_min[idx,0], log_global_max[idx,0]],
                                  ['Likelihood', log_global_min[idx,1], log_global_max[idx,1]],
                                  ['Log-Lihood', log_global_min[idx,2], log_global_max[idx,2]],
                                  ['NMSL Log-Lihood', log_global_min[idx,3], log_global_max[idx,3]],
                                  ['MMSN Log-Lihood', log_global_min[idx,4], log_global_max[idx,4]],
                                  ['ILMW Log-Lihood', log_global_min[idx,5], log_global_max[idx,5]]],
                                 headers=[set_name, 'Min', 'Max'], tablefmt='psql', floatfmt=fmt)
        print('\n{}'.format(summary_table))


def pdf_likelihood_combo_viz(subset, proportion_file, n_ratios, blen_std, property=('bpe','Bones-Ratio'),
                                frt_tag='', op_tag='_tch', augment=True, grp_props=False, log_spread=1.,
                                dom_prop_step=0.0001, plot_properties=True, fmt=".7f"):
    # single-variate probability-density-function value derived likelihood (BSE & BPE)
    # Computes the probability density for the single-variate 'normal' distribution of bone proportions
    # from mean (1,) and variance (1,) per bone ratios.
    print('{} for {}\n'.format(property[1], property[0]))
    properties_logged = {}
    rpps_priors_params = {'br_mean_variance':{}, 'br_logli_metadata':{},
                         'bsym_mean_variance':{}, 'bsym_logli_metadata':{},
                         'tlen_mean_variance':{}, 'tlen_logli_metadata':{}, 'summary_log':'',
                         'ploc_covariance':{}, 'ploc_inv_covariance':{}, 'ploc_axes_means':{},
                         'pori_covariance':{}, 'pori_inv_covariance':{}, 'pori_axes_means':{}}
    aug_tag = 'wxaug' if augment else 'nouag'
    std_tag = 'blstd{}'.format(str(blen_std).replace("0.", "."))
    grp_tag = 'grpprp' if grp_props and n_ratios!=114 else 'perprp'
    subset_dir = os.path.join('../priors', subset)
    bone_prior_property_file = proportion_file.format(aug_tag, std_tag, n_ratios, frt_tag, op_tag)
    bone_prior_params_file = 'rpps_prior_params_{}_{}_{}{}_{}_{:.0e}.pickle' \
        .format(std_tag, aug_tag, n_ratios, frt_tag, grp_tag, log_spread)
    bone_prior_figure_dir = 'rpps_{}{}{}_ls{:.0e}'.format(n_ratios, frt_tag, std_tag, log_spread)
    non_br_header = ['Symm Pair', 'Likelihood-Min', 'Likelihood-Max', 'LogLikeli-Min', 'LogLikeli-Max', 'Mean', 'Variance']

    meta_range, meta_subticks = [(0., 1.2), (0, 30), (0, 3.5), (-8,2)], [2, 5, 5, 5]
    prop_range = meta_range[0]
    dom_proportion = np.arange(prop_range[0], prop_range[1]+dom_prop_step, dom_prop_step)

    bpe_prior_src = pickle_load(os.path.join(subset_dir, 'properties', bone_prior_property_file))
    ordered_bone_prop_tags = bpe_prior_src['ratio_order']
    rpps_priors_params['ratio_order'] = ordered_bone_prop_tags
    rpps_priors_params['induced_blen_std'] = bpe_prior_src['induced_blen_std']
    rpps_priors_params['keypoint_indexes'] = bpe_prior_src['keypoint_indexes']
    ordered_symm_bone_pairs = bpe_prior_src['symm_order']
    rpps_priors_params['symm_order'] = ordered_symm_bone_pairs
    rpps_priors_params['rgt_sym_bones'] = bpe_prior_src['rgt_sym_bones']
    rpps_priors_params['lft_sym_bones'] = bpe_prior_src['lft_sym_bones']
    rpps_priors_params['ctr_bones'] = bpe_prior_src['ctr_bones']

    proportion_np = bpe_prior_src['bone_ratios'] # (n,1,13)
    pose_locat = bpe_prior_src['pose_locat']
    pose_orien = bpe_prior_src['pose_orien']
    bone_symms = bpe_prior_src['bone_symms']
    torso_lens = bpe_prior_src['torso_lens']
    log_global_max = np.zeros((2,3), dtype=np.float32)
    log_global_min = np.full((2,3), dtype=np.float32, fill_value=np.inf)

    # Compute and log Torso-to-Bone ratio proportions priors
    for comp_tag in ordered_bone_prop_tags:
        print('Computing and plotting Bone-to-Torso Proportions {}..'.format(comp_tag))
        r_idx = get_id_index(ordered_bone_prop_tags, comp_tag)
        proportion_x = proportion_np[:, 0, r_idx]
        proportion_min, proportion_max = np.min(proportion_x), np.max(proportion_x)

        # BPE prior parameters are generated from the combination of subset and domain bone proportions
        mean_mu = np.mean(proportion_x) # (1,)
        variance_sigma = np.var(proportion_x) # (1,)
        print('subset:{} - domain:{} - domain/subset:{:.2f}%'.format(proportion_x.shape,
            dom_proportion.shape, (dom_proportion.shape[0]/proportion_x.shape[0])*100))

        # computing data point weights as the likelihood via the probability density function
        likelihood = probability_density_func(proportion_x, mean_mu, variance_sigma)
        likelihood_min, likelihood_max = np.min(likelihood), np.max(likelihood)
        likelihood_mean, likelihood_var = np.mean(likelihood), np.var(likelihood)
        log_lihood = log_stretch(likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
        log_lihood_min, log_lihood_max = np.min(log_lihood), np.max(log_lihood)
        log_lihood_mean, log_lihood_std, log_lihood_var = np.mean(log_lihood), np.std(log_lihood), np.var(log_lihood)
        inv_logli_mean_wgt = 1/np.ceil(log_lihood_mean)

        # Bone-Ratio likelihood generated from closed dom
        dom_proportion_min, dom_proportion_max = np.min(dom_proportion), np.max(dom_proportion)
        dom_likelihood = probability_density_func(dom_proportion, mean_mu, variance_sigma)
        dom_likelihood_min, dom_likelihood_max = np.min(dom_likelihood), np.max(dom_likelihood)
        dom_log_lihood = log_stretch(dom_likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
        dom_log_lihood_min, dom_log_lihood_max = np.min(dom_log_lihood), np.max(dom_log_lihood)
        dom_log_lihood_mean, dom_log_lihood_var = np.mean(dom_log_lihood), np.var(dom_log_lihood)
        maximum_likelihood = max(likelihood_max, dom_likelihood_max)
        minimum_log_lihood = min(log_lihood_min, dom_log_lihood_min)
        maximum_log_lihood = max(log_lihood_max, dom_log_lihood_max)
        if dom_likelihood_max>=likelihood_max:
            boneprop_likelihood_argmax = get_value_with_max_loglihood(dom_proportion, dom_likelihood)
        else: boneprop_likelihood_argmax = get_value_with_max_loglihood(proportion_x, likelihood)

        properties_logged[comp_tag] = (likelihood_min, minimum_log_lihood, dom_likelihood, dom_log_lihood)
        meta_max = np.asarray([[proportion_max, likelihood_max, log_lihood_max],
                               [dom_proportion_max, dom_likelihood_max, dom_log_lihood_max]])
        log_global_max = np.where(log_global_max<meta_max, meta_max, log_global_max)
        meta_min = np.asarray([[proportion_min, likelihood_min, log_lihood_min],
                               [dom_proportion_min, dom_likelihood_min, dom_log_lihood_min]])
        log_global_min = np.where(log_global_min>meta_min, meta_min, log_global_min)

        rpps_priors_params['br_mean_variance'][comp_tag] = (mean_mu, variance_sigma)
        rpps_priors_params['br_logli_metadata'][comp_tag] = \
            (maximum_likelihood, boneprop_likelihood_argmax, log_spread, log_lihood_mean,
             log_lihood_std, minimum_log_lihood, maximum_log_lihood, inv_logli_mean_wgt)

        info_table = \
            tabulate([[property[1], proportion_min, proportion_max, proportion_max-proportion_min, mean_mu, variance_sigma],
                      ['Likelihood', likelihood_min, likelihood_max, likelihood_max-likelihood_min, likelihood_mean, likelihood_var],
                      ['Log-Lihood', log_lihood_min, log_lihood_max, log_lihood_max-log_lihood_min, log_lihood_mean, log_lihood_var],
                      ['Domain Log-Lihood', dom_log_lihood_min, dom_log_lihood_max, dom_log_lihood_max-dom_log_lihood_min, dom_log_lihood_mean, dom_log_lihood_var]],
                     headers=[comp_tag, 'Min', 'Max', 'Range', 'Mean', 'Variance'], tablefmt='psql', floatfmt=fmt)
        print(info_table)
        rpps_priors_params['summary_log'] += info_table+'\n'

    # Compute and log BSC priors
    table_list = []
    for s_idx, comp_tag in enumerate(ordered_symm_bone_pairs):
        norm_sym_diffs = bone_symms[s_idx]
        mean_mu, variance_sigma, likelihood_min, likelihood_max, log_lihood_min, log_lihood_max, \
        log_lihood_mean, log_lihood_std = get_1d_priors(norm_sym_diffs, log_spread)
        rpps_priors_params['bsym_mean_variance'][comp_tag] = (mean_mu, variance_sigma)
        rpps_priors_params['bsym_logli_metadata'][comp_tag] = \
            (likelihood_max, log_spread, log_lihood_mean, log_lihood_std, log_lihood_min, log_lihood_max)
        table_list.append([comp_tag, likelihood_min, likelihood_max, log_lihood_min, log_lihood_max, mean_mu, variance_sigma])
    print(tabulate(table_list, headers=non_br_header, tablefmt='psql', floatfmt=fmt))

    # Compute and log Torso length
    mean_mu, variance_sigma, likelihood_min, likelihood_max, log_lihood_min, log_lihood_max, \
    log_lihood_mean, log_lihood_std = get_1d_priors(torso_lens, log_spread)
    rpps_priors_params['tlen_mean_variance'] = (mean_mu, variance_sigma)
    rpps_priors_params['tlen_logli_metadata'] = \
        (likelihood_max, log_spread, log_lihood_mean, log_lihood_std, log_lihood_min, log_lihood_max)
    print(tabulate([['Torso Length', likelihood_min, likelihood_max, log_lihood_min, log_lihood_max, mean_mu, variance_sigma]],
                   headers=['']+non_br_header[1:], tablefmt='psql', floatfmt=fmt))

    data_to_plot = []
    likelihood_scl = np.around(np.pi**-(2*1), 1) # ~=0.1

    # Compute and log pose orientation
    table_list.clear()
    print("Computing pose orientation..")
    mean_per_axis, covariance_mtx, covariance_inv, msg = \
        parameterize_jmc_priors('Pose-Orientation', pose_orien[:,0,:], 'noctr', assume_ctr=True, atol=1e-05)
    assert (is_positive_definite(covariance_mtx)), 'PO covariance is not a positive definite matrix\n {}'.format(covariance_mtx)
    rpps_priors_params['pori_axes_means'] = mean_per_axis
    rpps_priors_params['pori_covariance'] = covariance_mtx
    rpps_priors_params['pori_inv_covariance'] = covariance_inv
    # computing color-scale weights for plots
    nms_pts_coords, nms_pts_wgts = non_max_supress_points(pose_orien[:,0,:], d=d_nms)
    nms_likelihood = multivariate_probability_density_func(nms_pts_coords, mean_per_axis, covariance_mtx, covariance_inv)
    nms_log_lihood = likelihood_scl * log_stretch(nms_likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
    data_to_plot.append(['Pose Orientation', nms_pts_coords, nms_pts_wgts, nms_likelihood, nms_log_lihood])
    table_list.append(['Pose Orientation', min(nms_likelihood), max(nms_likelihood),
                       min(nms_log_lihood), max(nms_log_lihood), mean_per_axis, covariance_mtx])

    # Compute and log pose position
    print("Computing pose location..")
    # print(np.min(pose_locat[:,0,0]), np.min(pose_locat[:,0,1]), np.min(pose_locat[:,0,2]))
    # print(np.max(pose_locat[:,0,0]), np.max(pose_locat[:,0,1]), np.max(pose_locat[:,0,2]))
    mean_per_axis, covariance_mtx, covariance_inv, msg = \
        parameterize_jmc_priors('Pose-Location', pose_locat[:,0,:], 'noctr', assume_ctr=True, atol=1e-05)
    assert (is_positive_definite(covariance_mtx)), 'PL covariance is not a positive definite matrix\n {}'.format(covariance_mtx)
    rpps_priors_params['ploc_axes_means'] = mean_per_axis
    rpps_priors_params['ploc_covariance'] = covariance_mtx
    rpps_priors_params['ploc_inv_covariance'] = covariance_inv
    # computing color-scale weights for plots
    nms_pts_coords, nms_pts_wgts = non_max_supress_points(pose_locat[:,0,:], d=d_nms)
    nms_likelihood = multivariate_probability_density_func(nms_pts_coords, mean_per_axis, covariance_mtx, covariance_inv)
    nms_log_lihood = likelihood_scl * log_stretch(nms_likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
    data_to_plot.append(['Pose\nLocation', nms_pts_coords, nms_pts_wgts, nms_likelihood, nms_log_lihood])
    table_list.append(['Pose\nLocation', min(nms_likelihood), max(nms_likelihood),
                       min(nms_log_lihood), max(nms_log_lihood), mean_per_axis, covariance_mtx])
    print(tabulate(table_list, headers=non_br_header, tablefmt='psql', floatfmt=fmt))

    # record priors
    pickle_write(rpps_priors_params, os.path.join(subset_dir, bone_prior_params_file))

    fig_dir = os.path.join('../images', subset, bone_prior_figure_dir)
    os.makedirs(fig_dir, exist_ok=True)

    if plot_properties:
        # Plot pose orientation and location
        SPS_FIG, SPS_AXS = plt.subplots(nrows=2, ncols=3, subplot_kw={'projection':'3d'}, num='3DPPlots', figsize=(9, 6),)
        col_tags = ['Frequency', 'Likelihood', 'Log-Likelihood']
        title = 'Pose Orientation and Location'
        for idx in range(len(data_to_plot)):
            axes = (-1,1) if idx==0 else (-2, 2)
            z_axis = (-1,1) if idx==0 else (0, 8)
            plot_axis = True if idx==0 else False
            for wgt_idx in range(2, 5):
                col_idx = wgt_idx - 2
                sb_tag = data_to_plot[idx][0]+' '+col_tags[col_idx]
                render_joint_mobility(sb_tag, data_to_plot[idx][1], data_to_plot[idx][wgt_idx], title, '3DPPlots',
                        None, (None, None), nms_tag, SPS_AXS[idx, col_idx], SPS_FIG, fig_dir, plot_axis=plot_axis,
                        axis_lim=axes, z_lim=z_axis, change_view=False, alpha=0.5, activate_fig=True, save_fig=False)

        # Plot bone proportion range
        SPS_FIG, SPS_AXS, FIG_IDS = [None]*2, [None]*2, ['pdfl','lkli']
        n_rows, n_cols = prop_fig_rowvcol[len(ordered_bone_prop_tags)]
        for i in range(len(FIG_IDS)):
            SPS_FIG[i], SPS_AXS[i] = plt.subplots(nrows=n_rows, ncols=n_cols, num=FIG_IDS[i], figsize=(3*n_cols, 3*n_rows))
            plt.subplots_adjust(left=0.03, bottom=None, right=0.97, top=None, wspace=0.3, hspace=0.5)

        first_line_plot_idx = 0 #4 if plot_hists else 0
        for c_idx, comp_tag in enumerate(ordered_bone_prop_tags):
            sp_comp_tag = comp_tag.format('')
            likelihood_min, minimum_log_lihood, dom_likelihood, dom_log_lihood = properties_logged[comp_tag]
            norm_meta_data = ((dom_log_lihood, dom_likelihood), (minimum_log_lihood, likelihood_min))
            row_idx, col_idx = c_idx//n_cols, c_idx%n_cols

            fig_idx = first_line_plot_idx
            title = 'Likelihood of Bone-Ratio Range [{} - LogSpred:{}]'.format(subset, log_spread)
            sps_axs_i = SPS_AXS[fig_idx][row_idx,col_idx] if n_rows>1 else SPS_AXS[fig_idx][col_idx]
            line_plots(sp_comp_tag, dom_proportion, dom_likelihood, likelihood_min, title, grp_tag, FIG_IDS[fig_idx],
                       SPS_FIG[fig_idx], sps_axs_i, fig_dir, meta_range[0], meta_range[1], activate_fig=True)
            fig_idx = first_line_plot_idx+1
            title = 'LogLihood (Red-Left) and {}-Loglihood (Blue-Right) Values for Bone-Ratio' \
                    ' [{} - LogSpred:{}]'.format(FIG_IDS[fig_idx].upper(), subset, log_spread)
            sps_axs_i = SPS_AXS[fig_idx][row_idx,col_idx] if n_rows>1 else SPS_AXS[fig_idx][col_idx]
            y_range = meta_range[1:3]
            twin_axis_line_plots(sp_comp_tag, dom_proportion, norm_meta_data, title, grp_tag, FIG_IDS[fig_idx],
                                 SPS_FIG[fig_idx], sps_axs_i, fig_dir, meta_range[0], y_range, activate_fig=True,
                                 twin_y=False, display=(c_idx==len(ordered_bone_prop_tags)-1))

    for idx, set_name in enumerate(['All-Global', 'All-Global Domain']):
        summary_table = tabulate([[property[1], log_global_min[idx,0], log_global_max[idx,0]],
                                  ['Likelihood', log_global_min[idx,1], log_global_max[idx,1]],
                                  ['Log-Lihood', log_global_min[idx,2], log_global_max[idx,2]]],
                                 headers=[set_name, 'Min', 'Max'], tablefmt='psql', floatfmt=fmt)
        print('\n{}'.format(summary_table))


def mv_pdf_fbvec_likelihood_viz(subset='S1.S5.S6.S7.S8', type='fbj_orients', n_fbs=12, op_tag='_tch', frt_tag='',
                cov_type='noctr', assume_ctr=True, augment=True, grp_jnts=True, rbo_type='rmtx', quintuple=False,
                duplicate=0, log_spread=1., cross_plot_wgt=True, dom_angle_step=1, fmt=".7f"):
    # multi-variate probability-density-function unit-vector derived likelihood (JME)
    # Computes the probability density for the multivariate 'normal' distribution of free-limb unit-vector
    # from mean-per-axis-component (3,1) and covariance matrix (3,3) per joint of non-maximum suppressed subset.
    # Then plots joints' free-limb unit-vector mobility spheres using probability from PDF as weight
    properties_logged = {}
    jmc_priors_params = {'jnt_rank_sets':[{},{}], 'fb_covariance':[{},{}], 'fb_inv_covariance':[{},{}],
                         'fb_axes_means':[{},{}], 'fb_logli_metadata':[{},{}], 'summary_log':''}  #[[None]*12,[None]*11]
    # nms_tag = str(d_nms).replace("0.", ".")
    fbq_tag = 5 if quintuple else 4
    aug_tag = 'wxaug' if augment else 'nouag'
    grp_tag = 'grpjnt' if grp_jnts else 'perjnt'
    dup_tag = '' if duplicate==0 else '+{}'.format(duplicate)
    cvc_tag = '{}{}cv'.format(cov_type, int(assume_ctr)) # mcd-cv vs. mle-cv # 0:assume_centered=False 1:assume_centered=True
    cvc_tag = cov_type if cov_type=='noctr' else cvc_tag # 'noctr' with assume_ctr=True
    subset_dir = os.path.join('../priors', subset)
    fbjnt_prior_property_file = '{}{}_{}{}_{}_{}_{}{}.pickle'.\
        format(type, frt_tag, rbo_type, fbq_tag, aug_tag, n_fbs, grp_tag, op_tag)
    type_prefix = 'rpbo' if type=='RP-BoneOrient' else 'fb'
    fbjnt_prior_params_file = '{}_prior_params_{}{}_{}{}_{}_{}_{}{}_{:.0e}.pickle'.\
        format(type_prefix, frt_tag, cvc_tag, rbo_type, fbq_tag, aug_tag, n_fbs, grp_tag, dup_tag, log_spread)
    fbjnt_prior_figure_dir = '{}_{}_nms{}{}_{}_ls{:.0e}'.format(type_prefix, cvc_tag, nms_tag, frt_tag, rbo_type, log_spread)
    n_rows, n_cols = 2, 5 if grp_jnts else 8 # 4 OR 6
    n_fg = 3 #6 # number of fig groups
    SPS_FIG, SPS_AXS = [None]*2*n_fg, [None]*2*n_fg
    FIG_IDS = ['nms','dnms','pdf','dpdf','log','dlog']#,'msnl','dmsnl','mmsn','dmmsn','ilmw','dilmw']
    for i in range(len(FIG_IDS)):
        if i==1: continue # skip dnms
        SPS_FIG[i], SPS_AXS[i] = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 4*n_rows),
                                              subplot_kw={'projection':'3d'}, num=FIG_IDS[i])
        SPS_FIG[i].subplots_adjust(left=0.0, right=1.0, wspace=-0.0)
    title_tags = ['Non-Max Suppression','Domain Orientation','Likelihood','Domain Likelihood','Log-Lihood',
                  'Domain Log-Lihood','Norm. Mean-&-Std Log-Lihood','Norm. Mean-&-Std Domain Log-Lihood',
                  'Norm. Min-&-Max Log-Lihood','Norm. Min-&-Max Domain Log-Lihood',
                  'Inv-Log-Mean Wgtd. Log-Lihood','Inv-Log-Mean Wgtd. Domain Log-Lihood']
    log_global_max = np.zeros((2, n_fg), dtype=np.float32) # max: nms,lli,log,msnl,mmsn,ilmw
    log_global_min = np.full((2, n_fg), dtype=np.float32, fill_value=np.inf) # min: nms,lli,log,msnl,mmsn,ilmw
    if cross_plot_wgt:
        wgt_suffix, wgt_tag = ' - Cross Plot Scaled', 'cps'
    else: wgt_suffix, wgt_tag = '', 'uns' # 'uns'->unscaled
    spherical_pts_coords, spherical_pts_ftrvec = points_on_spherical_surface(step=dom_angle_step) # 1<=dom_angle_step<=2.5

    jmc_prior_src = pickle_load(os.path.join(subset_dir, 'properties', fbjnt_prior_property_file))
    fbjnt_ordered_names = jmc_prior_src['joint_order']
    jmc_priors_params['joint_order'] = fbjnt_ordered_names
    jmc_priors_params['joint_align_config'] = jmc_prior_src['joint_align_config']
    jmc_priors_params['keypoint_indexes'] = jmc_prior_src['keypoint_indexes']
    jmc_priors_params['q_kpt_set'] = jmc_prior_src['q_kpt_set']
    jmc_priors_params['group'] = jmc_prior_src['group']

    jmc_priors_params['fb_logli_metadata'][0]['rank_const'] = (1, 3) # (rank, k)
    for fbjnt_tag in _rank1_id_tags:
        print('\n-------------------------\nLoading and parameterizing {} priors..'.format(fbjnt_tag))
        prefixes = ['R', 'L'] if grp_jnts and fbjnt_tag in SYM_JOINT_GROUP else ['']
        # merge free-bone unit vectors of symmetric joints if in grp-jnt mode
        symm_fb_uvecs_list = []
        if len(prefixes)>1:
            for prefix in prefixes:
                symm_fb_uvecs_list.append(jmc_prior_src[prefix+fbjnt_tag])
            free_bone_orient = np.concatenate(symm_fb_uvecs_list, axis=0) # was free_bone_uvecs
        else:
            free_bone_orient = jmc_prior_src[fbjnt_tag] # (?, 3:[x, y, z]) cartesian coordinate system
            # if duplicate>0 then free_bone_orient is repeated 2^(duplicate)
            for dup_idx in range(duplicate):
                print('\tduplicating 2^{} times..'.format(dup_idx+1))
                free_bone_orient = np.concatenate([free_bone_orient, free_bone_orient], axis=0)

        # Compute mean and covariance matrix of joint's free-limb unit vector components' distribution
        dom_pts_coords, dom_pts_ftrvec = spherical_pts_coords, spherical_pts_ftrvec
        print('subset:{} - domain:{} - domain/subset:{:.1f}%'.format(free_bone_orient.shape,
            dom_pts_coords.shape, (dom_pts_coords.shape[0]/free_bone_orient.shape[0])*100))

        mean_per_axis, covariance_mtx, covariance_inv, msg = \
            parameterize_jmc_priors(fbjnt_tag, free_bone_orient, cov_type, assume_ctr, atol=1e-05, rs_seed=n_fbs)
        assert (is_positive_definite(covariance_mtx)), 'not positive definite matrix\n {}'.format(covariance_mtx)
        if msg!='': jmc_priors_params['summary_log'] += '\n'+msg+'\n'

        likelihood = multivariate_probability_density_func(free_bone_orient, mean_per_axis, covariance_mtx, covariance_inv)
        likelihood_min, likelihood_max = np.min(likelihood), np.max(likelihood)
        likelihood_scl = np.around(np.pi**-(2*1), 1) # ~=0.1
        log_lihood = likelihood_scl * log_stretch(likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
        log_lihood_min, log_lihood_max = np.min(log_lihood), np.max(log_lihood)
        log_lihood_mean, log_lihood_std = np.mean(log_lihood), np.std(log_lihood)
        inv_logli_mean_wgt = 1/np.ceil(log_lihood_mean)

        # computing data point weights as the likelihood of each free-limb orientation via the probability density function
        nms_pts_coords, nms_pts_wgts = non_max_supress_points(free_bone_orient, d=d_nms)
        nms_likelihood = multivariate_probability_density_func(nms_pts_coords, mean_per_axis, covariance_mtx, covariance_inv)
        nms_log_lihood = likelihood_scl * log_stretch(nms_likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1

        # Free-bone likelihood generated from closed spherical dom
        dom_likelihood = multivariate_probability_density_func(dom_pts_ftrvec, mean_per_axis, covariance_mtx, covariance_inv)
        dom_likelihood_min, dom_likelihood_max = np.min(dom_likelihood), np.max(dom_likelihood)
        dom_log_lihood = likelihood_scl * log_stretch(dom_likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
        dom_log_lihood_min, dom_log_lihood_max = np.min(dom_log_lihood), np.max(dom_log_lihood)
        maximum_likelihood = max(likelihood_max, dom_likelihood_max)
        minimum_log_lihood = min(log_lihood_min, dom_log_lihood_min)
        maximum_log_lihood = max(log_lihood_max, dom_log_lihood_max)
        if dom_likelihood_max>=likelihood_max:
            freebone_likelihood_argmax = get_value_with_max_loglihood(dom_pts_ftrvec, dom_likelihood)
        else: freebone_likelihood_argmax = get_value_with_max_loglihood(free_bone_orient, likelihood)
        properties_logged[fbjnt_tag] = \
            (nms_pts_coords, nms_pts_wgts, nms_likelihood, dom_likelihood, nms_log_lihood, dom_log_lihood)

        # note pose prior parameters (free-bone mean & covariance, and log-likelihood mean)
        for prefix in prefixes:
            orient_id = prefix+fbjnt_tag
            jmc_priors_params['jnt_rank_sets'][0][orient_id] = [orient_id]
            jmc_priors_params['fb_axes_means'][0][orient_id] = mean_per_axis
            jmc_priors_params['fb_covariance'][0][orient_id] = covariance_mtx
            jmc_priors_params['fb_inv_covariance'][0][orient_id] = covariance_inv
            jmc_priors_params['fb_logli_metadata'][0][orient_id] = \
                (maximum_likelihood, freebone_likelihood_argmax, log_spread, log_lihood_mean,
                 log_lihood_std, minimum_log_lihood, maximum_log_lihood, inv_logli_mean_wgt)

        meta_max = np.asarray([[np.max(nms_pts_wgts), np.max(nms_likelihood), np.max(nms_log_lihood)],
                               [0, np.max(dom_likelihood), np.max(dom_log_lihood)]])
        log_global_max = np.where(log_global_max<meta_max, meta_max, log_global_max)
        meta_min = np.asarray([[np.min(nms_pts_wgts), np.min(nms_likelihood), np.min(nms_log_lihood)],
                               [np.inf, np.min(dom_likelihood), np.min(dom_log_lihood)]])
        log_global_min = np.where(log_global_min>meta_min, meta_min, log_global_min)

        info_table = \
            tabulate([['Free-Bone', 'N/A', 'N/A', np.around(mean_per_axis, 7), np.around(covariance_mtx, 7)],
                      ['Likelihood', format(likelihood_min, fmt), format(likelihood_max, fmt),
                       format(np.mean(likelihood), fmt), format(np.var(likelihood), fmt)],
                      ['Domain Likelihood', format(dom_likelihood_min, fmt), format(dom_likelihood_max, fmt),
                       format(np.mean(dom_likelihood), fmt), format(np.var(dom_likelihood), fmt)],
                      ['Log-Lihood', format(log_lihood_min, fmt), format(log_lihood_max, fmt),
                       format(log_lihood_mean, fmt), format(np.var(log_lihood), fmt)],
                      ['Domain Log-Lihood', format(dom_log_lihood_min, fmt), format(dom_log_lihood_max, fmt),
                       format(np.mean(dom_log_lihood), fmt), format(np.var(dom_log_lihood), fmt)]],
                     headers=[fbjnt_tag, 'Min', 'Max', 'Mean', 'Co-Variance'], tablefmt='psql', floatfmt=fmt)
        print(info_table)
        jmc_priors_params['summary_log'] += info_table+'\n'

    jmc_priors_params['fb_logli_metadata'][1]['rank_const'] = (2, 6) # (rank, k)
    for combo_joints_tag in _rank2_id_tags:
        print('---------------------------\nParameterizing combo joints: {}..'.format(combo_joints_tag))
        joints_tags = combo_joints_tag.split('-&-')
        combo_fb_uvecs_list = []

        contains_sym_jnt = False
        for fbjnt_tag in joints_tags:
            if grp_jnts and fbjnt_tag in SYM_JOINT_GROUP:
                contains_sym_jnt = True
                break

        fbjnt_pairs = []
        for fbjnt_tag in joints_tags:
            if contains_sym_jnt:
                combo_prefixes = ['R', 'L'] if grp_jnts and fbjnt_tag in SYM_JOINT_GROUP else ['', '']
                symm_fb_uvecs_list = []
                symm_dup_jnts = []
                for prefix in combo_prefixes:
                    symm_fb_uvecs_list.append(jmc_prior_src[prefix+fbjnt_tag])
                    symm_dup_jnts.append(prefix+fbjnt_tag)
                symm_fb_uvecs = np.concatenate(symm_fb_uvecs_list, axis=0)
                combo_fb_uvecs_list.append(symm_fb_uvecs) # (?, 3:[x, y, z]) cartesian coordinate system
                fbjnt_pairs.append(symm_dup_jnts)
            else:
                combo_fb_uvecs_list.append(jmc_prior_src[fbjnt_tag]) # (?, 3:[x, y, z]) cartesian coordinate system
                fbjnt_pairs.append([fbjnt_tag])
        assert (len(fbjnt_pairs[0])==len(fbjnt_pairs[1])), 'joint_pairs:{}'.format(fbjnt_pairs)

        combo_fbs_orient = np.concatenate(combo_fb_uvecs_list, axis=-1)

        # Extract JME Pose Priors parameters
        mean_per_axis, covariance_mtx, covariance_inv, msg = \
            parameterize_jmc_priors(combo_joints_tag, combo_fbs_orient, cov_type, assume_ctr, rs_seed=n_fbs)
        assert (is_positive_definite(covariance_mtx)), 'not positive definite matrix\n {}'.format(covariance_mtx)
        if msg!='': jmc_priors_params['summary_log'] += '\n'+msg+'\n'

        likelihood = multivariate_probability_density_func(combo_fbs_orient, mean_per_axis, covariance_mtx, covariance_inv)
        likelihood_min, likelihood_max = np.min(likelihood), np.max(likelihood)
        likelihood_scl = np.around(np.pi**-(2*2), 2) # ~= 0.01
        log_lihood = likelihood_scl * log_stretch(likelihood, spread=log_spread)  # 1e-5->1e-3->1e-1
        log_lihood_min, log_lihood_max = np.min(log_lihood), np.max(log_lihood)
        log_lihood_mean, log_lihood_std = np.mean(log_lihood), np.std(log_lihood)
        inv_logli_mean_wgt = 1/np.ceil(log_lihood_mean)

        freebone_likelihood_argmax = get_value_with_max_loglihood(combo_fbs_orient, likelihood)

        for sd_idx in range(len(fbjnt_pairs[0])):
            jnt_1 = fbjnt_pairs[0][sd_idx]
            jnt_2 = fbjnt_pairs[1][sd_idx]
            orient_combo_id = '{}-&-{}'.format(jnt_1, jnt_2)
            jmc_priors_params['jnt_rank_sets'][1][orient_combo_id] = [jnt_1, jnt_2]
            jmc_priors_params['fb_axes_means'][1][orient_combo_id] = mean_per_axis
            jmc_priors_params['fb_covariance'][1][orient_combo_id] = covariance_mtx
            jmc_priors_params['fb_inv_covariance'][1][orient_combo_id] = covariance_inv
            jmc_priors_params['fb_logli_metadata'][1][orient_combo_id] = \
                (likelihood_max, freebone_likelihood_argmax, log_spread, log_lihood_mean,
                 log_lihood_std, log_lihood_min, log_lihood_max, inv_logli_mean_wgt)

        info_table = \
            tabulate([['Free-Bone', 'N/A', 'N/A', np.around(mean_per_axis, 7), np.around(covariance_mtx, 7)],
                      ['Likelihood', format(likelihood_min, fmt), format(likelihood_max, fmt),
                       format(np.mean(likelihood), fmt), format(np.var(likelihood), fmt)],
                      ['Log-Lihood', format(log_lihood_min, fmt), format(log_lihood_max, fmt),
                       format(log_lihood_mean, fmt), format(np.var(log_lihood), fmt)]],
                     headers=[combo_joints_tag, 'Min', 'Max', 'Mean', 'Co-Variance'], tablefmt='psql', floatfmt=fmt)
        print(info_table)
        jmc_priors_params['summary_log'] += info_table+'\n'

    # record priors
    pickle_write(jmc_priors_params, os.path.join(subset_dir, fbjnt_prior_params_file))

    plt_tags = ('', grp_tag, wgt_tag, dup_tag)
    fig_dir = os.path.join('../images', subset, fbjnt_prior_figure_dir)
    os.makedirs(fig_dir, exist_ok=True)
    info_table_list = []

    # Plot free-bone data-points generated from dataset/subset
    for j_idx, fbjnt_tag in enumerate(_rank1_id_tags):
        nms_pts_coords, nms_pts_wgts, likelihood, dom_likelihood, log_lihood, dom_log_lihood = properties_logged[fbjnt_tag]
        dom_pts_coords = spherical_pts_coords

        plot_color_weights = [nms_pts_wgts, None, likelihood, dom_likelihood, log_lihood, dom_log_lihood]

        for sub_idx in range(2): # (train-subset & domain-subset)
            plot_points = nms_pts_coords if sub_idx==0 else dom_pts_coords
            row_idx, col_idx = j_idx//n_cols, j_idx%n_cols
            alpha = 0.5 if sub_idx==0 else 1.0
            flip_view = not grp_jnts and fbjnt_tag in ['LThigh','LBicep','LShoulder','LElbow','LHip']
            for fig_grp_idx in range(n_fg):
                fig_idx = (fig_grp_idx*2)+sub_idx
                point_weights = plot_color_weights[fig_idx]
                if point_weights is None: continue
                s_idx = 0 if fig_grp_idx==0 else 1 # use data_point free-limb-uvecs color range to plot free-limb-uvecs
                wgt_range = (log_global_min[s_idx, fig_grp_idx],
                             log_global_max[s_idx, fig_grp_idx]) if cross_plot_wgt else (None,None)
                title = 'Joint Mobility ({}{}) Priors [{} Log-Spread:{}]'.\
                        format(title_tags[fig_idx], wgt_suffix, subset, log_spread)
                render_joint_mobility(fbjnt_tag, plot_points, point_weights, title, FIG_IDS[fig_idx], plt_tags,
                                      wgt_range, nms_tag, SPS_AXS[fig_idx][row_idx,col_idx], SPS_FIG[fig_idx],
                                      fig_dir, change_view=flip_view, alpha=alpha, activate_fig=True, save_fig=True,
                                      display=(j_idx==len(_rank1_id_tags)-1 and sub_idx==1 and fig_grp_idx==n_fg-1))
                if j_idx==0: # execute once
                    info_table_list.append([title_tags[fig_idx],
                                            log_global_min[sub_idx,fig_grp_idx], log_global_max[sub_idx,fig_grp_idx]])

    info_table = tabulate(info_table_list, headers=['Properties', 'Min', 'Max'], tablefmt='psql', floatfmt=fmt)
    print('\n{}'.format(info_table))

    # free-bone endpoints' log-likelihood animation
    # --------------------------------------------------------------------------------------------------------------
    print('\nAnimating 3D scatter plots...')
    figsize_wxh = (15, 8) if grp_jnts else (20, 8)
    anim_fig, anim_axs = plt.subplots(n_rows, n_cols, figsize=figsize_wxh, subplot_kw={'projection':'3d'}, num='anim')
    anim_fig.subplots_adjust(left=0.0, right=1.0, wspace=-0.0)

    def init():
        for j_idx, fbjnt_tag in enumerate(_rank1_id_tags):
            nms_pts_coords, __, likelihood, __, log_lihood, __ = properties_logged[fbjnt_tag]
            row_idx, col_idx = j_idx//n_cols, j_idx%n_cols
            flip_view = not grp_jnts and fbjnt_tag in ['LThigh','LBicep','LShoulder','LElbow','LHip']
            title = 'Free-Bones Log-likelihood Animation Joint Mobility Priors [{} Log-Spread:{}]'.format(subset, log_spread)
            render_joint_mobility(fbjnt_tag, nms_pts_coords, log_lihood, title, 'anim', plt_tags,
                                  (None,None), nms_tag, anim_axs[row_idx,col_idx], anim_fig, fig_dir,
                                  change_view=flip_view, alpha=0.5, activate_fig=True, save_fig=True, display=False)
        return anim_fig,

    def animate(i):
        view_azim = (i + 30) % 360
        for j_idx, fbjnt_tag in enumerate(_rank1_id_tags):
            row_idx, col_idx = j_idx//n_cols, j_idx%n_cols
            flip_view = not grp_jnts and fbjnt_tag in ['LThigh','LBicep','LShoulder','LElbow','LHip']
            vc = -1 if flip_view else 1
            anim_axs[row_idx,col_idx].view_init(azim=view_azim*vc, vertical_axis='y')
        return anim_fig,

    anim = animation.FuncAnimation(anim_fig, animate, init_func=init, frames=360, interval=20, blit=True)
    anim_path = os.path.join(fig_dir, '{}_{}_{}{}{}.gif'.format(grp_tag, 'logli', wgt_tag, op_tag, nms_tag))
    anim.save(anim_path, writer='ffmpeg', fps=30)
    print('saved to {}'.format(anim_path))
    # --------------------------------------------------------------------------------------------------------------


def parameterize_jmc_priors(component_tag, free_bone_orient, cov_type, assume_ctr, atol=5e-03, rs_seed=None):
    msg = ''
    n = free_bone_orient.shape[1]
    if cov_type in ['mcd', 'mle']:
        if cov_type=='mcd':
            cov = MinCovDet(random_state=rs_seed, assume_centered=assume_ctr).fit(free_bone_orient) # best if assume_centered=False
        elif cov_type=='mle':
            cov = EmpiricalCovariance(assume_centered=assume_ctr).fit(free_bone_orient) # best if assume_centered=True and np.mean()

        if assume_ctr:
            x_data = free_bone_orient[cov.support_, :]
            all_data_mean = np.mean(free_bone_orient, axis=0, keepdims=True).T # (3,) -> (3,1)
            mean_per_axis = np.mean(x_data, axis=0, keepdims=True).T # (3,) -> (3,1)
            msg = 'using {} of {} data points ({:.1f}%), mean difference:{}'. \
                format(x_data.shape[0], free_bone_orient.shape[0],
                       (x_data.shape[0]/free_bone_orient.shape[0])*100, (all_data_mean-mean_per_axis)[:,0])
            print(msg)
        else: mean_per_axis = np.expand_dims(cov.location_, axis=-1) # (3,) -> (3,1) todo: mean from data to avoid nan??

        covariance_mtx = cov.covariance_
        covariance_inv = cov.precision_
        if component_tag.find('Hip')>=0:
            for i in range(n):
                covariance_mtx[-1,i], covariance_inv[-1,i] = 0., 0.
                covariance_mtx[i,-1], covariance_inv[i,-1] = 0., 0.
            covariance_mtx[-1,-1], covariance_inv[-1,-1] = 1., 1.

        print('\nraw-mean:{} vs. est-mean:{}\n'.format(cov.raw_location_, cov.location_))
        print('raw-covar:\n{}\nest-covar:\n{}\n'.format(cov.raw_covariance_, cov.covariance_))
        print('cov-precision:\n{}\ninv-covariance:\n{}\n'.format(cov.precision_, np.linalg.inv(cov.covariance_)))
        print('inverse matrix test: cov-precision:{} inv-covariance:{}'.format(
            np.all(np.isclose(cov.covariance_.dot(np.linalg.inv(cov.covariance_)), np.eye(n), atol=atol)),
            np.all(np.isclose(covariance_mtx.dot(covariance_inv), np.eye(n), atol=atol))))
    else:
        mean_per_axis = np.mean(free_bone_orient, axis=0, keepdims=True).T # (3,) -> (3,1)
        #mean_per_axis = np.where(np.isclose(mean_per_axis, 0, atol=1e-07), 0., mean_per_axis)
        #covariance_mtx = empirical_covariance(free_bone_orient, assume_centered=True).astype(np.float32) # (3,3) or (2,2)
        covariance_mtx = empirical_covariance(free_bone_orient, assume_centered=assume_ctr) # OPT* >> assume_ctr=True
        #covariance_mtx = np.where(np.isclose(covariance_mtx, 0, atol=1e-07), 0., covariance_mtx)
        for i in range(n):
            if np.isclose(covariance_mtx[i,i], 0., atol=1e-07):
                msg += '\n\t[ALERT!] setting {} null dimension-{} at {}'.format(component_tag, i, covariance_mtx[i,i])
                print(msg)
                for j in range(n):
                    if i==j: continue
                    covariance_mtx[i,j], covariance_mtx[j,i] = 0., 0.
                covariance_mtx[i,i] = 1.
        covariance_inv = np.linalg.inv(covariance_mtx)
        cov_dot_inv = covariance_mtx.dot(covariance_inv)
        all_are_identity = np.all(np.isclose(cov_dot_inv, np.eye(n), atol=atol)) # atol=1e-05/1e-02
        print('inverse covariance matrix test: {}'.format(all_are_identity))
        assert(all_are_identity), '{} inverse matrix-{} test: inv-covariance:\n{}'.format(component_tag, n, cov_dot_inv)

    return mean_per_axis, covariance_mtx, covariance_inv, msg


def self_sup_losses_and_weights():
    warmup_end = {'S1.001':128,'S1.01':25,'S1.05':5,'S1.1':5,'S1.5':4,'S1':4,'S1.S5':3,'S1.S5.S6':3,'S1.S5.S6.S7.S8':2}
    loss_ids = ['Traj-trn','3D-trn','3D-trn-eval','3D-valid','2D-trn','MBLE-trn','BSE-trn','BPE-trn','JME-trn','PTO-trn']
    all_loss_idxs = [0,1,2,3,4,5,9,6,7,8] # 3dt, 3dp, 3de, 3dv, 2d, mble, cmp, bse, bpe, jme
    self_sup_idxs = [4,5,6,7,8,9] # 2d, mble, cmp, bse, bpe, jme
    full_sup_idxs = [6,7,8,9,1] # cmp, bse, bpe, jme, 3dp
    print("Enter supervised subset and lr_decay to run (eg. 'dir:gt_2d,subset:S1.1,prior_code:200,lr:0.001,lr_decay:0.95')")
    exp_dir, sup_subset, priors_code, lr_str, lr_decay_str = input().split(",")
    if sup_subset.find('_')<0: start_idx = warmup_end[sup_subset] # epoch index after warmup
    else: start_idx = warmup_end[sup_subset[0:sup_subset.find('_')]] # epoch index after warmup
    ndarray = np.load("../experiments/{}/{}/{}/per_epoch_avg_losses.npy". format(exp_dir, sup_subset, priors_code))
    if ndarray.shape[0]<10:
        # losses for full-supervision without Traj-trn, 2D-trn, BLE-trn
        losses_ndarray = np.zeros((10, ndarray.shape[1]), dtype=np.float32)
        losses_ndarray[[1,2,3,6,7,8,9],:] = ndarray
    else: losses_ndarray = ndarray
    losses_ndarray = losses_ndarray[all_loss_idxs,:] # reorder
    loss_ids = [loss_ids[idx] for idx in all_loss_idxs] # reorder
    n_epochs = losses_ndarray.shape[1]
    # apply learning rate with decay
    lr = float(lr_str)
    gp_n_1 = np.arange(n_epochs) # (n) equivalent to (n-1) in geometric progression formula
    gp_r_n_1 = np.power(float(lr_decay_str), gp_n_1) # (n)
    gp_an_lr = lr * gp_r_n_1 # (4,n)
    losses_ndarray = losses_ndarray * gp_an_lr
    losses_ndarray = losses_ndarray[:,start_idx:]
    sup_combo_sota = np.sum(losses_ndarray[:2,:], axis=0) # 3d+traj
    self_sup_sota = np.sum(losses_ndarray[4:6,:], axis=0) # 2d+ble
    nw_epochs = losses_ndarray.shape[1]
    self_sup_combo_loss = np.sum(losses_ndarray[self_sup_idxs,:], axis=0) # 2dp+ble+cmp+bse+bpe+jme
    ss_regularizer_loss = np.sum(losses_ndarray[self_sup_idxs[1:],:], axis=0) # ble+cmp+bse+bpe+jme
    x_epochs = np.arange(nw_epochs) + start_idx+1
    print('total epochs:{}, warm-up epochs:{}, post warmup epochs:{}, lr_decay:{}'.
          format(n_epochs, start_idx, nw_epochs, lr_decay_str))

    sps_fig, sps_axs = plt.subplots(nrows=2, ncols=2, num='interactive_plot', figsize=(11, 9))
    plt.figure('interactive_plot')
    sps_fig.suptitle('{} Losses: Raw vs. Weighted'.format(sup_subset), fontweight='bold', size=12)
    sps_fig.tight_layout(pad=5.0)

    sps_axs[0][0].plot(x_epochs, losses_ndarray[0], color='C1') # traj-trn
    sps_axs[0][0].plot(x_epochs, losses_ndarray[1], color='C0') # 3d-trn
    sps_axs[0][0].plot(x_epochs, losses_ndarray[4], color='C2') # 2d-trn
    sps_axs[0][0].plot(x_epochs, losses_ndarray[5], '--', color='C3') # ble
    sps_axs[0][0].plot(x_epochs, losses_ndarray[6], color='C7') # cmp
    sps_axs[0][0].plot(x_epochs, losses_ndarray[7], color='C4') # bse
    sps_axs[0][0].set(title='Loss Terms', ylabel='Avg. Loss')
    sps_axs[0][0].legend(loss_ids[:2]+loss_ids[4:8])
    sps_axs[0][0].set_yscale('log')
    sps_axs[0][0].grid()

    sps_axs[0][1].plot(x_epochs, losses_ndarray[8], color='C6')
    sps_axs[0][1].plot(x_epochs, losses_ndarray[9], color='C5')
    sps_axs[0][1].plot(x_epochs, self_sup_combo_loss, color='C7')
    sps_axs[0][1].plot(x_epochs, ss_regularizer_loss, color='C8')
    sps_axs[0][1].set(title='Regularizers', ylabel='Avg. Loss')
    sps_axs[0][1].legend(loss_ids[8:]+['Self-Sup.','Reg-Losses'])
    sps_axs[0][1].set_yscale('symlog')
    sps_axs[0][1].grid()

    gp_n_1 = np.arange(nw_epochs).reshape((1,-1)) # (1,n) equivalent to (n-1) in geometric progression formula
    self_sup_wgts = np.ones((7,1), dtype=np.float32) # [2dp, mble, cmp, bse, bpe, jme, 3dp]
    self_sup_wgt_decay = np.ones((6,1), dtype=np.float32) # [2dp, mble, cmp, bse, bpe, jme]
    while True:
        # relevant to fully-supervised only
        wgt_sup_combo_losses = np.sum(losses_ndarray[full_sup_idxs,:] * self_sup_wgts[2:], axis=0) # cmp+bse+bpe+jme+3dp

        gp_r_n_1 = np.power(self_sup_wgt_decay, gp_n_1) # (4,n)
        gp_an_wgt = self_sup_wgts[:6] * gp_r_n_1 # (4,n)
        wgt_self_sup_losses = losses_ndarray[self_sup_idxs,:] * gp_an_wgt
        wgt_self_sup_combo_loss = np.sum(wgt_self_sup_losses, axis=0)
        wgt_reg_losses = np.sum(wgt_self_sup_losses[2:,:], axis=0) # cmp+bse+bpe+jme
        wgt_2dpvlog_ratios = np.abs(np.sum(wgt_self_sup_losses[4:], axis=0)) / wgt_self_sup_losses[0] # (bpe+jme) / 2dp
        wgt_2dpvlog_ratios *= lr*0.1 # just to scale to plot

        sps_axs[1][0].clear()
        sps_axs[1][0].plot(x_epochs, sup_combo_sota, color='C1')
        sps_axs[1][0].plot(x_epochs, wgt_sup_combo_losses, color='C0')
        sps_axs[1][0].plot(x_epochs, self_sup_sota, color='C3')
        sps_axs[1][0].plot(x_epochs, wgt_self_sup_combo_loss, color='C2')
        sps_axs[1][0].set(title='Combined Supervised vs. Semi-Supervised.', xlabel='Epochs', ylabel='Avg. Loss')
        sps_axs[1][0].legend(['SOTA Sup.', 'Wgt. Sup.', 'SOTA Semi-Sup', 'Wgt. Self-Sup'])
        sps_axs[1][0].set_yscale('symlog')
        sps_axs[1][0].grid()

        sps_axs[1][1].clear()
        sps_axs[1][1].plot(x_epochs, wgt_reg_losses, color='C1')
        sps_axs[1][1].plot(x_epochs, wgt_self_sup_losses[0], color='C2')
        sps_axs[1][1].plot(x_epochs, wgt_self_sup_losses[1], color='C3')
        sps_axs[1][1].plot(x_epochs, wgt_self_sup_losses[2], color='C7')
        sps_axs[1][1].plot(x_epochs, wgt_self_sup_losses[3], color='C4')
        sps_axs[1][1].plot(x_epochs, wgt_self_sup_losses[4], color='C6')
        sps_axs[1][1].plot(x_epochs, wgt_self_sup_losses[5], color='C5')
        sps_axs[1][1].plot(x_epochs, wgt_2dpvlog_ratios, '--', color='black')
        sps_axs[1][1].set(title='Weighted Semi-Supervised Losses', xlabel='Epochs', ylabel='Avg. Weighted Loss')
        sps_axs[1][1].legend(['Wgt. Reg','Wgt. 2D','Wgt. BLE','Wgt. PTO','Wgt. BSE','Wgt. BPE','Wgt. JME','P.2D Ratio'])
        sps_axs[1][1].set_yscale('symlog')
        sps_axs[1][1].grid()
        plt.pause(0.001)

        # interact
        print("Enter weights for self-supervised losses and weight decay:\n"
              "p2d,ble,cmp,bse,bpe,jme,nat_d,pos_d,log_d,p3d Or 'q' to terminate")
        self_sup_params_str = input()
        if self_sup_params_str=='q': break
        self_sup_params_num = self_sup_params_str.split(',')
        if len(self_sup_params_num)==10:
            for idx, numeral_str in enumerate(self_sup_params_num[:6]): self_sup_wgts[idx,0] = float(numeral_str)
            self_sup_wgt_decay[:2] = float(self_sup_params_num[6])
            self_sup_wgt_decay[2:4] = float(self_sup_params_num[7])
            self_sup_wgt_decay[4:] = float(self_sup_params_num[8])
            self_sup_wgts[6] = float(self_sup_params_num[9])
            print('  wgts:{}, wgt-decay:[{:.4f}, {:.4f}, {:.4f}], p3d:{:.2f}'.format(self_sup_wgts[:6,0],
                    self_sup_wgt_decay[1, 0], self_sup_wgt_decay[3, 0], self_sup_wgt_decay[5, 0], self_sup_wgts[6,0]))
        else:
            print('len(self_sup_params_num) should be 8 not {}'.format(len(self_sup_params_num)))



if __name__ == '__main__':
    d_nms = 0.05 # 0.01, 0.025, 0.05
    dups = 1
    quintuple = False
    group_props = False # *False
    group_fbjnts = False # *True
    print_metadata = False
    nms_tag = str(d_nms).replace("0.", ".")
    prop_fig_rowvcol = {9:(3,3), 10:(2,5), 15:(3,5), 114:(19,6)}

    rank1_perjnt_tags = ['UFace', 'Head',  'RHip', 'RThigh', 'RLeg', 'RShoulder', 'RBicep', 'RForearm',
                         'Thorax', 'Abdomen', 'LHip', 'LThigh', 'LLeg', 'LShoulder', 'LBicep', 'LForearm'] # 16
    rank1_grpjnt_tags = ['UFace', 'Head', 'Hip', 'Thigh', 'Leg', 'Thorax', 'Abdomen', 'Shoulder', 'Bicep', 'Forearm'] # 10
    rank2_perjnt_tags = ['RForearm-&-RBicep', 'LForearm-&-LBicep', 'RBicep-&-RShoulder', 'LBicep-&-LShoulder',
                         'Head-&-RShoulder', 'Head-&-LShoulder', 'UFace-&-Head', 'Thorax-&-RShoulder',
                         'Thorax-&-LShoulder', 'Head-&-Thorax', 'Thorax-&-Abdomen', 'Abdomen-&-RHip',
                         'Abdomen-&-LHip', 'RThigh-&-RHip', 'LThigh-&-LHip', 'RLeg-&-RThigh', 'LLeg-&-LThigh'] # 17
    rank2_grpjnt_tags = ['UFace-&-Bicep', 'Bicep-&-Shoulder', 'Head-&-Shoulder', 'UFace-&-Head', 'Thorax-&-Shoulder',
                         'Head-&-Thorax', 'Thorax-&-Abdomen', 'Abdomen-&-Hip', 'Thigh-&-Hip', 'Leg-&-Thigh'] # 10
    _rank1_id_tags = rank1_grpjnt_tags if group_fbjnts else rank1_perjnt_tags
    _rank2_id_tags = rank2_grpjnt_tags if group_fbjnts else rank2_perjnt_tags

    ptd_hist_range, ptd_hist_subticks = [(-2, 8), (0, 1.5), (0, 1)], [2, 4, 2]
    blen_hist_range, blen_hist_subticks = [(100, 500), (0, 220), (0, 6)], [4, 5, 2]
    bse_hist_range, bse_hist_subticks = [(0, 0.0001), (0, 300000), (0, 15), (0, 1)], [3, 5, 2, 2]

    bse_comp_tags = ['R/L-Thigh','R/L-Thigh','R/L-Leg','R/L-Bicep','R/L-Bicep','R/L-Forearm']
    blen_comp_tags = ['R-Hip','R-Thigh','R-Leg','L-Hip','L-Thigh','L-Leg','Abdomen','Vertebra',
                      'Head','UFace','L-Shoulder','L-Bicep','L-Forearm','R-Shoulder','R-Bicep','R-Forearm']

    mv_pdf_fbvec_likelihood_viz('S1', 'RP-BoneOrient', 16, frt_tag='', grp_jnts=group_fbjnts,
                                quintuple=quintuple, duplicate=dups, cross_plot_wgt=False)
    pdf_likelihood_combo_viz('S1', 'RP-PoseStruct_{}_{}_{}{}_m{}.pickle',9, .05, frt_tag='', grp_props=group_props)

    self_sup_losses_and_weights()

    if print_metadata:
        pdir = '../priors/S1'
        filepath = os.path.join('{}/bone_priors_{}_{}_br_{}_{:.0e}.pickle'.format(pdir, 'wxaug', 15, 'grpprp', 1))
        bpe_priors_params = pickle_load(filepath)
        print('\nBPC\n{}\n'.format(bpe_priors_params['summary_log']))

        filepath = os.path.join('{}/fbj_priors_{}{}{}_{}_{:.0e}.pickle'.format(pdir, '', 'noctr', '_wxaug_16', 'grpjnt', 1))
        jme_priors_params = pickle_load(filepath)
        print('\nJMC\n{}\n'.format(jme_priors_params['summary_log']))

    print('\n All done.\n')
