import torch.nn as nn
import torch.nn.functional as F
from utils.warp_utils import flow_warp
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, charbonnier, EPE
from utils.rigid_utils import depth_flow2pose_pt, depth_pose2flow_pt, \
    gaussianblur_pt, percentile_pt


class RigidFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(RigidFlowLoss, self).__init__()
        self.cfg = cfg

    def loss_consistancy(self, flow, rigid_flow, rigid_mask):
        """

        :param flow: [B, 2, H, W]
        :param rigid_flow:  [B, 2, H, W]
        :param rigid_mask:  [B, 1, H, W]
        :return:
        """
        loss = (flow - rigid_flow).abs().mean(1, keepdim=True) * rigid_mask
        loss = charbonnier(loss)
        l = loss.sum() / (rigid_mask.sum() + 1e-6)
        return l

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        '''
        Configs:
            w_l1, w_ssim, w_ternary: weights of photometric loss.
            mask_first: multiply mask before loss or not.
            p_charbonnier: charbonnier or not
        :return:
        '''
        loss = []

        # compute L1 loss
        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        # compute SSIM loss
        if self.cfg.w_ssim > 0:
            if self.cfg.mask_first:
                loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1,
                                                im1_scaled * occu_mask1)]
            else:
                loss += [self.cfg.w_ssim * \
                         SSIM(im1_recons, im1_scaled) * \
                         occu_mask1[:, :, 1:-1, 1:-1]]

        if self.cfg.p_charbonnier:
            loss = [charbonnier(l) for l in loss]

        return sum([l.mean() for l in loss]) / (occu_mask1.mean() + 1e-6)

    def loss_smooth(self, flow, im1_scaled, mask=None):
        loss = []

        # compute Smooth loss
        if self.cfg.w_s2 > 0:
            loss += [self.cfg.w_s2 * smooth_grad_2nd(flow, im1_scaled, 10., mask,
                                                     self.cfg.s_charbonnier)]
        if self.cfg.w_s1 > 0:
            loss += [self.cfg.w_s1 * smooth_grad_1st(flow, im1_scaled, 10., mask,
                                                     self.cfg.s_charbonnier)]

        return sum([l.mean() for l in loss])

    def forward(self, pyramid_disp, fl_bl, pyramid_K, pyramid_K_inv, raw_W,
                pyramid_flow, images):
        """

        :param pyramid_depths: Multi-scale disparities n * [B x h x w]
        :param fl_bl: focal length * baseline [B]
        :param pyramid_K: Multi-scale intrinsics n * [B, 3, 3]
        :param pyramid_K_inv: Multi-scale inverse of intrinsics n * [B, 3, 3]
        :param raw_W: Original width of images [B]
        :param pyramid_flows: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        B = images.size(0)
        im1_origin = images[:, :3]
        im2_origin = images[:, 3:]

        pyramid_l_photomatric = []
        pyramid_l_smooth = []
        pyramid_l_consistancy = []
        pyramid_l_photomatric_rigid = []
        pyramid_rigid_mask = []

        for i, (disp, flow, K, K_inv, md) in enumerate(zip(pyramid_disp, pyramid_flow,
                                                           pyramid_K, pyramid_K_inv,
                                                           self.cfg.pyramid_md)):
            # only the first n scales compute loss.
            if i >= self.cfg.valid_s:
                break
            _, _, h, w = flow.size()

            if i == 0 and self.cfg.norm_smooth:
                s = min(h, w)

            disp = F.interpolate(disp.unsqueeze(1), (h, w), mode='bilinear',
                                 align_corners=True).squeeze(1) * raw_W.reshape(-1, 1, 1)

            depth = fl_bl.reshape(-1, 1, 1) / disp.clamp(min=1e-3)  # [B, h ,w]

            # use the largest depth and flow to predict pose
            if i == 0:
                pose_mat, _, inlier_ratio = depth_flow2pose_pt(depth, flow[:, :2], K,
                                                               K_inv,
                                                               gs=16, th=2.,
                                                               method=self.cfg.PnP_method)

            rigid_flow = depth_pose2flow_pt(depth, pose_mat, K, K_inv)

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons, occu_mask1 = flow_warp(im2_scaled, flow[:, :2], flow[:, 2:])

            im1_recons_rigid = flow_warp(im2_scaled, rigid_flow)

            th_mask = EPE(flow[:, :2], rigid_flow) < self.cfg.mask_th / 2 ** i

            flow_e = F.pad(SSIM(im1_scaled, im1_recons, md=md), [md] * 4
                           ).mean(1, keepdim=True)  # [B, 1, h ,w]
            rigid_e = F.pad(SSIM(im1_scaled, im1_recons_rigid, md=md), [md] * 4
                            ).mean(1, keepdim=True)

            dist_e = rigid_e - flow_e
            dist_e = gaussianblur_pt(dist_e, (11, 11), 5)

            delta = percentile_pt(dist_e, th=self.cfg.recons_p).reshape(-1, 1, 1, 1)
            rigid_mask = dist_e < delta # [B, 1, h ,w]
            rigid_mask = rigid_mask & th_mask

            # mask out the failure depth region
            rigid_mask = rigid_mask & (depth.unsqueeze(1) < 80)

            # for the failure pose estimation, rigid_mask should be all false
            valid_poses = (inlier_ratio > 0.2).type_as(rigid_mask)
            rigid_mask = rigid_mask & valid_poses.reshape(-1, 1, 1, 1)

            rigid_mask = rigid_mask.float()

            # for the occlusion region, rigid_mask should be true or false
            if self.cfg.mask_with_occu:  # the original tf implementation:
                rigid_mask = (rigid_mask + (occu_mask1 < 0.2).float()).clamp(0., 1.)

            if self.cfg.smooth_mask_by == 'th':
                sm_mask = 1 - (th_mask & (depth.unsqueeze(1) < 80)).float()
            else:
                sm_mask = 1 - rigid_mask    # same as paper

            l_photomatric = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            l_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled, sm_mask)

            l_consistancy = self.loss_consistancy(flow[:, :2], rigid_flow.detach(),
                                                  rigid_mask)

            # occlusion mask?
            l_photomatric_rigid = self.loss_photomatric(im1_scaled, im1_recons_rigid,
                                                        rigid_mask)

            pyramid_l_photomatric.append(l_photomatric * self.cfg.w_scales[i])
            pyramid_l_smooth.append(l_smooth * self.cfg.w_sm_scales[i])
            pyramid_l_consistancy.append(l_consistancy * self.cfg.w_cons_scales[i])
            pyramid_l_photomatric_rigid.append(
                l_photomatric_rigid * self.cfg.w_rigid_scales[i])

            pyramid_rigid_mask.append(rigid_mask.mean() * B / (valid_poses.sum() + 1e-6))

        w_l_pohotometric = sum(pyramid_l_photomatric)
        w_l_pohotometric_rigid = sum(pyramid_l_photomatric_rigid)
        w_l_smooth = sum(pyramid_l_smooth)
        w_l_consistancy = sum(pyramid_l_consistancy)

        final_loss = w_l_pohotometric + \
                     self.cfg.w_rigid_warp * w_l_pohotometric_rigid + \
                     self.cfg.w_smooth * w_l_smooth + \
                     self.cfg.w_cons * w_l_consistancy

        return final_loss, w_l_pohotometric, w_l_pohotometric_rigid, \
               1000 * w_l_smooth, w_l_consistancy, \
               sum(pyramid_rigid_mask) / len(pyramid_disp), \
               inlier_ratio.mean()
