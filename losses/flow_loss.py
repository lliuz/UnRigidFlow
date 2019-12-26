import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd
from utils.warp_utils import flow_warp


class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        '''
        Configs:
            w_l1, w_ssim: weights of photometric loss.
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
            loss = [(l ** 2 + 1e-6) ** 0.5 for l in loss]

        return sum([l.mean() for l in loss]) / occu_mask1.mean()

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

    def forward(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons, occu_mask1 = flow_warp(im2_scaled, flow[:, :2], flow[:, 2:])
            im2_recons, occu_mask2 = flow_warp(im1_scaled, flow[:, 2:], flow[:, :2])

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            if self.cfg.hard_occu:
                occu_mask1 = (occu_mask1 > self.cfg.hard_occu_th).float()
                occu_mask2 = (occu_mask2 > self.cfg.hard_occu_th).float()

            loss_photomatric = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            if i == 0 and self.cfg.norm_smooth:
                s = min(h, w)

            if self.cfg.s_mask:
                loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled, occu_mask1)
            else:
                loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled, None)

            if self.cfg.with_bk:
                loss_photomatric += self.loss_photomatric(im2_scaled, im2_recons,
                                                          occu_mask2)

                if self.cfg.s_mask:
                    loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled,
                                                    occu_mask2)
                else:
                    loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled, None)
                loss_photomatric /= 2.
                loss_smooth /= 2.

            pyramid_warp_losses.append(loss_photomatric)
            pyramid_smooth_losses.append(loss_smooth)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.cfg.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]

        return sum(pyramid_warp_losses) + self.cfg.w_smooth * sum(pyramid_smooth_losses), \
               sum(pyramid_warp_losses), self.cfg.w_smooth * sum(pyramid_smooth_losses), \
               pyramid_flows[0].abs().mean()
