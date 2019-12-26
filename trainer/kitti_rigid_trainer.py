import time
import torch
import numpy as np
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from utils.flow_utils import load_flow, evaluate_kitti_flow
from utils.depth_utils import load_disp, convert_disp_to_depth, compute_depth_errors
from utils.torch_utils import load_checkpoint, save_checkpoint, bias_parameters, \
    weight_parameters, AdamW
from utils.misc_utils import AverageMeter


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config)
        self.best_errors = [np.inf] * 2

    def _init_model(self, models):
        models = [m.to(self.device) for m in models]
        if self.cfg.pretrained_model:
            self._log.info("=> Flow model using pre-trained weights {}.".format(
                self.cfg.pretrained_model))
            epoch, weights = load_checkpoint(self.cfg.pretrained_model)
            models[0].load_state_dict(weights)
        else:
            self._log.info("=> Train flow model from scratch.")
            models[0].init_weights()

        if self.cfg.pretrained_model_depth:
            self._log.info("=> Depth model using pre-trained weights {}.".format(
                self.cfg.pretrained_model_depth))
            epoch, weights = load_checkpoint(self.cfg.pretrained_model_depth)
            models[1].load_state_dict(weights)
        else:
            self._log.info("=> Train depth model from scratch.")
            models[1].init_weights()

        models = [torch.nn.DataParallel(m, device_ids=self.device_ids) for m in models]
        return models

    def save_model(self, errors, names):
        is_best_depth = errors[0] < self.best_errors[0]
        is_best_flow = errors[1] < self.best_errors[1]

        if is_best_depth:
            self.best_error[0] = errors[0]

        model = {'epoch': self.i_epoch,
                 'state_dict': self.model[1].module.state_dict()}
        save_checkpoint(self.save_root, model, names[0], is_best_depth)

        if is_best_flow:
            self.best_error[1] = errors[1]
        model = {'epoch': self.i_epoch,
                 'state_dict': self.model[0].module.state_dict()}
        save_checkpoint(self.save_root, model, names[1], is_best_flow)

    def _create_optimizer(self):
        self._log.info('=> setting Adam solver')

        param_groups = [
            {'params': bias_parameters(self.model[0].module),
             'weight_decay': self.cfg.bias_decay,
             'lr': self.cfg.lr_flow},
            {'params': weight_parameters(self.model[0].module),
             'weight_decay': self.cfg.wd_flow,
             'lr': self.cfg.lr_flow},
        ]
        if self.cfg.train_depth:
            param_groups += [
                {'params': bias_parameters(self.model[1].module),
                 'weight_decay': self.cfg.bias_decay,
                 'lr': self.cfg.lr_depth},
                {'params': weight_parameters(self.model[1].module),
                 'weight_decay': self.cfg.wd_depth,
                 'lr': self.cfg.lr_depth}
            ]
        else:
            for param in self.model[1].parameters():
                param.requires_grad = False

        if self.cfg.optim == 'adamw':
            optimizer = AdamW(param_groups, betas=(self.cfg.momentum, self.cfg.beta))
        elif self.cfg.optim == 'adam':
            optimizer = torch.optim.Adam(param_groups,
                                         betas=(self.cfg.momentum, self.cfg.beta),
                                         eps=1e-7)
        else:
            raise NotImplementedError(self.cfg.optim)
        return optimizer

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'Loss_depth', 'Loss_flow',
                           'lf_1', 'lf_2', 'lf_3', 'lf_4', 'lf_5', 'in_r']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model[0].train()
        if self.cfg.train_depth:
            self.model[1].train()
        else:
            self.model[1].eval()
        end = time.time()

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.cfg.epoch_size:
                break
            # read data to device
            img1, img2 = data['img1'], data['img2']
            img1r, img2r = data['img1r'], data['img2r']

            img_pair = torch.cat([img1, img2], 1).to(self.device)
            # random pick a stereo image pair from img1 and img2
            if np.random.rand(1) > 0.5:
                img_l = img1.to(self.device)  # left
                img_r = img1r.to(self.device)  # right
            else:
                img_l = img2.to(self.device)  # left
                img_r = img2r.to(self.device)  # right

            fl_bl = data['fl_bl'].to(self.device).type_as(img_pair)
            pyramid_K = list(map(
                lambda p: p.to(self.device).type_as(img_pair), data['pyramid_K']))
            pyramid_K_inv = list(map(
                lambda p: p.to(self.device).type_as(img_pair), data['pyramid_K_inv']))
            raw_W = data['im_shape'][1].to(self.device).type_as(img_pair)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # compute output
            flows = self.model[0](img_pair, with_bk=True)  # n * [B, 4, h / 4, w / 4]

            t_in = torch.cat([img_l, img_r], 1)
            disparities = self.model[1](t_in)[:4]  # n * [B, 2, h , w]
            disps = [d[:, 0] for d in disparities]

            # compute loss
            if self.cfg.train_depth:
                l_depth = self.loss_func[0](disparities, [img_l, img_r])
            else:
                l_depth = torch.tensor(0).type_as(img_l)

            flow_res = self.loss_func[1](disps, fl_bl, pyramid_K,
                                         pyramid_K_inv, raw_W, flows[:4], img_pair)

            loss = l_depth + flow_res[0]

            # update meters
            key_meters.update(
                [loss.item(), l_depth.item(), flow_res[0].item(),
                 flow_res[1].item(), flow_res[2].item(),
                 flow_res[3].item(), flow_res[4].item(), flow_res[5].item(),
                 flow_res[6].item()],
                img_pair.size(0))

            # compute gradient and do optimization step
            self.optimizer.zero_grad()
            # loss.backward()

            scaled_loss = 1024. * loss
            scaled_loss.backward()

            for param in [p for p in self.model[0].parameters() if p.requires_grad]:
                param.grad.data.mul_(1. / 1024)

            for param in [p for p in self.model[1].parameters() if p.requires_grad]:
                param.grad.data.mul_(1. / 1024)

            self.optimizer.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if self.i_iter % self.cfg.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)

            if self.i_iter % self.cfg.print_freq == 0:
                istr = '{}:{:04d}/{:04d}'.format(
                    self.i_epoch, i_step, self.cfg.epoch_size) + \
                       ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
                       ' Loss {}'.format(key_meters)
                self._log.info(istr)

            self.i_iter += 1
        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        batch_time = AverageMeter()

        error_names = ['rmse', 'rmse_log', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3',
                       'EPE', 'E_noc', 'E_occ', 'F1_all']
        error_meters = AverageMeter(i=len(error_names))

        [m.eval() for m in self.model]
        end = time.time()
        for i_step, data in enumerate(self.valid_loader):
            img1, img2, img_r = data['img1'], data['img2'], data['img1r']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            fl_bl = data['fl_bl'].to(self.device).type_as(img_pair)
            pyramid_K = list(map(
                lambda p: p.to(self.device).type_as(img_pair), data['pyramid_K']))
            pyramid_K_inv = list(map(
                lambda p: p.to(self.device).type_as(img_pair), data['pyramid_K_inv']))
            raw_W = data['im_shape'][1].to(self.device).type_as(img_pair)

            # compute output
            flows = self.model[0](img_pair, with_bk=True)
            disparities = self.model[1](torch.cat([img1, img_r], 1).to(self.device))

            disps = [d[:, 0] for d in disparities[:4]]
            self.loss_func[1](disps, fl_bl, pyramid_K, pyramid_K_inv, raw_W,
                              flows[:4], img_pair)

            disp_lr = disparities[0].detach().cpu().numpy()
            disp = disp_lr[:, 0, :, :]  # only the largest left disp is used

            gt_disp_occ = list(map(load_disp, data['disp_occ']))
            fl_bl_np = [f.detach().cpu().numpy() for f in fl_bl]
            gt_depth_occ = list(
                map(lambda p, q: convert_disp_to_depth(p, normed=False, fl_bl=q),
                    gt_disp_occ, fl_bl_np))
            im_size = list(map(lambda p: p.shape[:2], gt_disp_occ))
            pred_depth = list(
                map(lambda p, q, r: convert_disp_to_depth(p, None, q, fl_bl=r),
                    disp, im_size, fl_bl_np))

            err_depth = compute_depth_errors(gt_depth_occ, pred_depth)

            flow = flows[0][:, :2].detach().cpu().numpy().transpose([0, 2, 3, 1])
            res = list(map(load_flow, data['flow_occ']))
            gt_flows, occ_masks = [r[0] for r in res], [r[1] for r in res]
            res = list(map(load_flow, data['flow_noc']))
            _, noc_masks = [r[0] for r in res], [r[1] for r in res]

            gt_flows = [np.concatenate([f, o, no], axis=2) for
                        f, o, no in zip(gt_flows, occ_masks, noc_masks)]
            err_flow = evaluate_kitti_flow(gt_flows, flow)

            error_meters.update(err_depth + err_flow, img_pair.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i_step % self.cfg.print_freq == 0:
                self._log.info('Test: [{0}/{1}]\t Time {2}\t '.format(
                    i_step, self.cfg.valid_size, batch_time) + ' '.join(
                    map('{:.2f}'.format, error_meters.avg)))

            if i_step > self.cfg.valid_size:
                break

        # write error to tf board.
        for value, name in zip(error_meters.avg, error_names):
            self.summary_writer.add_scalar('Valid_' + name, value, self.i_epoch)

        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_model([error_meters.avg[0], error_meters.avg[7]],
                            ['KITTI_rigid_depth', 'KITTI_rigid_flow'])

        return error_meters.avg, error_names

    def _validate_with_gt_1(self):
        from utils.rigid_utils import depth_flow2pose_pt, depth_pose2flow_pt, \
            gaussianblur_pt, percentile_pt
        from utils.warp_utils import flow_warp
        from losses.loss_blocks import EPE, SSIM
        import cv2
        import scipy.misc as sm
        from temp.evaluate_mask import eval_mask

        batch_time = AverageMeter()

        error_names = ['EPE', 'E_noc', 'E_occ', 'F1_all',
                       'Pixel Acc', 'Mean Acc', 'Mean IoU', 'f.w. IoU']
        # ['rmse', 'rmse_log', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3',
        # 'EPE', 'E_noc', 'E_occ', 'F1_all']
        error_meters = AverageMeter(i=len(error_names))

        [m.eval() for m in self.model]
        end = time.time()
        for i_step, data in enumerate(self.valid_loader):
            img1, img2, img_r = data['img1'], data['img2'], data['img1r']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            fl_bl = data['fl_bl'].to(self.device).type_as(img_pair)
            pyramid_K = list(map(
                lambda p: p.to(self.device).type_as(img_pair), data['pyramid_K']))
            pyramid_K_inv = list(map(
                lambda p: p.to(self.device).type_as(img_pair), data['pyramid_K_inv']))
            raw_W = data['im_shape'][1].to(self.device).type_as(img_pair)

            res = list(map(load_flow, data['flow_occ']))
            gt_flows, occ_masks = [r[0] for r in res], [r[1] for r in res]
            get_obj_masks = lambda p: (sm.imread(p.replace('flow_occ', 'obj_map')) != 0)
            obj_masks = list(map(get_obj_masks, data['flow_occ']))
            res = list(map(load_flow, data['flow_noc']))
            _, noc_masks = [r[0] for r in res], [r[1] for r in res]

            # compute pose
            flows = self.model[0](img_pair, with_bk=True)
            disparities = self.model[1](torch.cat([img1, img_r], 1).to(self.device))

            disps = [d[:, 0] for d in disparities[:4]]

            disp, flow, K, K_inv, = disps[0], flows[0], \
                                    pyramid_K[0], pyramid_K_inv[0]

            _, _, h, w = flow.size()
            h, w = h * 4, w * 4
            disp = F.interpolate(disp.unsqueeze(1), (h, w), mode='bilinear',
                                 align_corners=True).squeeze(1) * raw_W.reshape(-1, 1, 1)

            depth = fl_bl.reshape(-1, 1, 1) / disp.clamp(min=1e-3)
            flow = F.interpolate(flow * 4, (h, w), mode='bilinear', align_corners=True)

            pose_mat, _, inlier_ratio = depth_flow2pose_pt(depth, flow[:, :2], K,
                                                           K_inv,
                                                           gs=16, th=2,
                                                           method='PnP')

            i_scale = 0
            disp, flow, K, K_inv, = disps[i_scale], flows[i_scale], \
                                    pyramid_K[i_scale], pyramid_K_inv[i_scale]
            mask_th = 3
            md = 3
            ph_p = 80
            k_gauss = 15
            s_gauss = 5

            _, _, h, w = flow.size()
            h, w = h * 4, w * 4
            disp = F.interpolate(disp.unsqueeze(1), (h, w), mode='bilinear',
                                 align_corners=True).squeeze(1) * raw_W.reshape(-1, 1, 1)

            depth = fl_bl.reshape(-1, 1, 1) / disp.clamp(min=1e-3)
            flow = F.interpolate(flow * 4, (h, w), mode='bilinear', align_corners=True)

            rigid_flow = depth_pose2flow_pt(depth, pose_mat, K, K_inv)

            # mask from non-rigid residual
            th_mask = EPE(flow[:, :2], rigid_flow) < mask_th / 2. ** i_scale

            # mask from photometric error
            im1_origin = img_pair[:, :3]
            im2_origin = img_pair[:, 3:]
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons, occu_mask1 = flow_warp(im2_scaled, flow[:, :2], flow[:, 2:])

            im1_recons_rigid = flow_warp(im2_scaled, rigid_flow)

            flow_e = F.pad(SSIM(im1_scaled, im1_recons, md=md), [md] * 4
                           ).mean(1, keepdim=True) * 0.85 + 0.15 * (
                                 im1_scaled - im1_recons).abs()
            rigid_e = F.pad(SSIM(im1_scaled, im1_recons_rigid, md=md), [md] * 4
                            ).mean(1, keepdim=True) * 0.85 + 0.15 * (
                                  im1_scaled - im1_recons_rigid).abs()

            dist_e = rigid_e - flow_e
            dist_e = gaussianblur_pt(dist_e, (k_gauss, k_gauss), s_gauss)

            delta = percentile_pt(dist_e, th=ph_p).reshape(-1, 1, 1, 1)
            ph_mask = dist_e < delta

            rigid_mask = (ph_mask | th_mask)  # | (occu_mask1 < 0.2)
            valid_poses = inlier_ratio > 0.3
            rigid_mask = (rigid_mask & valid_poses.reshape(-1, 1, 1, 1)).float()

            # cv2.imshow('occu_mask', rigid_mask[0].detach().cpu().numpy()[0] * 255)
            # cv2.waitKey()
            fused_flow = rigid_mask * rigid_flow + (1 - rigid_mask) * flow[:, :2]

            # cv2.imshow('img1',
            #            im1_origin.detach().cpu().numpy()[0].transpose([1, 2, 0])[:, :,
            #            ::-1])
            # cv2.imshow('th_mask',
            #            th_mask.detach().cpu().numpy()[0].transpose([1, 2, 0])[:, :,
            #            ::-1] * 255)
            # cv2.imshow('occu_mask',
            #            (occu_mask1 < 0.2).detach().cpu().numpy()[0].transpose([1, 2, 0])[
            #            :, :,
            #            ::-1] * 255)
            # cv2.imshow('obj_masks',
            #            obj_masks[0].astype(np.float32) * 255)
            # cv2.imshow('gt_occu_mask',
            #            (noc_masks[0] - occ_masks[0]).astype(np.float32) * 255)
            #
            # # FN FP
            #
            # cv2.waitKey()

            # eval object masks

            obj_masks_pred = np.split(1 - rigid_mask.detach().cpu().numpy(),
                                      rigid_mask.size(0))

            errs_mask = eval_mask([np.squeeze(i) for i in obj_masks_pred], obj_masks)

            fused_flow = fused_flow.detach().cpu().numpy().transpose([0, 2, 3, 1])

            gt_flows = [np.concatenate([f, o, no], axis=2) for
                        f, o, no in zip(gt_flows, occ_masks, noc_masks)]
            err_flow = evaluate_kitti_flow(gt_flows, fused_flow)

            error_meters.update(err_flow + list(errs_mask[:-1]), img_pair.size(0))
            if error_meters.avg[0] > 100:
                print('.')

            #
            #
            #
            #
            # disp_lr = disparities[0].detach().cpu().numpy()
            # disp = disp_lr[:, 0, :, :]  # only the largest left disp is used
            #
            # gt_disp_occ = list(map(load_disp, data['disp_occ']))
            # fl_bl_np = [f.detach().cpu().numpy() for f in fl_bl]
            #
            # gt_depth_occ = list(
            #     map(lambda p, q: convert_disp_to_depth(p, normed=False, fl_bl=q),
            #         gt_disp_occ, fl_bl_np))
            # im_size = list(map(lambda p: p.shape[:2], gt_disp_occ))
            # pred_depth = list(
            #     map(lambda p, q, r: convert_disp_to_depth(p, None, q, fl_bl=r),
            #         disp, im_size, fl_bl_np))
            #
            # err_depth = compute_depth_errors(gt_depth_occ, pred_depth)
            #

            #
            # # measure elapsed time
            # error_meters.update(err_depth + err_flow, )
            #
            # batch_time.update(time.time() - end)
            # end = time.time()
            #
            if i_step % self.cfg.print_freq == 0:
                self._log.info('Test: [{0}/{1}]\t Time {2}\t '.format(
                    i_step, self.cfg.valid_size, batch_time) + ' '.join(
                    map('{:.2f}'.format, error_meters.avg)))

            if i_step > self.cfg.valid_size:
                break

        # write error to tf board.
        for value, name in zip(error_meters.avg, error_names):
            self.summary_writer.add_scalar('Valid_' + name, value, self.i_epoch)

        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_checkpoint([error_meters.avg[0], error_meters.avg[7]],
                                 ['KITTI_rigid_depth', 'KITTI_rigid_flow'])

        return error_meters.avg, error_names
