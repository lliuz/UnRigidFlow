import time
import torch
import numpy as np
from .base_trainer import BaseTrainer
from utils.flow_utils import load_flow, evaluate_kitti_flow
from utils.misc_utils import AverageMeter


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l1', 'l2', 'flow_ave']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.cfg.epoch_size:
                break
            # read data to device
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # compute output
            output = self.model(img_pair, with_bk=True)
            loss, l1, l2, fm = self.loss_func(output, img_pair)

            # update meters
            key_meters.update([loss.item(), l1.item(), l2.item(), fm.item()],
                              img_pair.size(0))

            # compute gradient and do optimization step
            self.optimizer.zero_grad()
            # loss.backward()

            scaled_loss = 1024. * loss
            scaled_loss.backward()

            for param in [p for p in self.model.parameters() if p.requires_grad]:
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
                       ' Loss/EPE {}'.format(key_meters)
                self._log.info(istr)

            self.i_iter += 1
        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        batch_time = AverageMeter()

        error_names = ['EPE', 'E_noc', 'E_occ', 'F1_all']
        error_meters = AverageMeter(i=len(error_names))

        self.model.eval()
        self.model = self.model.float()
        end = time.time()
        for i_step, data in enumerate(self.valid_loader):
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            # compute output
            output = self.model(img_pair)

            res = list(map(load_flow, data['flow_occ']))
            gt_flows, occ_masks = [r[0] for r in res], [r[1] for r in res]
            res = list(map(load_flow, data['flow_noc']))
            _, noc_masks = [r[0] for r in res], [r[1] for r in res]

            gt_flows = [np.concatenate([flow, occ_mask, noc_mask], axis=2) for
                        flow, occ_mask, noc_mask in zip(gt_flows, occ_masks, noc_masks)]
            pred_flows = output[0].detach().cpu().numpy().transpose([0, 2, 3, 1])
            es = evaluate_kitti_flow(gt_flows, pred_flows)
            error_meters.update([l.item() for l in es], img_pair.size(0))

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
            self.save_model(error_meters.avg[0], 'KITTI_flow')

        return error_meters.avg, error_names

    @torch.no_grad()
    def _validate_with_gt2(self):
        import cv2
        import torch.nn.functional as F
        from utils.warp_utils import flow_warp
        from utils.misc_utils import plot_imgs

        batch_time = AverageMeter()

        error_names = ['EPE', 'E_noc', 'E_occ', 'F1_all']
        error_meters = AverageMeter(i=len(error_names))

        self.model.eval()
        self.model = self.model.float()
        end = time.time()
        for i_step, data in enumerate(self.valid_loader):
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            # compute output
            flow = self.model(img_pair, with_bk=True)[0]
            _, _, h, w = flow.size()

            im1_origin = img_pair[:, :3]
            _, occu_mask1 = flow_warp(im1_origin, flow[:, :2], flow[:, 2:])

            res = list(map(load_flow, data['flow_occ']))
            gt_flows, occ_masks = [r[0] for r in res], [r[1] for r in res]
            res = list(map(load_flow, data['flow_noc']))
            _, noc_masks = [r[0] for r in res], [r[1] for r in res]

            gt_flows = [np.concatenate([flow, occ_mask, noc_mask], axis=2) for
                        flow, occ_mask, noc_mask in zip(gt_flows, occ_masks, noc_masks)]
            pred_flows = flow[:, :2].detach().cpu().numpy().transpose([0, 2, 3, 1])
            es = evaluate_kitti_flow(gt_flows, pred_flows)
            error_meters.update([l.item() for l in es], img_pair.size(0))

            plot_list = []
            occu_mask1 = (occu_mask1 < 0.2).detach().cpu().numpy()[0, 0] * 255
            plot_list.append({'im': occu_mask1, 'title': 'occu mask 1'})

            gt_occu_mask1 = (noc_masks[0] - occ_masks[0])[:, :, 0].astype(
                np.float32) * 255
            plot_list.append({'im': gt_occu_mask1, 'title': 'gt occu mask 1'})
            plot_imgs(plot_list,
                      save_path='./tmp/occu_soft_hard/occu_hard_{:03d}.jpg'.format(
                          i_step))

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
            self.save_model(error_meters.avg[0], 'KITTI_flow')

        return error_meters.avg, error_names
