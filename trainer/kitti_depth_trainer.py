import time
import torch
import numpy as np
from .base_trainer import BaseTrainer
from utils.misc_utils import AverageMeter
from utils.depth_utils import load_disp, convert_disp_to_depth, compute_depth_errors


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.cfg.epoch_size:
                break
            # read data to device
            img1, img2 = data['img1'], data['img2']
            img1r, img2r = data['img1r'], data['img2r']

            if np.random.rand(1) > 0.5:
                img_l = img1.to(self.device)  # left
                img_r = img1r.to(self.device)  # right
            else:
                img_l = img2.to(self.device)  # left
                img_r = img2r.to(self.device)  # right

            t_in = torch.cat([img_l, img_r], 1)
            # measure data loading time
            am_data_time.update(time.time() - end)

            # compute output
            disparities = self.model(t_in)[:4]
            loss = self.loss_func(disparities, [img_l, img_r])

            # update meters
            key_meters.update([loss.item()], img_l.size(0))

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

        error_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        error_meters = AverageMeter(i=len(error_names))

        self.model.eval()
        self.model = self.model.float()
        end = time.time()
        for i_step, data in enumerate(self.valid_loader):
            img_l, img_r = data['img1'], data['img1r']
            t_in = torch.cat([img_l, img_r], 1).to(self.device)

            # compute output
            disparities = self.model(t_in)[:4]

            disp_lr = disparities[0].detach().cpu().numpy()
            disp = disp_lr[:, 0, :, :]

            gt_disp_occ = list(map(load_disp, data['disp_occ']))
            fl_bl = [f.detach().cpu().numpy() for f in data['fl_bl']]
            gt_depth_occ = list(
                map(lambda p, q: convert_disp_to_depth(p, normed=False, fl_bl=q),
                    gt_disp_occ, fl_bl))
            im_size = list(map(lambda p: p.shape[:2], gt_disp_occ))
            pred_depth = list(map(lambda p, q, r: convert_disp_to_depth(
                p, None, q, fl_bl=r), disp, im_size, fl_bl))

            err_depth = compute_depth_errors(gt_depth_occ, pred_depth)
            error_meters.update(err_depth, img_l.size(0))

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
            self.save_model(error_meters.avg[0], 'KITTI_depth')

        return error_meters.avg, error_names
