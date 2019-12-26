import cv2
import numpy as np


def load_flow(path):
    if path.endswith('.png'):
        # for KITTI which uses 16bit PNG images
        # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flo_file = cv2.imread(path, -1)
        flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
        invalid = (flo_file[:, :, 0] == 0)  # mask
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0
        return flo_img, np.expand_dims(flo_file[:, :, 0], 2)
    else:
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (w, h, 2))
        return data2D


def evaluate_kitti_flow(gt_flows, pred_flows, moving_masks=None):
    def calculate_error_rate(epe_map, gt_flow, mask):
        bad_pixels = np.logical_and(
            epe_map * mask > 3,
            epe_map * mask / np.maximum(
                np.sqrt(np.sum(np.square(gt_flow), axis=2)), 1e-10) > 0.05)
        return bad_pixels.sum() / mask.sum() * 100.

    error, error_noc, error_occ, error_move, error_static, error_rate = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    error_move_rate, error_static_rate = 0.0, 0.0
    B = len(gt_flows)
    for gt_flow, pred_flow, i in zip(gt_flows, pred_flows, range(B)):
        H, W = gt_flow.shape[:2]

        h, w = pred_flow.shape[:2]
        pred_flow = np.copy(pred_flow)
        pred_flow[:, :, 0] = pred_flow[:, :, 0] / w * W
        pred_flow[:, :, 1] = pred_flow[:, :, 1] / h * H

        # flo_pred = sm.imresize(pred_flow, (H, W), interp="bilinear", mode='F')
        flo_pred = cv2.resize(pred_flow, (W, H), interpolation=cv2.INTER_LINEAR)

        epe_map = np.sqrt(
            np.sum(np.square(flo_pred[:, :, :2] - gt_flow[:, :, :2]),
                   axis=2))
        if gt_flow.shape[-1] == 2:
            error += np.mean(epe_map)

        elif gt_flow.shape[-1] == 4:
            error += np.sum(epe_map * gt_flow[:, :, 2]) / np.sum(gt_flow[:, :, 2])
            noc_mask = gt_flow[:, :, -1]
            error_noc += np.sum(epe_map * noc_mask) / np.sum(noc_mask)

            error_occ += np.sum(epe_map * (gt_flow[:, :, 2] - noc_mask)) / max(
                np.sum(gt_flow[:, :, 2] - noc_mask), 1.0)

            error_rate += calculate_error_rate(epe_map, gt_flow[:, :, 0:2],
                                               gt_flow[:, :, 2])

            if moving_masks is not None:
                move_mask = moving_masks[i]

                error_move_rate += calculate_error_rate(
                    epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2] * move_mask)
                error_static_rate += calculate_error_rate(
                    epe_map, gt_flow[:, :, 0:2],
                    gt_flow[:, :, 2] * (1.0 - move_mask))

                error_move += np.sum(epe_map * gt_flow[:, :, 2] *
                                     move_mask) / np.sum(gt_flow[:, :, 2] *
                                                         move_mask)
                error_static += np.sum(epe_map * gt_flow[:, :, 2] * (
                        1.0 - move_mask)) / np.sum(gt_flow[:, :, 2] *
                                                   (1.0 - move_mask))

    if gt_flows[0].shape[-1] == 4:
        res = [error / B, error_noc / B, error_occ / B, error_rate / B]
        if moving_masks is not None:
            res += [error_move / B, error_static / B]
        return res
    else:
        return [error / B]