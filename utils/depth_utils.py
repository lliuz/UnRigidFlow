import cv2
import scipy.misc as sm


def load_disp(path):
    return sm.imread(path, -1).astype(np.float32) / 256.0


def convert_disp_to_depth(disp, cam2cam=None, im_size=None, normed=True, raw_w=None,
                          fl_bl=None):
    """

    :param disp:    disparity, can be normed or not.
    :param cam2cam: to read baseline and focal length
    :param im_size: can be raw size or other size.
    :param normed:  if disparity is normed.
    :param raw_w:   required if im_size is not the raw size.
    :return:
    """
    if im_size is not None:
        disp = cv2.resize(disp, (im_size[1], im_size[0]),
                          interpolation=cv2.INTER_LINEAR)
    if normed:
        disp *= raw_w if raw_w is not None else disp.shape[1]

    if fl_bl is None:
        bl, fl = get_focal_length_baseline(cam2cam)
        fl_bl = fl * bl

    depth = fl_bl / np.clip(disp, a_min=1e-3, a_max=None)
    depth[np.isinf(depth)] = 0
    return depth


def read_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def get_focal_length_baseline(cam2cam, cam=2):
    P2_rect = cam2cam['P_rect_02'].reshape(3, 4)
    P3_rect = cam2cam['P_rect_03'].reshape(3, 4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0, 3] / -P2_rect[0, 0]
    b3 = P3_rect[0, 3] / -P3_rect[0, 0]
    baseline = b3 - b2

    focal_length = None
    if cam == 2:
        focal_length = P2_rect[0, 0]
    elif cam == 3:
        focal_length = P3_rect[0, 0]

    return focal_length, baseline


import numpy as np


# Adopted from https://github.com/mrharicot/monodepth
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / (gt))

    sq_rel = np.mean(((gt - pred) ** 2) / (gt))

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_depth_errors(gt, pred, crop=True):
    errors = np.array([0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
    batch_size = len(gt)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            h, w = current_gt.shape[:2]
            crop_mask = current_gt != current_gt
            y1, y2 = int(0.40810811 * h), int(0.99189189 * h)
            x1, x2 = int(0.03594771 * w), int(0.96405229 * w)
            crop_mask[y1:y2, x1:x2] = 1
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clip(1e-3, 80)

        # rescale depth
        # valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)
        res = compute_errors(valid_gt, valid_pred)
        errors += np.array(res)
    return list(errors / batch_size)
