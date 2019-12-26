import torch
import numpy as np
import cv2
from utils.warp_utils import mesh_grid


def get_valid_depth(gt, crop=True):
    valid = (gt > 0) & (gt < 80)
    if crop:
        h, w = gt.shape[:2]
        crop_mask = gt != gt
        y1, y2 = int(0.40810811 * h), int(0.99189189 * h)
        x1, x2 = int(0.03594771 * w), int(0.96405229 * w)
        crop_mask[y1:y2, x1:x2] = 1
        valid = valid & crop_mask
    return valid


def depth_flow2pose(depth, flow, K, K_inv, gs=16, th=1., method='AP3P', depth2=None):
    """

    :param depth:       H x W
    :param flow:        h x w x2
    :param K:           3 x 3
    :param K_inv:       3 x 3
    :param gs:          grad size for sampling
    :param th:          threshold for RANSAC
    :param method:      PnP method
    :return:
    """
    if method == 'PnP':
        PnP_method = cv2.SOLVEPNP_ITERATIVE
    elif method == 'AP3P':
        PnP_method = cv2.SOLVEPNP_AP3P
    elif method == 'EPnP':
        PnP_method = cv2.SOLVEPNP_EPNP
    else:
        raise ValueError('PnP method ' + method)

    H, W = depth.shape[:2]
    valid_mask = get_valid_depth(depth)
    sample_mask = np.zeros_like(valid_mask)
    sample_mask[::gs, ::gs] = 1
    valid_mask &= sample_mask == 1

    h, w = flow.shape[:2]
    flow[:, :, 0] = flow[:, :, 0] / w * W
    flow[:, :, 1] = flow[:, :, 1] / h * H
    flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)

    grid = np.stack(np.meshgrid(range(W), range(H)), 2).astype(
        np.float32)  # HxWx2
    one = np.expand_dims(np.ones_like(grid[:, :, 0]), 2)
    homogeneous_2d = np.concatenate([grid, one], 2)
    d = np.expand_dims(depth, 2)
    points_3d = d * (K_inv @ homogeneous_2d.reshape(-1, 3).T).T.reshape(H, W, 3)

    points_2d = grid + flow
    valid_mask &= (points_2d[:, :, 0] < W) & (points_2d[:, :, 0] >= 0) & \
                  (points_2d[:, :, 1] < H) & (points_2d[:, :, 1] >= 0)

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d[valid_mask],
                                                  points_2d[valid_mask],
                                                  K, np.zeros([4, 1]),
                                                  reprojectionError=th,
                                                  flags=PnP_method)
    if not ret:
        inlier_ratio = 0.
    else:
        inlier_ratio = len(inliers) / np.sum(valid_mask)
    pose_mat = np.eye(4, dtype=np.float32)
    pose_mat[:3, :] = cv2.hconcat([cv2.Rodrigues(rvec)[0], tvec])

    return pose_mat, np.concatenate([rvec, tvec]), inlier_ratio


def depth_flow2pose_pt(depth, flow, K, K_inv, gs=16, th=1., method='AP3P'):
    """
    This operation is non-differentiable and is run with the original image size only.
    :param depth:       B x H x W
    :param flow:        B x 2 x h x w
    :param K:           B x 3 x 3
    :param K_inv:       B x 3 x 3
    :param gs:          grad size for sampling
    :param th:          threshold for RANSAC
    :param method:      PnP method
    :return:
    """

    B = depth.size(0)
    dtype = depth.type()
    depth = [arr.squeeze(0) for arr in
             np.split(depth.detach().cpu().numpy(), B, axis=0)]
    flow = [arr.squeeze(0) for arr in
            np.split(flow.detach().cpu().numpy().transpose([0, 2, 3, 1]), B, axis=0)]
    K = [arr.squeeze(0) for arr in
         np.split(K.detach().cpu().numpy(), B, axis=0)]
    K_inv = [arr.squeeze(0) for arr in
             np.split(K_inv.detach().cpu().numpy(), B, axis=0)]

    pose_mat = []
    pose_vec = []
    inlier_ratio = []
    for i, (a, b, c, d) in enumerate(zip(depth, flow, K, K_inv)):
        mat, vec, r = depth_flow2pose(a, b, c, d, gs=gs, th=th, method=method)
        pose_mat.append(mat)
        pose_vec.append(vec)
        inlier_ratio.append(r)

    pose_mat = torch.tensor(np.stack(pose_mat)).type(dtype)
    pose_vec = torch.tensor(np.stack(pose_vec)).type(dtype)
    inlier_ratio = torch.tensor(np.stack(inlier_ratio)).type(dtype)
    return pose_mat, pose_vec, inlier_ratio


def gaussianblur_pt(x_batch, kernel_sz, sigma):
    B = x_batch.size(0)
    dtype = x_batch.type()
    x_batch = np.split(x_batch.detach().cpu().numpy(), B, axis=0)
    x_out = []
    for x in x_batch:
        x_out.append(cv2.GaussianBlur(x[0][0], kernel_sz, sigma))
    x_out = torch.tensor(np.stack(x_out)).type(dtype).unsqueeze(1)
    return x_out


def percentile_pt(x_batch, th=85):
    B = x_batch.size(0)
    dtype = x_batch.type()
    x_batch = np.split(x_batch.detach().cpu().numpy(), B, axis=0)
    x_out = []
    for x in x_batch:
        x_out.append(np.percentile(x, th))
    x_out = torch.tensor(x_out).type(dtype)
    return x_out


def depth_pose2flow_pt(depth, pose, K, K_inv):
    """ The intrinsic K and K_inv should match with the size of depth.
    :param depth:   B x H x W
    :param pose:    B x 4 x 4
    :param K:       B x 3 x 3
    :param K_inv:   3 x 3
    :return:        B x 2 x H x W
    """
    # depth to camera coordinates
    B, H, W = depth.size()

    grid = mesh_grid(B, H, W).type_as(depth)  # Bx2xHxW
    ones = torch.ones(B, 1, H, W).type_as(depth)
    homogeneous_2d = torch.cat((grid, ones), dim=1).reshape(B, 3, -1)  # [B, 3, H*W]
    d = depth.unsqueeze(1)
    points_3d = d * (K_inv @ homogeneous_2d).reshape(B, 3, H, W)  # [B, 3, H, W]

    # camera coordinates to pixel coordinates
    homogeneous_3d = torch.cat((points_3d, ones), dim=1).reshape(B, 4, -1)  # [B, 4, H*W]
    points_2d = K @ (pose @ homogeneous_3d)[:, :3]  # [B, 3, H*W]
    points_2d = points_2d.reshape(B, 3, H, W)
    points_2d = points_2d[:, :2] / points_2d[:, 2].clamp(min=1e-3).unsqueeze(
        1)  # [B, 2, H, W]
    flow = points_2d - grid

    return flow
