import torch
import torch.nn as nn

def EPE(x, y):
    return torch.norm(x - y, 2, 1, keepdim=True)

def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def charbonnier(dist):
    return (dist ** 2 + 1e-6) ** 0.5 - 1e-3


def smooth_grad_1st(flo, image, alpha, mask, do_charb):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    if mask is not None:
        loss_x *= mask[:, :, :, 1:]
        loss_y *= mask[:, :, 1:, :]

    if do_charb:
        return charbonnier(loss_x).mean() / 2. + charbonnier(loss_y).mean() / 2.
    else:
        return loss_x.mean() / 2. + loss_y.mean() / 2.


def smooth_grad_2nd(flo, image, alpha, mask, do_charb):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    if mask is not None:
        loss_x *= mask[:, :, :, 1:-1]
        loss_y *= mask[:, :, 1:-1, :]

    if do_charb:
        return charbonnier(loss_x).mean() / 2. + charbonnier(loss_y).mean() / 2.
    else:
        return loss_x.mean() / 2. + loss_y.mean() / 2.
