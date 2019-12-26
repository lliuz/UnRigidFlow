from .flow_loss import unFlowLoss
from .depth_loss import MonodepthLoss
from .rigid_loss import RigidFlowLoss


def get_loss(cfg):
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    elif cfg.type == 'monodepth':
        loss = MonodepthLoss(n=cfg.n, SSIM_w=cfg.w_ssim, disp_gradient_w=cfg.w_disp_grad,
                             lr_w=cfg.w_lr)
    elif cfg.type == 'rigidflow':
        depth_loss = MonodepthLoss(n=cfg.n, SSIM_w=cfg.w_ssim,
                                   disp_gradient_w=cfg.w_disp_grad,
                                   lr_w=cfg.w_lr)
        rigid_flow_loss = RigidFlowLoss(cfg)
        loss = [depth_loss, rigid_flow_loss]
    else:
        raise NotImplementedError(cfg.type)
    return loss
