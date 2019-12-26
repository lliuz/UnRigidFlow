from models.pwcnet import PWCFlow, PWCStereo


def get_model(cfg):
    if cfg.type == 'pwcflow':
        model = PWCFlow(lv_chs=cfg.lv_chs, n_out=cfg.n_out, bn=cfg.bn)
    elif cfg.type == 'pwcdisp':
        model = PWCStereo(lv_chs=cfg.lv_chs, n_out=cfg.n_out, bn=cfg.bn)
    elif cfg.type == 'rigidflow':
        flow_net = PWCFlow(lv_chs=cfg.lv_chs, n_out=cfg.n_out, bn=cfg.bn)
        depth_net = PWCStereo(lv_chs=cfg.lv_chs, n_out=cfg.n_out, bn=cfg.bn)
        model = [flow_net, depth_net]
    else:
        raise NotImplementedError(cfg.type)
    return model
