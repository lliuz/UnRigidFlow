from trainer import kitti_flow_trainer, kitti_depth_trainer, kitti_rigid_trainer


def get_trainer(name):
    if name == 'KITTI_flow':
        TrainFramework = kitti_flow_trainer.TrainFramework
    elif name == 'KITTI_depth':
        TrainFramework = kitti_depth_trainer.TrainFramework
    elif name == 'KITTI_rigid_flow':
        TrainFramework = kitti_rigid_trainer.TrainFramework
    else:
        raise NotImplementedError(name)

    return TrainFramework
