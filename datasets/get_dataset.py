import copy
from torchvision import transforms
from datasets.KITTI import KITTIFlow, KITTIRawFile
from transforms import sep_transforms
from transforms.co_transforms import get_co_transforms

def get_dataset(all_cfg):
    cfg = all_cfg.data

    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])

    co_transform = get_co_transforms(aug_args=all_cfg.data_aug)

    if cfg.type == 'KITTI_15':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))
        train_set = KITTIRawFile(
            cfg.root,
            cfg.train_file,
            with_stereo=cfg.train_stereo,
            transform=train_input_transform,
            co_transform=co_transform  # no target here
        )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set = KITTIFlow(cfg.flow_data, with_stereo=cfg.test_stereo,
                              transform=valid_input_transform,
                              )
    else:
        raise NotImplementedError(cfg.type)
    return train_set, valid_set