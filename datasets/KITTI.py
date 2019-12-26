from abc import abstractmethod, ABCMeta

import imageio
import numpy as np
from path import Path
from torch.utils.data import Dataset

from utils.flow_utils import load_flow
from utils.depth_utils import read_calib_file, get_focal_length_baseline


class ListDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, input_transform=None, target_transform=None,
                 co_transform=None):
        self.root = Path(root)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.samples = self.collect_samples()

    @abstractmethod
    def collect_samples(self):
        ...

    def __len__(self):
        return len(self.samples)

    def load_sample(self, s):
        img1_path, img2_path = s['img1'], s['img2']
        img1, img2 = map(imageio.imread, (self.root / img1_path, self.root / img2_path))

        sample = {'inputs': [img1.astype(np.float32), img2.astype(np.float32)]}
        if 'flow' in s:
            sample['target'] = load_flow(self.root / s['flow'])
        return sample


def rescale_intrinsics(K, im_size, raw_size):
    K_scaled = K.copy()
    K_scaled[0] = K[0] * im_size[1] / raw_size[1]
    K_scaled[1] = K[1] * im_size[0] / raw_size[0]
    K_scaled_inv = np.linalg.inv(K_scaled)
    return K_scaled, K_scaled_inv


def get_pyramid_intrinsics(K, raw_size, base_size, n_scale):
    base_size = np.array(base_size)
    pyramid_K = []
    pyramid_K_inv = []
    for i in range(n_scale):
        scale_size = base_size / 2 ** i
        K_scaled, K_scaled_inv = rescale_intrinsics(K, scale_size, raw_size)
        pyramid_K.append(K_scaled)
        pyramid_K_inv.append(K_scaled_inv)
    return pyramid_K, pyramid_K_inv


class KITTIRawFile(ListDataset):
    def __init__(self, root, sp_file, with_stereo=False,
                 transform=None, target_transform=None, co_transform=None):
        self.sp_file = sp_file
        self.with_stereo = with_stereo
        super(KITTIRawFile, self).__init__(root,
                                           transform, target_transform, co_transform)

    def collect_samples(self):
        samples = []
        with open(self.sp_file, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                s = {'img1': sp[0], 'img2': sp[2], 'img1r': sp[1], 'img2r': sp[3]}
                samples.append(s)
            return samples

    def __getitem__(self, idx):
        s = self.samples[idx]
        imgs = [s['img1'], s['img2']]
        imgs += [s['img1r'], s['img2r']]
        inputs = [imageio.imread(self.root / p).astype(np.float32) for p in imgs]

        raw_size = inputs[0].shape[:2]

        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, None)
        if self.input_transform is not None:
            inputs = [self.input_transform(i) for i in inputs]

        calib_dir = self.root / Path(s['img1']).split(sep='/')[0]
        cam2cam = read_calib_file(calib_dir / 'calib_cam_to_cam.txt')
        fl, bl = get_focal_length_baseline(cam2cam, cam=2)

        # hard code for multi-scale intrinsics
        K = cam2cam['P_rect_02'].reshape(3, 4)[:, :3]
        pyramid_K, pyramid_K_inv = get_pyramid_intrinsics(K, raw_size, (256, 832), 4)

        data = {
            'img1': inputs[0],
            'img2': inputs[1],
            'im_shape': raw_size,
            'calib_dir': self.root / Path(s['img1']).split(sep='/')[0],
            'fl_bl': fl * bl,
            'pyramid_K': pyramid_K,
            'pyramid_K_inv': pyramid_K_inv,
        }

        if self.with_stereo:
            data.update({
                'img1r': inputs[2],
                'img2r': inputs[3],
            })

        return data


class KITTIFlow(ListDataset):
    """
    This dataset is used for validation only, so all files about target are stored as
    file filepath and there is no transform about target.
    """

    def __init__(self, root, with_stereo=False, transform=None):
        self.with_stereo = with_stereo
        super(KITTIFlow, self).__init__(root, transform, None, None)

    def __getitem__(self, idx):
        s = self.samples[idx]
        imgs = [s['img1'], s['img2']]
        if self.with_stereo:
            imgs += [s['img1r'], s['img2r']]
        inputs = list(map(lambda p: imageio.imread(self.root / p).astype(np.float32),
                          imgs))
        raw_size = inputs[0].shape[:2]

        data = {
            'flow_occ': self.root / s['flow_occ'],
            'flow_noc': self.root / s['flow_noc'],
            'disp_occ': self.root / s['disp_occ'],
            'disp_noc': self.root / s['disp_noc'],
        }

        calib_dir_name = 'calib_cam_to_cam'
        cam2cam = read_calib_file(
            self.root / calib_dir_name / s['img1'].split('/')[-1][:-7] + '.txt')

        if 'P_rect_02' not in cam2cam:
            cam2cam['P_rect_02'] = cam2cam['P2']
            cam2cam['P_rect_03'] = cam2cam['P3']
        fl, bl = get_focal_length_baseline(cam2cam, cam=2)
        K = cam2cam['P_rect_02'].reshape(3, 4)[:, :3].astype(np.float32)
        pyramid_K, pyramid_K_inv = get_pyramid_intrinsics(K, raw_size, (256, 832), 4)

        if self.input_transform is not None:
            inputs = [self.input_transform(i) for i in inputs]
        data.update({
            'img1': inputs[0],
            'img2': inputs[1],
            'im_shape': raw_size,
            'fl_bl': fl * bl,
            'pyramid_K': pyramid_K,
            'pyramid_K_inv': pyramid_K_inv,
            'img1_path': self.root / s['img1'],
        })
        if self.with_stereo:
            data.update({
                'img1r': inputs[2],
                'img2r': inputs[3]
            })
        return data

    def collect_samples(self):
        '''Will search in training folder for folders 'flow_noc' or 'flow_occ'
               and 'image_2' (KITTI 2015) '''
        flow_occ_dir = 'flow_' + 'occ'
        flow_noc_dir = 'flow_' + 'noc'
        assert (self.root / flow_occ_dir).isdir()

        img_dir = 'image_2'
        assert (self.root / img_dir).isdir()

        if self.with_stereo:
            img_r_dir = 'image_3'
            assert (self.root / img_r_dir).isdir()

        disp_occ_dir = 'disp_occ_0'
        disp_noc_dir = 'disp_noc_0'

        assert (self.root / disp_occ_dir).isdir() and (
                self.root / disp_noc_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]

            flow_occ_map = flow_occ_dir + '/' + flow_map
            flow_noc_map = flow_noc_dir + '/' + flow_map
            img1 = img_dir + '/' + root_filename + '_10.png'
            img2 = img_dir + '/' + root_filename + '_11.png'
            assert (self.root / img1).isfile() and (self.root / img2).isfile()
            s = {'img1': img1, 'img2': img2,
                 'flow_occ': flow_occ_map, 'flow_noc': flow_noc_map, }

            if self.with_stereo:
                img1r = img_r_dir + '/' + root_filename + '_10.png'
                img2r = img_r_dir + '/' + root_filename + '_11.png'
                assert (self.root / img1r).isfile() and (self.root / img2r).isfile()
                s.update({'img1r': img1r, 'img2r': img2r})

            disp_occ = disp_occ_dir + '/' + root_filename + '_10.png'
            disp_noc = disp_noc_dir + '/' + root_filename + '_10.png'
            assert (self.root / disp_occ).isfile() and (self.root / disp_noc).isfile()
            s.update({'disp_occ': disp_occ, 'disp_noc': disp_noc})
            samples.append(s)
        return samples
