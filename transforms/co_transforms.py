import torch
import random
import numbers
import numpy as np
import scipy.ndimage as ndimage
from scipy.misc import imresize


def get_co_transforms(aug_args):
    transforms = []
    if aug_args.trans:
        transforms.append(RandomTranslate(10.))
    if aug_args.rotate:
        transforms.append(RandomRotate(10, 5))
    if aug_args.crop:
        transforms.append(RandomCrop(aug_args.para_crop))
    if aug_args.vflip:
        transforms.append(RandomVerticalFlip())
    if aug_args.hflip:
        transforms.append(RandomHorizontalFlip())
    if aug_args.swap:
        transforms.append(RandomSwap())
    return Compose(transforms)


class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class Zoom(object):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, image):
        h, w, _ = image.shape
        if h == self.new_h and w == self.new_w:
            return image
        return imresize(image, (self.new_h, self.new_w))


class RandomSwap(object):
    def __call__(self, inputs, target):
        # Only for unsupervised training
        assert target is None
        if random.random() < 0.5:
            if len(inputs) == 2:
                inputs = inputs[::-1]  # return img2 and img1
            elif len(inputs) == 4:
                inputs = [inputs[i] for i in [1, 0, 3, 2]]  # return 2l, 1l, 2r, 1r
            else:
                raise ValueError(len(inputs))
        return inputs, None


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        # Only for unsupervised training
        assert target is None
        if random.random() < 0.5:
            if len(inputs) == 4:
                img1_l = np.copy(np.fliplr(inputs[2]))  # flip img1_r to img1
                img2_l = np.copy(np.fliplr(inputs[3]))  # filp img2_r to img2
                img1_r = np.copy(np.fliplr(inputs[0]))
                img2_r = np.copy(np.fliplr(inputs[1]))
                inputs = [img1_l, img2_l, img1_r, img2_r]
            else:
                inputs = [np.copy(np.fliplr(im)) for im in inputs]
        return inputs, target


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.flipud(inputs[0]))
            inputs[1] = np.copy(np.flipud(inputs[1]))
            if target is not None:
                target = np.copy(np.flipud(target))
                target[:, :, 1] *= -1
        return inputs, target


class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1, x2, x3, x4 = max(0, tw), min(w + tw, w), max(0, -tw), min(w - tw, w)
        y1, y2, y3, y4 = max(0, th), min(h + th, h), max(0, -th), min(h - th, h)

        inputs[0] = inputs[0][y1:y2, x1:x2]
        inputs[1] = inputs[1][y3:y4, x3:x4]
        if target is not None:
            target = target[y1:y2, x1:x2]
            target[:, :, 0] += tw
            target[:, :, 1] += th

        return inputs, target


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, inputs, target):
        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2
        angle2 = applied_angle + diff / 2
        angle1_rad = angle1 * np.pi / 180

        h, w, _ = inputs[0].shape

        def rotate_flow(i, j, k):
            return -k * (j - w / 2) * (diff * np.pi / 180) + (1 - k) * (i - h / 2) * (
                    diff * np.pi / 180)

        rotate_flow_map = np.fromfunction(rotate_flow, (h, w, 2))

        inputs[0] = ndimage.interpolation.rotate(inputs[0], angle1, reshape=self.reshape,
                                                 order=self.order)
        inputs[1] = ndimage.interpolation.rotate(inputs[1], angle2, reshape=self.reshape,
                                                 order=self.order)

        if target is not None:
            target[:, :, :2] += rotate_flow_map
            target = ndimage.interpolation.rotate(target, angle1, reshape=self.reshape,
                                                  order=self.order)
            # flow vectors must be rotated too! careful about Y flow which is upside down
            target_ = np.copy(target)
            target[:, :, 0] = np.cos(angle1_rad) * target_[:, :, 0] + np.sin(
                angle1_rad) * target_[:, :, 1]
            target[:, :, 1] = -np.sin(angle1_rad) * target_[:, :, 0] + np.cos(
                angle1_rad) * target_[:, :, 1]
        return inputs, target


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs[0] = inputs[0][y1: y1 + th, x1: x1 + tw]
        inputs[1] = inputs[1][y1: y1 + th, x1: x1 + tw]
        if target is not None:
            return inputs, target[y1: y1 + th, x1: x1 + tw]
        else:
            return inputs, None
