## UnRigidFlow

This is the official PyTorch implementation of [UnRigidFlow](https://www.ijcai.org/proceedings/2019/0123) (IJCAI2019).

Here are two sample results (~10MB gif for each) of our unsupervised models.

|         KITTI 15         |             Cityscapes             |
| :----------------------: | :--------------------------------: |
| ![kitti](demo/kitti.gif) | ![cityscapes](demo/cityscapes.gif) |

If you find this repo useful in your research, please consider citing:

```
@inproceedings{Liu:2019:unrigid, 
title = {Unsupervised Learning of Scene Flow Estimation Fusing with Local Rigidity}, 
author = {Liang Liu, Guangyao Zhai, Wenlong Ye, Yong Liu}, 
booktitle = {International Joint Conference on Artificial Intelligence, IJCAI}, 
year = {2019}
}
```

## Requirements

This codebase was developed and tested with Python 3.5, Pytorch>=0.4.1, OpenCV 3.4, CUDA 9.0 and Ubuntu 16.04.

Most of python packages can be install by

```sh
pip3 install -r requirements.txt
```

In addition, [Optimized correlation with CUDA kernel](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package) should be compiled manually with:

```
cd <correlation_package>
python3 setup.py install
```

and add `<correlation_package>` to `$PYTHONPATH`.

> Note that if you are use PyTorch >= 1.0, you should make some changes, see [NVIDIA/flownet2-pytorch#98](https://github.com/NVIDIA/flownet2-pytorch/pull/98).
>
> Just replace `#include <torch/torch.h>` with `#include <torch/extension.h>` , adding  `#include <ATen/cuda/CUDAContext.h>` and then replacing all `at::globalContext().getCurrentCUDAStream()` with `at::cuda::getCurrentCUDAStream()`.

## Training the models

We are mainly focus on KITTI benchmark. You will need to download all of the [KITTI raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php) and calibration files to train the model. You will also need the training files of [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) with calibration files [[1]](http://www.cvlibs.net/download.php?file=data_stereo_flow_calib.zip), [[2]](http://www.cvlibs.net/download.php?file=data_scene_flow_calib.zip) for validating the models. 

The complete training contains 3 steps:

1. Train the flow model separately:

   ```
   python3 train.py -c configs/KITTI_flow.json
   ```

2. Train the depth model separately:

   ```
   python3 train.py -c configs/KITTI_depth_stereo.json
   ```

3. Train the flow and depth models jointly:

   ```
   python3 train.py -c configs/KITTI_rigid_flow_stereo.json
   ```

## Pre-trained Models

You can download our [pre-trained models]()(coming soon), here are the model list and the performance:

- `KITTI_flow`: The separately trained optical flow network on KITTI raw data (from scratch), which reaches 6.50 EPE on the KITTI 15 training set.
- `KITTI_stereo_depth`: The stereo depth network on KITTI raw data.
- `KITTI_flow_joint`: The optical flow network jointly trained with stereo depth on KITTI raw data, which reaches 5.49 EPE on the KITTI 15 training set. It can reach 5.17 EPE on the KITTI 15 training set, and 11.66% F1 scores on KITTI 15 test set by fusing with rigidity segmentation.

## Acknowledgement

This repository refers some snippets from several great work, including [PWC-Net](https://github.com/NVlabs/PWC-Net), [monodepth](https://github.com/mrharicot/monodepth), [UnFlow](https://github.com/simonmeister/UnFlow), [UnDepthFlow](https://github.com/baidu-research/UnDepthflow), [DF-Net](https://github.com/vt-vl-lab/DF-Net). Although most of these are TensorFlow implementations, we are grateful for the sharing of these works, which save us a lot of time. 