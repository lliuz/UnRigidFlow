## Declaration

This work has not been published yet. This repository will be transferred to the official account when the paper is accepted, and the paper information will be provided. To ensure anonymity, I will not accept any PR or issue for the time being.

Here are two sample results (~10MB gif for each) of our unsupervised models.

|         KITTI 15         |             Cityscapes             |
| :----------------------: | :--------------------------------: |
| ![kitti](demo/kitti.gif) | ![cityscapes](demo/cityscapes.gif) |

Basic training and testing are already included in this repository, so that you can use it to reproduce the results in the paper. 

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

> Note that if you are use PyTorch 1.0, you should make some changes, see [NVIDIA/flownet2-pytorch#98](https://github.com/NVIDIA/flownet2-pytorch/pull/98).
>
> Just replace `#include <torch/torch.h>` with `#include <torch/extension.h>` , adding  `#include <ATen/cuda/CUDAContext.h>` and then replacing all `at::globalContext().getCurrentCUDAStream()` with `at::cuda::getCurrentCUDAStream()`.

## Data Preparation

### KITTI

We are mainly focus on KITTI benchmark. You will need to download all of the [KITTI raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php) and calibration files to train the model. You will also need the training files of [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) with calibration files [[1]](http://www.cvlibs.net/download.php?file=data_stereo_flow_calib.zip), [[2]](http://www.cvlibs.net/download.php?file=data_scene_flow_calib.zip) for validating the models. 

### FlyingChairs

The unsupervised flow model can be also trained on [FlyingChairs]() dataset. The train / val splits are same as .

### Cityscapes

[Cityscapes](https://www.cityscapes-dataset.com/) can be used for pre-training to get better results(maybe, i have not tried yet). There is no ground truth in this dataset, so we use the hype-parameters from KITTI validation experiments. We discard the bottom 20% portion of each video frame, removing the very reflective car hoods from the input, as the setting in [MonoDepth](https://github.com/mrharicot/monodepth#cityscapes).

For the optical flow model, we used `leftImg8bit_sequence_trainvaltest.zip`. The original sequences are sampled at 17 Hz, we manually skip frames to generate ~10 Hz data like KITTI. 

For the depth model, we used `leftImg8bit_trainvaltest.zip`, `rightImg8bit_trainvaltest.zip`, `leftImg8bit_trainextra.zip` and `rightImg8bit_trainextra.zip`. (coming soon)

## Train and Validation

coming soon.

## Inference

coming soon.

## Models

coming soon.

## Acknowledgement

This repository refers some snippets from several great work, including [PWC-Net](https://github.com/NVlabs/PWC-Net), [monodepth](https://github.com/mrharicot/monodepth), [UnFlow](https://github.com/simonmeister/UnFlow), [UnDepthFlow](https://github.com/baidu-research/UnDepthflow), [DF-Net](https://github.com/vt-vl-lab/DF-Net). Although most of these are TensorFlow implementations, we are grateful for the sharing of these works, which save us a lot of time. We will gradually complete the references in our code.