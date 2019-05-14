# Scaled Hand Gesture Recognition with CNNs: 2D models

### Requirements

- Python 3.6
- Pytorch 0.4 or later versions.
- Other packages e.g. numpy, etc.

### Overview 

This project includes training using two models [SqueezeNet version 1.1](<https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py>)  and [MobileNetV2](<https://github.com/d-li14/mobilenetv2.pytorch>) , which are already pretrained with ImageNet. 

You can pretrain SqueezeNet 1.1 and MobileNetV2 with Jester, fine tune using SHGD15/13 and test models on SHGD Tuples.

### Dataset 

- [Jester v1](https://20bn.com/datasets/jester) for pretraining. Please follow the download instructions. 
- SHGD15/SHGD13 for training and validation. 
- SHGD Tuples for testing. 

### Training

**Before running the code, please give the paths to the datasets and their annotations in** ``datasets_video.py`` !

Here are some examples for (pre)training with Jester and fine-tuning with SHGD15/SHGD13 using SqueezeNet 1.1. 

- Pretraining with Jester: using SqueezeNet 1.1

  ```
  python main.py --dataset SHGD \
                --modality IRD  \
                --arch squeezenet1_1 \
                --num_segments 8 \
                --batch-size 32 \
                --epochs 60 \
                --img_feature_dim 64 \
                --lr 0.001 \
                --dropout 0.8 \
                 
  ```

- Training with SHGD 15: using pretrained (with jester) models; inputs are IR and Depth images.

  ```
  python main.py --dataset SHGD \
                 --modality IRD  \
                 --arch squeezenet1_1 \
                 --num_segments 8 \
                 --batch-size 32 \
                 --epochs 60 \
                 --img_feature_dim 64 \
                 --lr 0.001 \
                 --dropout 0.8 \
                 --pretrained './pretrained_models/jester_RGB_squeezenet1_1_segment8_best.pth.tar' \
  ```

- Training with SHGD13: using pretrained (with SHGD15) models; inputs are IR and Depth images. 

  **Please change the annotations** (train list, validation list and label list) for SHGD13 in ``datasets_video.py`` first!

  ```
  python main.py --dataset SHGD \
                 --modality IRD  \
                 --arch squeezenet1_1 \
                 --num_segments 8 \
                 --img_feature_dim 64 \
                 --batch-size 32 \
                 --epochs 60 \
                 --lr 0.001 \
                 --dropout 0.8 \
                 --pretrained './pretrained_models/SHGD_IRD_squeezenet1_1_segment8_best.pth.tar' \
  ```

### Test

For testing to recognize gesture tuples in SHGD Tuples, Please use the models trained with SHGD13. 

- Test with SHGD Tuples: using MobileNetV2; inputs are IR images only.

  ```
  python main.py SHGD IR
                 --arch mobilenetv2 \
                 --num_segments 8 \
                 --img_feature_dim 64 \
                 --test \
                 --resume './pretrained_models/SHGD_IR_mobilenetv2_segment8_best.pth.tar' \
  ```

### Acknowledgment

Thank projects [TSN](<https://github.com/yjxiong/temporal-segment-networks>) and [MFFs](<https://github.com/okankop/MFF-pytorch>), this work is based on the framework of them. Thank [SqueezeNet version 1.1](<https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py>)  and [MobileNetV2](<https://github.com/d-li14/mobilenetv2.pytorch>) for pretrained models.
