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
  python main.py jester RGB
                 --arch squeezenet1_1 \
                 --num_segments 8 \
                 --batch-size 32 \
                 --img_feature_dim 64 \
                 
  ```

- Training with SHGD 15: using pretrained (with jester) models; inputs are IR and Depth images.

  ```
  python main.py SHGD IRD
                 --arch squeezenet1_1 \
                 --num_segments 8 \
                 --batch-size 32 \
                 --img_feature_dim 64 \
                 --pretrain_path pretrained_models/.. \
  ```

- Training with SHGD13: using pretrained (with SHGD15) models; inputs are IR and Depth images. 

  **Please change the annotations** (train list, validation list and label list) for SHGD13 in ``datasets_video.py`` first!

  ```
  python main.py SHGD IRD
                 --arch squeezenet1_1 \
                 --num_segments 8 \
                 --batch-size 32 \
                 --img_feature_dim 64 \
                 --pretrain_path pretrained_models/.. \
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
                 --resume_path pretrained_models/.. \
  ```

### Acknowledgment

Thank projects [TSN](<https://github.com/yjxiong/temporal-segment-networks>) and [MFFs](<https://github.com/okankop/MFF-pytorch>), this work is based on the framework of them. Thank [SqueezeNet version 1.1](<https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py>)  and [MobileNetV2](<https://github.com/d-li14/mobilenetv2.pytorch>) for pretrained models.
