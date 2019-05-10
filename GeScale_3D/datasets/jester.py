'''
/* ===========================================================================
** Copyright (C) 2019 Infineon Technologies AG. All rights reserved.
** ===========================================================================
**
** ===========================================================================
** Infineon Technologies AG (INFINEON) is supplying this file for use
** exclusively with Infineon's sensor products. This file can be freely
** distributed within development tools and software supporting such 
** products.
** 
** THIS SOFTWARE IS PROVIDED "AS IS".  NO WARRANTIES, WHETHER EXPRESS, IMPLIED
** OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
** MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
** INFINEON SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR DIRECT, INDIRECT, 
** INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, FOR ANY REASON 
** WHATSOEVER.
** ===========================================================================
*/
'''
import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random

from utils import load_value_file

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, sample_duration, image_loader):
    video = []


    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(annotation_path,filename):
    # check the frame number is large >3:
    # For Jester [video_id, num_frames, class_idx]
    data_file_path = os.path.join(annotation_path,filename)
    tmp = [x.strip().split(' ') for x in open (data_file_path)]
    tmp = [item for item in tmp if int(item[1])>=3]
    video_list = [VideoRecord(item) for item in tmp]
    print('video number:%d'%(len(video_list)))
    
    return video_list


def make_dataset(root_path, annotation_path, filename, sample_duration):
    data = load_annotation_data(annotation_path, filename)

    dataset = []
    for i in range(len(data)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(data)))

        video_path = os.path.join(root_path, data[i].path)
        if not os.path.exists(video_path):
            print(video_path,'Video file not found!')
            break

        n_frames = data[i].num_frames
        if n_frames <= 5:
            print('Something wrong with the video id', video_names[i])
            break

        begin_t = 1
        end_t = n_frames
        sample = {
            'video_path': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': data[i].path,
            'label': data[i].label
        }

        sample['frame_indices'] = list(range(begin_t, end_t + 1))
        dataset.append(sample)

    return dataset


class Jester(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 filename,
                 modality,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=8,
                 get_loader=get_default_video_loader):
        self.data = make_dataset(root_path, annotation_path, filename, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.sample_duration = sample_duration
        self.loader = get_loader()
        self.modality = modality

    def __getitem__(self, index):
        """f
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video_path']
        frame_indices = self.data[index]['frame_indices']

        ### for pretraining, always using center segment
        if self.temporal_transform is not None:
           frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.sample_duration)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target

    def __len__(self):
        return len(self.data)
