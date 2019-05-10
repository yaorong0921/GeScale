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
import pandas as pd
import copy
import random


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
        return self._data[2:]

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


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



def video_loader(video_dir_path, n_frames, modality, image_loader):
    video = []

    for idx in range(0, n_frames):
        if modality == 'IR':
            image_gray_path = os.path.join(video_dir_path[0], 'img_{}.png'.format(idx))
            if os.path.exists(image_gray_path):
                video.append(image_loader(image_gray_path))
            else:
                print(image_gray_path + 'Images not found!')
                break

        elif modality == 'D':
            image_depth_path = os.path.join(video_dir_path[1], 'img_{}.png'.format(idx))
            if os.path.exists(image_depth_path):
                video.append(image_loader(image_depth_path))
            else:
                print(image_depth_path + 'Images not found!')
                break

        elif modality == 'IRD':
            image_gray_path = os.path.join(video_dir_path[0], 'img_{}.png'.format(idx))
            image_depth_path = os.path.join(video_dir_path[1], 'img_{}.png'.format(idx))
            if os.path.exists(image_gray_path):
                video.append(image_loader(image_gray_path))
                video.append(image_loader(image_depth_path))
            else:
                print(image_gray_path + 'Images not found!')
                break

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(annotation_path,filename,class_to_idx):
    # For SHGD Tuples, txt files includes all the labels only. 
    # The order is also the order for image folders.

#    data_file_path = os.path.join(annotation_path,filename)
#    tmp = [x.strip().split(' ') for x in open (data_file_path)]
#    tmp = [item for item in tmp if int(item[1])>=3]
#    video_list = [VideoRecord(item) for item in tmp]
#    print('video number:%d'%(len(video_list)))

#    return video_list
    data_file_path = os.path.join(annotation_path,filename)
    video_list = []

    data = pd.read_csv(data_file_path, delimiter=',', header=None)
    for i in range(0,data.shape[0]):
        row = data.iloc[i, :]
        class_name = []
        class_name.append(class_to_idx[row[0]])
        class_name.append(class_to_idx[row[1]])
        class_name.append(class_to_idx[row[2]])

        video_list.append(class_name)
    print('video number:%d'%(len(video_list)))

    return video_list



def load_label_dict():
    labels = ["Check", "Fist", "Flat Hand", "Hand Down", "Hand Up", "No Gesture", "Show Five Fingers", 
              "Show Two Fingers", "Stop Sign", "Thumb Left", "Thumb Right", "Thumb Up", "Zero"]

    class_labels_map = {}
    index = 0
    for class_label in labels:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

def make_dataset(root_path, annotation_path, filename):

    class_idx_map = load_label_dict()
    data = load_annotation_data(annotation_path, filename, class_idx_map)
    dataset = []
#    for i in range(len(data)):
#        if i % 1000 == 0:
#            print('dataset loading [{}/{}]'.format(i, len(data)))
#        video_path_gray = os.path.join(root_path, 'Grayscale', data[i].path)
#        video_path_depth = os.path.join(root_path, 'Depth', data[i].path)
#        if not os.path.exists(video_path_gray):
#            print('Video file id %s not found!'%data[i].path)
#            break

#        n_frames = data[i].num_frames
#        if n_frames <= 5:
#            print('Something wrong with the video id', video_names[i])

    for i in range(len(data)):
        if i % 100 == 0:
            print('dataset loading [{}/{}]'.format(i, len(data)))

        video_path_gray = os.path.join(root_path, 'Grayscale', str(i))
        video_path_depth = os.path.join(root_path, 'Depth', str(i))
        if not os.path.exists(video_path_depth):
            print(video_path_depth,'Video file not found!')
            break
        else:
            list = os.listdir(video_path_gray)  # dir is your directory path
            n_frames = len(list)

        label = data[i]
        #label_list = label.split(',')
        #labels = [int(label_item) for label_item in label_list]

        sample = {
            'video_gray': video_path_gray,
            'video_depth': video_path_depth,
            'n_frames': n_frames,
            'video_id': i,
            'label' : label
        }

        dataset.append(sample)

    return dataset


class SHGD_Tuples(data.Dataset):
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
        imgs (list): List of (image path, class_index, n_frames) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 filename,
                 modality,
                 spatial_transform=None,
                 target_transform=None,
                 sample_duration = 8,
                 get_loader=get_default_video_loader):
        self.data = make_dataset(root_path, annotation_path, filename)

        self.spatial_transform = spatial_transform
        self.target_transform = target_transform
        self.sample_duration = sample_duration
        self.loader = get_loader()
        self.modality = modality

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path_gray = self.data[index]['video_gray']
        path_depth = self.data[index]['video_depth']
        path = [path_gray,path_depth]
        n_frames = self.data[index]['n_frames']
        target = self.data[index]['label']

        clip = self.loader(path, n_frames, self.modality)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((n_frames, -1) + im_dim).permute(1, 0, 2, 3)

        return clip, target, n_frames

    def __len__(self):
        return len(self.data)
