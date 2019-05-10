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


import sys
import os
import torch
import torchvision
import torchvision.datasets as datasets

### please give the path of the GeScale_2D folder
ROOT_DATASET= ''


### To use SHGD and Jester, please give the path to the images to the "root_data".
### For SHGD15 and SHGD13, please give the path  (in func "return_SHGD")  until ../SHGD_Single
### For Jester, please give the path  (in func "return_jester") until ../20bn-jester-v1
### For SHGD Tuples, please give the path  (in func "return_SHGDTuples") until ../set1  or ../set2
 
def return_SHGD(modality):
    filename_categories = 'GeScale_2D/list/SHGD/SHGD15_category.txt'
    filename_imglist_train = 'GeScale_2D/list/SHGD/SHGD15_trainlist.txt'
    filename_imglist_val = 'GeScale_2D/list/SHGD/SHGD15_vallist.txt'
    if modality == 'IR' or modality == 'D' or modality == 'IRD':
        prefix = 'img_{}.png'
        root_data = '' ## plese give the path to SHGD
    else:
        print('no such modality:'+modality)
        sys.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_jester(modality):
    filename_categories = 'GeScale_2D/list/jester/category.txt'
    filename_imglist_train = 'GeScale_2D/list/jester/train_videofolder.txt'
    filename_imglist_val = 'GeScale_2D/list/jester/val_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ''  ## please give the path to jester
    else:
        print('no such modality:'+modality)
        sys.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_SHGDTuples(modality):
    filename_categories = 'GeScale_2D/list/SHGD/SHGD13_category.txt'
    filename_imglist_test = 'GeScale_2D/list/SHGD/test_list.txt'
    if modality == 'IRD' or modality == 'D' or modality == 'IRD':
        prefix = 'img_{}.png'
        root_data = ''  ## please give the path SHGD
    else:
        print('no such modality:'+modality)
        sys.exit()
    return filename_categories, filename_imglist_test, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester':return_jester, 'SHGD': return_SHGD, 'SHGDTuples': return_SHGDTuples}
    if dataset in dict_single:
        if dataset == 'SHGDTuples':
            file_categories, file_imglist_test, root_data, prefix = dict_single[dataset](modality)
            file_imglist_test = os.path.join(ROOT_DATASET, file_imglist_test)
        else:
            file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
            file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
            file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_categories = os.path.join(ROOT_DATASET, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    if dataset == 'SHGDTuples':
        return categories, file_imglist_test, root_data, prefix
    else:
        return categories, file_imglist_train, file_imglist_val, root_data, prefix
