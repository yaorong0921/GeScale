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

from datasets.jester import Jester
from datasets.SHGD import SHGD
from datasets.SHGD_Tuples import SHGD_Tuples

def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['jester', 'SHGD']

    if opt.dataset == 'jester':
        training_data = Jester(
            opt.video_path,
            opt.annotation_path,
            opt.train_list,
            opt.modality,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)

    else :
        training_data = SHGD(
            opt.video_path,
            opt.annotation_path,
            opt.train_list,
            opt.modality,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['jester', 'SHGD']

    if opt.dataset == 'jester':
        validation_data = Jester(
            opt.video_path,
            opt.annotation_path,
            opt.val_list,
            opt.modality,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    else: 
        validation_data = SHGD(
            opt.video_path,
            opt.annotation_path,
            opt.val_list,
            opt.modality,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_data


def get_test_set(opt, spatial_transform, target_transform):
    assert opt.dataset in ['SHGD']

    test_data = SHGD_Tuples(
            opt.video_path,
            opt.annotation_path,
            opt.test_list,
            opt.modality,
            spatial_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return test_data
