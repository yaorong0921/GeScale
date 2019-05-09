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
import torch.nn as nn

class MLPmodule(torch.nn.Module):
    """
    This is the 2-layer MLP implementation used for linking spatio-temporal
    features coming from different segments.
    """
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(MLPmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = 256
        self.classifier = nn.Sequential(
                                       nn.ReLU(),
                                       nn.Linear(self.num_frames * self.img_feature_dim,
                                                 self.num_bottleneck),
                                       #nn.Dropout(0.90), # Add an extra DO if necess.
                                       nn.ReLU(),
                                       nn.Linear(self.num_bottleneck,self.num_class),
                                       #nn.Softmax().cuda(),
                                       )
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input


def return_MLP(img_feature_dim, num_frames, num_class):
    MLPmodel = MLPmodule(img_feature_dim, num_frames, num_class)

    return MLPmodel
