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


from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant


import MLPmodule

class MFF(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='squeezenet1_1',
                 dropout=0.8,img_feature_dim=256, dataset='jester',
                 crop_num=1, partial_bn=True, print_spec=True):
        super(MFF, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.dropout = dropout
        self.dataset = dataset
        self.crop_num = crop_num
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame

        if print_spec == True:
            print(("""
    Initializing base model: {}.
    Configurations:
        input_modality:     {}
        num_segments:       {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class, base_model)

        if self.modality == 'IR' or self.modality == 'D':
            self.base_model = self._construct_onechannel_model(self.base_model)
        if self.modality == 'IRD':
            self.base_model = self._construct_twochannel_model(self.base_model)
        
        self.consensus = MLPmodule.return_MLP( self.img_feature_dim, self.num_segments, num_class)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class, base_model):
        if base_model == 'squeezenet1_1':
            last_Fire = getattr(self.base_model, self.base_model.last_layer_name)
            last_layer = getattr(last_Fire, 'expand3x3')
            feature_dim = last_layer.out_channels * 2 # Squeeze net concatenates two output from 3x3 and 1x1 kernel. So the output dimension should be doubled.
            self.base_model.add_module('AvgPooling', nn.AvgPool2d(13, stride=1))
            self.base_model.add_module('fc', nn.Linear(feature_dim,num_class))
            self.base_model.last_layer_name = 'fc'

        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            # set the MFFs feature dimension
            self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)


        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'squeezenet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            if base_model == 'squeezenet1_1':
                self.base_model = self.base_model.features
                self.base_model.last_layer_name = '12'

            else:
                self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]


        elif 'mobilenetv2' in base_model:
            import mobilenetv2
            self.base_model = mobilenetv2.mobilenetv2(input_size=224, width_mult=1.)
            self.base_model.load_state_dict(torch.load('pretrained_models/mobilenetv2-0c6065bc.pth'))
            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(MFF, self).train(mode)
        count = 0
        if self._enable_pbn:
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):

        if self.modality == 'RGB':
            n_chann = 3 
        elif self.modality == 'IR' or self.modality == 'D':
            n_chann = 1 
        else:
            n_chann = 2 

        base_out = self.base_model(input.view((-1, n_chann) + input.size()[-2:]))
        base_out = base_out.squeeze()
        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = self.consensus(base_out)
        return output.squeeze(1)

    def _construct_onechannel_model(self, base_model):

        modules = list(self.base_model.modules())
        filter_conv2d = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))
        first_conv_idx = next(filter_conv2d)
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
      
        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (1,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        print(new_kernel_size)
        
        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name
        
        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_twochannel_model(self, base_model):

        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        print(kernel_size)
        new_kernel_size = kernel_size[:1] + (2 , ) + kernel_size[2:]
        print(new_kernel_size)
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model


    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'IRD' or self.modality == 'D' or self.modality == 'IR':
            return torchvision.transforms.Compose([GroupMultiScaleResize(0.2),
                                                   GroupMultiScaleRotate(10),
                                                   #GroupSpatialElasticDisplacement(),
                                                   GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   #GroupRandomHorizontalFlip(is_flow=False)
                                                  ])

