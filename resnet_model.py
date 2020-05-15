#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/5/13
import torch.nn as nn
import torchvision.models as models
import torch

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def SIMnetFromResnet(net_name='resnet34'):
    r"""

    :param net_name: This is a name for choosing differnet resnet
    :return: SIMnet   output channel is 3 :  wavevector_kx wavevector_ky and initial phase of illumination patteen

    """
    if net_name == 'resnet34':
        pre_net = models.resnet34()
    elif net_name == 'resnet50':
        pre_net = models.resnet50()
    elif net_name == 'resnet101':
        pre_net = models.resnet101()
    resnet_layer = nn.Sequential(*list(pre_net.children())[:-1])
    test_input = torch.rand(1, 3, 256, 256)
    test_output = resnet_layer(test_input)
    output_channel = test_output.shape[1]
    fc_layer = [
        nn.Linear(in_features=output_channel, out_features=128, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128, out_features=3, bias=True)]

    return nn.Sequential(*resnet_layer, Flatten(), *fc_layer)


if __name__ == '__main__':
    SIMnet=SIMnetFromResnet(net_name='resnet50')


