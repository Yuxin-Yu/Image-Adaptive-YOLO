#! /usr/bin/env python
# coding=utf-8
import numpy as np
import torch
from torch import nn

#Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class ExtractParameters2(nn.Module):
    def __init__(self,output_dims,img_c,img_h,img_w):
        super(ExtractParameters2, self).__init__()
        print('extract_parameters_2 CNN:')
        channels = 16
        
        self.model = nn.Sequential()
        convolutional(self.model,(img_c,channels,3),name='ex_conv0',
            downsample=True, activate=True, bn=False)
        convolutional(self.model,(channels,2*channels,3),name='ex_conv1',
            downsample=True, activate=True, bn=False)
        convolutional(self.model,(2*channels,2*channels,3),name='ex_conv2',
            downsample=True, activate=True, bn=False)
        convolutional(self.model,(2*channels,2*channels,3),name='ex_conv3',
            downsample=True, activate=True, bn=False)
        convolutional(self.model,(2*channels,2*channels,3),name='ex_conv4',
            downsample=True, activate=True, bn=False)
        self.flatten = nn.Flatten()
        flatten_num = int(2*channels*img_h*img_w/np.power(2,10))
        self.linear1 = nn.Linear(flatten_num,64)
        self.linear2 = nn.Linear(64,output_dims)
    def forward(self, x): 
        print('    ', str(x.size()))
        x = self.model(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

#(batchsize, channel, height, width)
'''
filter_shape = (in_feature,out_feature,kernel_size)
'''
def convolutional(model, filters_shape,name, downsample=False, activate=True, bn=True):
    pad_w = (filters_shape[2] - 2) // 2 + 1
    if downsample: 
        model.add_module(name+'_conv',nn.Conv2d(filters_shape[0],filters_shape[1],filters_shape[2],stride=2,padding=pad_w))
    else:
        model.add_module(name+'_conv',nn.Conv2d(filters_shape[0],filters_shape[1],filters_shape[2],stride=1,padding=pad_w))
    if bn:
        model.add_module(name+'_bn',nn.BatchNorm2d(filters_shape[1],affine=True))
    
    if activate == True: model.add_module(name+'_relu',nn.LeakyReLU())

c,h,w = 3,256,256

model = ExtractParameters2(16,c,h,w).to(device)
print(model)


x = torch.rand(2,c,h,w)

pred = model(x.to(device))
print(pred)
