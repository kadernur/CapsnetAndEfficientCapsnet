from param import CapsNetParam

import os
import torch
from torch import nn
from typing import List
from typing import Union

class Squash(nn.Module):
    def __init__(self, eps=1e-7, name="squash"):
        super().__init__()
        self.eps = eps

    def forward(self, input_vector):
        norm = torch.norm(input_vector, dim=-1, keepdim=True)
        coef = 1 - 1 / torch.exp(norm)
        unit = input_vector / (norm + self.eps)
        return coef * unit

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class FeatureMap(nn.Module):
   def __init__(self, param: CapsNetParam) -> None:
        super(FeatureMap, self).__init__()
        self.param = param
        self.conv1 = nn.Sequential(nn.Conv2d(
            in_channels = 1,
            out_channels = self.param.conv1_filter,
            kernel_size = self.param.conv1_kernel,
            stride = self.param.conv1_stride),
            nn.ReLU())
        self.norm1 = nn.BatchNorm2d(self.param.conv1_filter)

        self.conv2 = nn.Sequential(nn.Conv2d(
            in_channels = self.param.conv1_filter,
            out_channels = self.param.conv2_filter,
            kernel_size = self.param.conv2_kernel,
            stride = self.param.conv2_stride), 
            nn.ReLU())
        self.norm2 = nn.BatchNorm2d(self.param.conv2_filter) 

        self.conv3 = nn.Sequential(nn.Conv2d(
            in_channels= self.param.conv2_filter,
            out_channels = self.param.conv3_filter,
            kernel_size = self.param.conv3_kernel,
            stride = self.param.conv3_stride),
            nn.ReLU())
        self.norm3 = nn.BatchNorm2d(self.param.conv3_filter)

        self.conv4 = nn.Sequential(nn.Conv2d(
            in_channels = self.param.conv3_filter,
            out_channels = self.param.conv4_filter,
            kernel_size = self.param.conv4_kernel,
            stride = self.param.conv4_stride),
            nn.ReLU())
        self.norm4 = nn.BatchNorm2d(self.param.conv4_filter) 
        
        
   def forward(self, input_images: torch.Tensor) -> torch.Tensor:
       feature_maps = self.norm1(self.conv1(input_images))
       feature_maps = self.norm2(self.conv2(feature_maps))
       feature_maps = self.norm3(self.conv3(feature_maps))
       return self.norm4(self.conv4(feature_maps))
    
    
class PrimaryCap(nn.Module):
    def __init__(self, param: CapsNetParam) -> None:
        super(PrimaryCap, self).__init__()
        self.param = param
        self.dconv = nn.Sequential(nn.Conv2d(
            in_channels = self.param.conv4_filter,
            out_channels = self.param.dconv_filter,
            kernel_size = self.param.dconv_kernel,
            stride = self.param.dconv_stride,
            groups = self.param.dconv_filter), 
            nn.ReLU())
        self.squash = Squash()
        
    def forward(self, feature_maps):
        dconv_outputs = self.dconv(feature_maps)
        return self.squash(dconv_outputs.view(dconv_outputs.shape[0], -1, self.param.dim_primary_caps))
    

class DigitCap(nn.Module):
    def __init__(self, param: CapsNetParam) -> None:
        super(DigitCap, self).__init__()
        self.param = param
        self.attention_coef = 1 / torch.sqrt(torch.tensor(self.param.dim_primary_caps))
        self.W = nn.Parameter(torch.randn(self.param.num_digit_caps, self.param.num_primary_caps, 
                self.param.dim_digit_caps, self.param.dim_primary_caps))
        self.B = nn.Parameter(torch.randn(self.param.num_digit_caps, 1, self.param.num_primary_caps))
        self.squash = Squash()
        self.softmaxFunc = nn.Softmax(dim = -2)
    
    def forward(self, primary_caps):
        U = torch.unsqueeze(torch.tile(
                            torch.unsqueeze(primary_caps, axis = 1), [1, self.param.num_digit_caps, 1, 1]), 
                            axis = -1)
        U_hat = torch.squeeze(torch.matmul(self.W, U), axis = -1)
        A = self.attention_coef * torch.matmul(U_hat, U_hat.transpose(2, 3))
        C = self.softmaxFunc(torch.sum(A, axis = -2, keepdims = True))
        S = torch.squeeze(torch.matmul(C + self.B, U_hat), axis=-2)
        return self.squash(S)
   
    