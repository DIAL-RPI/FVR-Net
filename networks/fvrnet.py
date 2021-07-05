import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import time
import sys
import os
import tools
from copy import copy
import numpy as np

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)

        # if stride == 2:
        #     stride = (1, 2, 2)

        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        # print('stride {}'.format(stride))
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class mynet(nn.Module):

#     def __init__(self, ):
#         self.inplanes = 64
#         super(mynet, self).__init__()
#         self.conv1_vol = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
#         self.conv2_vol = nn.Conv3d(64, 64, kernel_size=5, stride=(2, 1, 1), padding=(2, 2, 2), bias=False)
#         self.conv3_vol = nn.Conv3d(64, 64, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=False)
#         self.conv4_vol = nn.Conv3d(64, 64, kernel_size=3, stride=(3, 1, 1), padding=(1, 1, 1), bias=False)
#         self.bn1_vol = nn.BatchNorm3d(64)
#         self.bn2_vol = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
#         # self.conv_pool = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

#         self.conv1_frame = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
#         self.conv2_frame = nn.Conv3d(64, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

#         self.layer1 = self._make_layer(
#             ResNeXtBottleneck, 128, 3, shortcut_type='B', cardinality=32, stride=2)
#         self.layer2 = self._make_layer(
#             ResNeXtBottleneck, 256, 4, shortcut_type='B', cardinality=32, stride=2)
#         self.layer3 = self._make_layer(
#             ResNeXtBottleneck, 512, 6, shortcut_type='B', cardinality=32, stride=2)
#         self.layer4 = self._make_layer(
#             ResNeXtBottleneck, 1024, 3, shortcut_type='B', cardinality=32, stride=2)
        
#         self.avgpool = nn.AvgPool3d((1, 5, 8), stride=1)

#         self.fc1 = nn.Linear(2048, 16)

#     def _make_layer(self,
#                     block,
#                     planes,
#                     blocks,
#                     shortcut_type,
#                     cardinality,
#                     stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))

#         layers = []
#         layers.append(
#             block(self.inplanes, planes, cardinality, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, cardinality))

#         return nn.Sequential(*layers)

#     def volBranch(self, vol):
#         # print('\n********* Vol *********')
#         # print('vol {}'.format(vol.shape))
#         vol = self.conv1_vol(vol)
#         vol = self.bn1_vol(vol)
#         vol = self.relu(vol)
#         # print('conv1 {}'.format(vol.shape))


#         vol = self.conv2_vol(vol)
#         # print('conv2 {}'.format(vol.shape))
#         vol = self.bn2_vol(vol)
#         vol = self.relu(vol)

#         vol = self.conv3_vol(vol)
#         # print('conv3 {}'.format(vol.shape))

#         vol = self.conv4_vol(vol)
#         # print('conv3 {}'.format(vol.shape))
#         return vol
    
#     def frameBranch(self, frame):
#         # print('\n********* Frame *********')
#         # print('frame {}'.format(frame.shape))
#         frame = self.conv1_frame(frame)
#         frame = self.bn2_vol(frame)

#         # print('conv1 {}'.format(frame.shape))
#         frame = self.conv2_frame(frame)
#         # print('conv2 {}\n'.format(frame.shape))
#         return frame

#     def forward(self, vol, frame, device=None):
#         input_vol = vol.clone()

#         show_size = False
#         # show_size = True
#         if show_size:
#             vol = self.volBranch(vol)
#             frame = self.frameBranch(frame)

#             x = torch.cat((vol, frame), 2)
#             print('cat {}'.format(x.shape))

#             x = self.layer1(x)
#             print('layer1 {}'.format(x.shape))

#             x = self.layer2(x)
#             print('layer2 {}'.format(x.shape))

#             x = self.layer3(x)
#             print('layer3 {}'.format(x.shape))

#             x = self.layer4(x)
#             print('layer4 {}'.format(x.shape))

#             x = self.avgpool(x)
#             print('avgpool {}'.format(x.shape))

#             x = x.view(x.size(0), -1)
#             print('view {}'.format(x.shape))

#             x = self.fc1(x)
#             print('fc1 {}'.format(x.shape))
#             mat_out = x.clone()

#             x = torch.reshape(x, (x.shape[0], 4, 4))
#             print('reshape {}'.format(x.shape))
#             print('input_vol {}'.format(input_vol.shape))

#             grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=x, 
#                                        input_spacing=(1, 1, 1), device=device)
                                       
#             # grid = grid.to(device)
#             print('grid {}'.format(grid.shape))
#             vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)
#             print('resample {}'.format(vol_resampled.shape))
#             print('mat_out {}'.format(mat_out.shape))

#             # time.sleep(30)
#             sys.exit()
#         else:
#             vol = self.volBranch(vol)
#             frame = self.frameBranch(frame)

#             x = torch.cat((vol, frame), 2)

#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#             x = self.layer4(x)

#             x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#             x = self.fc1(x)

#             mat_out = x.clone()

#             x = torch.reshape(x, (x.shape[0], 4, 4))
#             grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=x, 
#                                        input_spacing=(1, 1, 1), device=device)
#             # grid = torch.affine_grid_generator(theta=)
#             # grid = grid.to(device)
#             vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)

            
#         return vol_resampled, mat_out
    

# class mynet2(nn.Module):

#     def __init__(self, ):
#         self.inplanes = 64
#         super(mynet2, self).__init__()
#         self.conv1_vol = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
#         self.conv2_vol = nn.Conv3d(64, 64, kernel_size=5, stride=(2, 1, 1), padding=(2, 2, 2), bias=False)
#         self.conv3_vol = nn.Conv3d(64, 64, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=False)
#         self.conv4_vol = nn.Conv3d(64, 64, kernel_size=3, stride=(3, 1, 1), padding=(1, 1, 1), bias=False)
#         self.bn1_vol = nn.BatchNorm3d(64)
#         self.bn2_vol = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
#         # self.conv_pool = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

#         self.conv1_frame = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
#         self.conv2_frame = nn.Conv3d(64, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

#         self.layer1 = self._make_layer(
#             ResNeXtBottleneck, 128, 3, shortcut_type='B', cardinality=32, stride=2)
#         self.layer2 = self._make_layer(
#             ResNeXtBottleneck, 256, 4, shortcut_type='B', cardinality=32, stride=2)
#         self.layer3 = self._make_layer(
#             ResNeXtBottleneck, 512, 6, shortcut_type='B', cardinality=32, stride=2)
#         self.layer4 = self._make_layer(
#             ResNeXtBottleneck, 1024, 3, shortcut_type='B', cardinality=32, stride=2)
        
#         self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)

#         self.fc1 = nn.Linear(2048, 16)

#     def _make_layer(self,
#                     block,
#                     planes,
#                     blocks,
#                     shortcut_type,
#                     cardinality,
#                     stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))

#         layers = []
#         layers.append(
#             block(self.inplanes, planes, cardinality, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, cardinality))

#         return nn.Sequential(*layers)

#     def volBranch(self, vol):
#         # print('\n********* Vol *********')
#         # print('vol {}'.format(vol.shape))
#         vol = self.conv1_vol(vol)
#         vol = self.bn1_vol(vol)
#         vol = self.relu(vol)
#         # print('conv1 {}'.format(vol.shape))


#         vol = self.conv2_vol(vol)
#         # print('conv2 {}'.format(vol.shape))
#         vol = self.bn2_vol(vol)
#         vol = self.relu(vol)

#         vol = self.conv3_vol(vol)
#         # print('conv3 {}'.format(vol.shape))

#         vol = self.conv4_vol(vol)
#         # print('conv3 {}'.format(vol.shape))
#         return vol
    
#     def frameBranch(self, frame):
#         # print('\n********* Frame *********')
#         # print('frame {}'.format(frame.shape))
#         frame = self.conv1_frame(frame)
#         frame = self.bn2_vol(frame)

#         # print('conv1 {}'.format(frame.shape))
#         frame = self.conv2_frame(frame)
#         # print('conv2 {}\n'.format(frame.shape))
#         return frame

#     def forward(self, vol, frame, device=None):
#         input_vol = vol.clone()

#         show_size = False
#         # show_size = True
#         if show_size:
#             vol = self.volBranch(vol)
#             frame = self.frameBranch(frame)

#             x = torch.cat((vol, frame), 2)
#             print('cat {}'.format(x.shape))

#             x = self.layer1(x)
#             print('layer1 {}'.format(x.shape))

#             x = self.layer2(x)
#             print('layer2 {}'.format(x.shape))

#             x = self.layer3(x)
#             print('layer3 {}'.format(x.shape))

#             x = self.layer4(x)
#             print('layer4 {}'.format(x.shape))

#             x = self.avgpool(x)
#             print('avgpool {}'.format(x.shape))

#             x = x.view(x.size(0), -1)
#             print('view {}'.format(x.shape))

#             x = self.fc1(x)
#             print('fc1 {}'.format(x.shape))
#             mat_out = x.clone()

#             x = torch.reshape(x, (x.shape[0], 4, 4))
#             print('reshape {}'.format(x.shape))
#             print('input_vol {}'.format(input_vol.shape))

#             grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=x, 
#                                        input_spacing=(1, 1, 1), device=device)
#             # grid = grid.to(device)
#             print('grid {}'.format(grid.shape))
#             vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)
#             print('resample {}'.format(vol_resampled.shape))
#             print('mat_out {}'.format(mat_out.shape))

#             sys.exit()
#         else:
#             vol = self.volBranch(vol)
#             frame = self.frameBranch(frame)

#             x = torch.cat((vol, frame), 2)

#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#             x = self.layer4(x)

#             x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#             x = self.fc1(x)

#             mat_out = x.clone()

#             x = torch.reshape(x, (x.shape[0], 4, 4))
#             grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=x, 
#                                        input_spacing=(1, 1, 1), device=device)
#             # grid = grid.to(device)
#             vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)

            
#         return vol_resampled, mat_out


class mynet3(nn.Module):
    """ First working model! """
    def __init__(self, layers):
        self.inplanes = 64
        super(mynet3, self).__init__()
        """ Balance """
        layers = layers
        # layers = [3, 4, 6, 3]  # resnext50
        # layers = [3, 4, 23, 3]  # resnext101
        # layers = [3, 8, 36, 3]  # resnext150
        self.conv1_vol = nn.Conv3d(1, 32, kernel_size=9, stride=(2, 2, 2), padding=(4, 4, 4), bias=False)
        self.conv2_vol = nn.Conv3d(32, 64, kernel_size=7, stride=(2, 1, 1), padding=(3, 3, 3), bias=False)
        # self.conv3_vol = nn.Conv3d(128, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        # self.conv4_vol = nn.Conv3d(64, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn1_vol = nn.BatchNorm3d(32)
        self.bn2_vol = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.conv_pool = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv1_frame = nn.Conv3d(1, 32, kernel_size=9, stride=(2, 2, 2), padding=(4, 4, 4), bias=False)
        self.conv2_frame = nn.Conv3d(32, 64, kernel_size=7, stride=(2, 1, 1), padding=(3, 3, 3), bias=False)
        # self.conv3_frame = nn.Conv3d(128, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        self.conv2d_frame = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)

        self.layer1 = self._make_layer(
            ResNeXtBottleneck, 128, layers[0], shortcut_type='B', cardinality=32, stride=2)
        self.layer2 = self._make_layer(
            ResNeXtBottleneck, 256, layers[1], shortcut_type='B', cardinality=32, stride=2)
        self.layer3 = self._make_layer(
            ResNeXtBottleneck, 512, layers[2], shortcut_type='B', cardinality=32, stride=2)
        self.layer4 = self._make_layer(
            ResNeXtBottleneck, 1024, layers[3], shortcut_type='B', cardinality=32, stride=2)
        
        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        self.maxpool = nn.MaxPool3d((1, 4, 7), stride=1)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 6)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def volBranch(self, vol):
        # print('\n********* Vol *********')
        # print('vol {}'.format(vol.shape))
        vol = self.conv1_vol(vol)
        vol = self.bn1_vol(vol)
        vol = self.relu(vol)
        # print('conv1 {}'.format(vol.shape))


        vol = self.conv2_vol(vol)
        vol = self.relu(vol)
        # print('conv2 {}'.format(vol.shape))

        return vol
    
    def frameBranch(self, frame):
        # print('\n********* Frame *********')
        # print('frame {}'.format(frame.shape))
        frame = frame.squeeze(1)
        # print('squeeze {}'.format(frame.shape))

        frame = self.conv2d_frame(frame)
        # print('conv2d_frame {}'.format(frame.shape))

        frame = frame.unsqueeze(1)
        # print('unsqueeze {}'.format(frame.shape))

        frame = self.conv1_frame(frame)
        frame = self.bn1_vol(frame)
        frame = self.relu(frame)
        # print('conv1 {}'.format(frame.shape))

        frame = self.conv2_frame(frame)
        frame = self.relu(frame)
        # print('conv2 {}\n'.format(frame.shape))

        return frame

    def forward(self, vol, frame, device=None):
        input_vol = vol.clone()

        show_size = False
        # show_size = True
        if show_size:
            vol = self.volBranch(vol)
            frame = self.frameBranch(frame)

            x = torch.cat((vol, frame), 2)
            print('cat {}'.format(x.shape))
            # sys.exit()

            x = self.layer1(x)
            print('layer1 {}'.format(x.shape))

            x = self.layer2(x)
            print('layer2 {}'.format(x.shape))

            x = self.layer3(x)
            print('layer3 {}'.format(x.shape))

            x = self.layer4(x)
            print('layer4 {}'.format(x.shape))

            x = self.avgpool(x)
            print('avgpool {}'.format(x.shape))

            x = x.view(x.size(0), -1)
            print('view {}'.format(x.shape))

            x = self.fc1(x)
            print('fc1 {}'.format(x.shape))
            # dof_out = x.clone()
            x = self.relu(x)
            x = self.fc2(x)

            
            mat = tools.dof2mat_tensor(input_dof=x, device=device)
            print('mat {}'.format(mat.shape))
            
            print('input_vol {}'.format(input_vol.shape))
            grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=mat, 
                                       input_spacing=(1, 1, 1), device=device)
            # grid = grid.to(device)
            print('grid {}'.format(grid.shape))
            vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)
            print('resample {}'.format(vol_resampled.shape))
            print('mat_out {}'.format(x.shape))

            sys.exit()
        else:
            vol = self.volBranch(vol)
            frame = self.frameBranch(frame)

            x = torch.cat((vol, frame), 2)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            # x = self.maxpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

            # x = torch.reshape(x, (x.shape[0], 3, 4))

            mat = tools.dof2mat_tensor(input_dof=x, device=device)
            # indices = torch.tensor([0, 1, 2]).to(device)
            # mat = torch.index_select(mat, 1, indices)

            grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=mat, 
                                       input_spacing=(1, 1, 1), device=device)

            # input_tensor=input_vol
            # input_mat=mat
            # input_spacing=(1, 1, 1)
            # device=device

            # input_spacing = np.asarray(input_spacing)
            # image_size = np.asarray([input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]])
            # image_phy_size = (image_size - 1) * input_spacing
            # # image_phy_size = [input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]]
            # grid_size = input_tensor.shape
            # t_mat = input_mat

            # # generate grid of input image (i.e., the coordinate of the each pixel in the input image. The center point of the input image volume is assigned as (0, 0, 0).)
            # grid_x_1d = torch.linspace(-0.5 * image_phy_size[0], 0.5 * image_phy_size[0], steps=grid_size[4])
            # grid_y_1d = torch.linspace(-0.5 * image_phy_size[1], 0.5 * image_phy_size[1], steps=grid_size[3])
            # grid_z_1d = torch.linspace(-0.5 * image_phy_size[2], 0.5 * image_phy_size[2], steps=grid_size[2])
            # grid_z, grid_y, grid_x = torch.meshgrid(grid_z_1d, grid_y_1d, grid_x_1d)
            # grid_x = grid_x.unsqueeze(0)
            # grid_y = grid_y.unsqueeze(0)
            # grid_z = grid_z.unsqueeze(0)
            # origin_grid = torch.cat([grid_x, grid_y, grid_z, torch.ones_like(grid_x)], dim=0)
            # origin_grid = origin_grid.view(4, -1)
            # if device:
            #     origin_grid = origin_grid.to(device)
            #     origin_grid.requires_grad = True

            # # compute the rasample grid through matrix multiplication
            # # print('t_mat {}, origin_grid {}'.format(t_mat.shape, origin_grid.shape))
            # t_mat = torch.tensor(t_mat)
            # t_mat = t_mat.float()
            # # t_mat = t_mat.squeeze()
            # # origin_grid = origin_grid.unsqueeze(0)
            # # print('t_mat shape {}'.format(t_mat.shape))
            # # print('origin_grid shape {}'.format(origin_grid.shape))
            # # resample_grid = torch.matmul(t_mat, origin_grid)[0:3, :]
            # resample_grid = torch.matmul(t_mat, origin_grid)[:, 0:3, :]
            # # print('resample_grid {}'.format(resample_grid.shape))

            # # convert the resample grid coordinate from physical coordinate system to a range of [-1, 1] (which is required by the PyTorch interface 'grid_sample'). 
            # resample_grid[:, 0, :] = (resample_grid[:, 0, :] + 0.5 * image_phy_size[0]) / image_phy_size[0] * 2 - 1
            # resample_grid[:, 1, :] = (resample_grid[:, 1, :] + 0.5 * image_phy_size[1]) / image_phy_size[1] * 2 - 1
            # resample_grid[:, 2, :] = (resample_grid[:, 2, :] + 0.5 * image_phy_size[2]) / image_phy_size[2] * 2 - 1
            # # print('resample_grid2 {}'.format(resample_grid.shape))
            # resample_grid = resample_grid.permute(0,2,1).contiguous()
            # resample_grid = resample_grid.reshape(grid_size[0], grid_size[2], grid_size[3], grid_size[4], 3)
            
            # grid = F.affine_grid(x, input_vol.size())
            # vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)
            vol_resampled = F.grid_sample(input_vol, grid)
            # del mat
            
        return vol_resampled, x
    

# class mynet4(nn.Module):

#     def __init__(self, ):
#         self.inplanes = 64
#         super(mynet4, self).__init__()
#         """ Balance """
#         # layers = [3, 4, 6, 3]  # resnext50
#         # layers = [3, 4, 23, 3]  # resnext101
#         layers = [3, 8, 36, 3]  # resnext150
#         self.conv1_vol = nn.Conv3d(1, 16, kernel_size=7, stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
#         self.conv2_vol = nn.Conv3d(16, 32, kernel_size=5, stride=(1, 1, 1), padding=(2, 2, 2), bias=False)
#         self.conv3_vol = nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
#         self.mpl1_vol = nn.MaxPool3d((5, 5, 5), stride=(2, 2, 2), padding=(2, 2, 2))
#         self.mpl2_vol = nn.MaxPool3d((5, 5, 5), stride=(2, 1, 1), padding=(2, 2, 2))

#         self.bn16_vol = nn.BatchNorm3d(16)
#         self.bn32_vol = nn.BatchNorm3d(32)
#         self.bn48_vol = nn.BatchNorm3d(48)
#         self.bn64_vol = nn.BatchNorm3d(64)

#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
#         # self.conv_pool = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

#         self.conv1_frame = nn.Conv3d(1, 16, kernel_size=7, stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
#         self.conv2_frame = nn.Conv3d(16, 32, kernel_size=5, stride=(1, 1, 1), padding=(2, 2, 2), bias=False)
#         self.conv3_frame = nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

#         # self.conv3_frame = nn.Conv3d(128, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

#         self.conv2d_frame = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)

#         self.layer1 = self._make_layer(
#             ResNeXtBottleneck, 128, layers[0], shortcut_type='B', cardinality=32, stride=2)
#         self.layer2 = self._make_layer(
#             ResNeXtBottleneck, 256, layers[1], shortcut_type='B', cardinality=32, stride=2)
#         self.layer3 = self._make_layer(
#             ResNeXtBottleneck, 512, layers[2], shortcut_type='B', cardinality=32, stride=2)
#         self.layer4 = self._make_layer(
#             ResNeXtBottleneck, 1024, layers[3], shortcut_type='B', cardinality=32, stride=2)
        
#         self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
#         self.maxpool = nn.MaxPool3d((1, 4, 7), stride=1)

#         self.fc1 = nn.Linear(2048, 512)
#         self.fc2 = nn.Linear(512, 6)

#     def _make_layer(self,
#                     block,
#                     planes,
#                     blocks,
#                     shortcut_type,
#                     cardinality,
#                     stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))

#         layers = []
#         layers.append(
#             block(self.inplanes, planes, cardinality, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, cardinality))

#         return nn.Sequential(*layers)

#     def volBranch(self, vol):
#         # print('\n********* Vol *********')
#         # print('vol {}'.format(vol.shape))
#         vol = self.conv1_vol(vol)
#         vol = self.bn16_vol(vol)
#         vol = self.relu(vol)
#         # print('conv1 {}'.format(vol.shape))

#         vol = self.conv2_vol(vol)
#         # vol = self.bn32_vol(vol)
#         vol = self.relu(vol)
#         # print('conv2 {}'.format(vol.shape))

#         vol = self.mpl1_vol(vol)
#         # print('mpl1_vol {}'.format(vol.shape))

#         vol = self.conv3_vol(vol)
#         vol = self.bn64_vol(vol)
#         vol = self.relu(vol)
#         # print('conv3_vol {}'.format(vol.shape))

#         vol = self.mpl2_vol(vol)
#         # print('mpl2_vol {}'.format(vol.shape))

#         # sys.exit()
#         return vol
    
#     def frameBranch(self, frame):
#         # print('\n********* Frame *********')
#         # print('frame {}'.format(frame.shape))
#         frame = frame.squeeze(1)
#         # print('squeeze {}'.format(frame.shape))

#         frame = self.conv2d_frame(frame)
#         # print('conv2d_frame {}'.format(frame.shape))

#         frame = frame.unsqueeze(1)
#         # print('unsqueeze {}'.format(frame.shape))

#         frame = self.conv1_frame(frame)
#         frame = self.bn16_vol(frame)
#         frame = self.relu(frame)
#         # print('conv1_frame {}'.format(frame.shape))

#         frame = self.conv2_frame(frame)
#         # frame = self.bn32_vol(frame)
#         frame = self.relu(frame)
#         # print('conv2_frame {}'.format(frame.shape))

#         frame = self.mpl1_vol(frame)
#         # print('mpl1_vol {}'.format(frame.shape))

#         frame = self.conv3_frame(frame)
#         frame = self.bn64_vol(frame)
#         frame = self.relu(frame)
#         # print('conv3_frame {}'.format(frame.shape))

#         frame = self.mpl2_vol(frame)
#         # print('mpl2_vol {}'.format(frame.shape))

#         # sys.exit()
#         return frame

#     def forward(self, vol, frame, device=None):
#         input_vol = vol.clone()

#         show_size = False
#         # show_size = True

#         if show_size:

#             vol = self.volBranch(vol)
#             frame = self.frameBranch(frame)
#             x = torch.cat((vol, frame), 2)
#             print('\ncat {}'.format(x.shape))
#             # sys.exit()

#             x = self.layer1(x)
#             print('layer1 {}'.format(x.shape))

#             x = self.layer2(x)
#             print('layer2 {}'.format(x.shape))

#             x = self.layer3(x)
#             print('layer3 {}'.format(x.shape))

#             x = self.layer4(x)
#             print('layer4 {}'.format(x.shape))

#             x = self.avgpool(x)
#             print('avgpool {}'.format(x.shape))

#             x = x.view(x.size(0), -1)
#             print('view {}'.format(x.shape))

#             x = self.fc1(x)
#             print('fc1 {}'.format(x.shape))
#             x = self.relu(x)
#             x = self.fc2(x)
            
#             mat = tools.dof2mat_tensor(input_dof=x, device=device)
#             print('mat {}'.format(mat.shape))
            
#             print('input_vol {}'.format(input_vol.shape))
#             grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=mat, 
#                                         input_spacing=(1, 1, 1), device=device)
#             print('grid {}'.format(grid.shape))
#             vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)
#             print('resample {}'.format(vol_resampled.shape))
#             print('mat_out {}'.format(x.shape))

#             sys.exit()
#         else:
#             vol = self.volBranch(vol)
#             frame = self.frameBranch(frame)

#             x = torch.cat((vol, frame), 2)

#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#             x = self.layer4(x)

#             x = self.avgpool(x)
#             # x = self.maxpool(x)
#             x = x.view(x.size(0), -1)
#             x = self.fc1(x)
#             x = self.relu(x)
#             x = self.fc2(x)

#             # x = torch.reshape(x, (x.shape[0], 3, 4))

#             mat = tools.dof2mat_tensor(input_dof=x, device=device)
#             # indices = torch.tensor([0, 1, 2]).to(device)
#             # mat = torch.index_select(mat, 1, indices)

#             grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=mat, 
#                                        input_spacing=(1, 1, 1), device=device)
#             vol_resampled = F.grid_sample(input_vol, grid)

#         return vol_resampled, x
