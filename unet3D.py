import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
affine_par = True
import functools

import sys, os

in_place = True

class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)




class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, weight_std=False, group = 16):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(group, inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.relu = nn.ReLU(inplace=in_place)

        self.gn2 = nn.GroupNorm(group, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)


        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class EAM_identity(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, dim, input_resolution, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super(EAM_identity, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        #self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.k = nn.Identity()
        self.q = nn.Identity()

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        #self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x, modality_token):
        #print(x.shape, modality_token.shape)
        B_, N, C = x.shape
        B_, Nt, ct = modality_token.shape

        #print(x.shape, modality_token.shape)

        k = self.k(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.k(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        q = self.q(modality_token).reshape(B_, Nt, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attnf = self.softmax(attn)

        x = (attnf @ v).transpose(1, 2).reshape(B_, Nt, C)
        x = self.proj(self.norm2(x)) + x
        #x = self.proj_drop(x)

        #print(x.shape)

        return x, attn

class EAM(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, dim, input_resolution, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super(EAM, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        #self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.kv = nn.Linear(dim, dim * 2, bias = False)
        self.q = nn.Linear(dim, dim, bias = False)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        #self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x, modality_token):
        #print(x.shape, modality_token.shape)
        B_, N, C = x.shape
        B_, Nt, ct = modality_token.shape
        #print(modality_token.max(), modality_token.min(), x.max(), x.min())
        x = self.norm2(x)
        
        #modality_token = self.norm2(modality_token)
        modality_token = self.norm3(modality_token)
        #print(modality_token.max(), modality_token.min(), x.max(), x.min())
        #print(x.shape, modality_token.shape)

        kv = self.kv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]
        q = self.q(modality_token).reshape(B_, Nt, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1))
        attnf = self.softmax(attn * self.scale)

        x = (attnf @ v).transpose(1, 2).reshape(B_, Nt, C)
        x = self.proj(self.norm2(x)) + x
        #x = self.proj_drop(x)

        #print(x.shape)
        #print(attn.max(), attn.min(), q.max(), q.min(), modality_token.max(), modality_token.min())

        return x, attn

class EAM_bk(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, dim, input_resolution, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super(EAM_bk, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        #self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.kv = nn.Linear(dim, dim * 2)
        self.q = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        #self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x, modality_token):
        #print(x.shape, modality_token.shape)
        B_, N, C = x.shape
        B_, Nt, ct = modality_token.shape

        #print(x.shape, modality_token.shape)

        kv = self.kv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]
        q = self.q(modality_token).reshape(B_, Nt, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attnf = self.softmax(attn)

        x = (attnf @ v).transpose(1, 2).reshape(B_, Nt, C)
        x = self.proj(self.norm2(x)) + x
        #x = self.proj_drop(x)

        #print(x.shape)

        return x, attn

class unet3D_with_deepsup(nn.Module):
    def __init__(self, layers, num_classes=12, weight_std = False, ema = False, use_cm = [True, True, True]):
        self.inplanes = 128
        self.weight_std = weight_std
        self.num_classes = num_classes
        self.use_cm = use_cm
        super(unet3D_with_deepsup, self).__init__()

        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))

        self.deepout1 = nn.Sequential(
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(128, num_classes, kernel_size=1)
        )



        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))

        self.deepout2 = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(64, num_classes, kernel_size=1)
        )


        
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))

        self.deepout3 = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )




        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )


        if ema:
            for param in self.parameters():
                param.detach_()

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def forward(self, input, ids):
        
        atten_map = []

        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        #print(x.shape) # 256 4 12 12 

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        atten_map.append(self.deepout1(x))

        #print(x.shape) # 128 8 24 24 
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        atten_map.append(self.deepout2(x))
            
        #print(x.shape) # 64 16 48 48
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        atten_map.append(self.deepout3(x))

        #print(x.shape) # 32 32 96 96 
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        #print(x.shape) # 32 64 192 192

        logits = self.precls_conv(x)

        if self.training:
            return logits, atten_map
        else:
            return logits

class unet3D_with_eam(nn.Module):
    def __init__(self, layers, num_classes=12, weight_std = False, ema = False, use_cm = [True, True, True]):
        self.inplanes = 128
        self.weight_std = weight_std
        self.num_classes = num_classes
        self.use_cm = use_cm
        super(unet3D_with_eam, self).__init__()

        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))

        self.eam84 = EAM(128,  input_resolution = None, num_heads = 4)
        self.linear84_2_42 = nn.Linear(128, 64)

        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))

        self.eam42 = EAM(64,  input_resolution = None, num_heads = 4)
        self.linear42_2_21 = nn.Linear(64, 32)
        
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))

        self.eam21 = EAM(32,  input_resolution = None, num_heads = 4)

        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )

        # class token
        parameter = nn.Parameter(torch.randn(num_classes, 128))
        setattr(self, f'class_token', parameter)


        if ema:
            for param in self.parameters():
                param.detach_()

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def forward(self, input, task_id):
        
        atten_map = []

        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        #print(x.shape) # 256 4 12 12 

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        if self.use_cm[0]:

            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            #print(x_t.shape)
            cm, cattn = self.eam84(x_t, self.class_token.view(1, self.num_classes, 128))
            
            #
            atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes, *x.shape[2:])))

        #print(x.shape) # 128 8 24 24 
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        if self.use_cm[0] and self.use_cm[1]:
            cm = self.linear84_2_42(cm)
            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam42(x_t, cm)
            atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes, *x.shape[2:])))
            
        #print(x.shape) # 64 16 48 48
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        if self.use_cm[0] and self.use_cm[1] and self.use_cm[2]:
            cm = self.linear42_2_21(cm)
            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam21(x_t, cm)
            atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes, *x.shape[2:])))

        #print(x.shape) # 32 32 96 96 
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        #print(x.shape) # 32 64 192 192

        logits = self.precls_conv(x)

        if self.training:
            return logits, cm, atten_map
        else:
            return logits

class unet3D_baseline(nn.Module):
    def __init__(self, layers, num_classes=12, weight_std = False, ema = False, use_cm = [True, True, True], deep_up = False):
        self.inplanes = 128
        self.weight_std = weight_std
        self.num_classes = num_classes
        self.use_cm = use_cm
        self.alpha = 0.01
        self.deep_up = deep_up
        super(unet3D_baseline, self).__init__()

        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.upsamplex3 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=8, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))




        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))



        
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))




        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )



        if ema:
            for param in self.parameters():
                param.detach_()

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def forward(self, input, mask = None):
        

        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        #print(x.shape) # 256 4 12 12 

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x) # 1 128 8 24 24


        #print(x.shape) # 128 8 24 24 
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

     
        #print(x.shape) # 64 16 48 48
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        
        #print(x.shape) # 32 32 96 96 
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        #print(x.shape) # 32 64 192 192

        logits = self.precls_conv(x)

        if self.training:
            return logits, [], []
        else:
            return logits


class unet3D_with_feam2(nn.Module):
    def __init__(self, layers, num_classes=12, weight_std = False, ema = False, use_cm = [True, True, True], deep_up = False):
        self.inplanes = 128
        self.weight_std = weight_std
        self.num_classes = num_classes
        self.use_cm = use_cm
        self.alpha = 0.01
        self.deep_up = deep_up
        super(unet3D_with_feam2, self).__init__()

        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.upsamplex3 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=8, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))

        self.deepout1 = nn.Sequential(
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(128, num_classes, kernel_size=1)
        )

        self.eam84 = EAM(128,  input_resolution = None, num_heads = 4)

        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))

        self.deepout2 = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(64, num_classes, kernel_size=1)
        )

        self.eam42 = EAM(64,  input_resolution = None, num_heads = 4)
        
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))

        self.deepout3 = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )


        self.eam21 = EAM(32,  input_resolution = None, num_heads = 4)

        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )

        # class token
        parameter = nn.Parameter(torch.randn(num_classes-1, 128))
        setattr(self, f'class_token1', parameter)
        parameter = nn.Parameter(torch.randn(num_classes-1, 64))
        setattr(self, f'class_token2', parameter)
        parameter = nn.Parameter(torch.randn(num_classes-1, 32))
        setattr(self, f'class_token3', parameter)


        if ema:
            for param in self.parameters():
                param.detach_()

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def forward(self, input, mask = None):
        
        atten_map = []
        deep_map = []

        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        #print(x.shape) # 256 4 12 12 

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x) # 1 128 8 24 24

        deep_map.append(self.deepout1(x))

        if self.training:
            for l in range(self.num_classes):
                if (mask == (l+1)).sum() != 0:
                    cmask = torch.nn.functional.interpolate((mask == (l+1)).float(), x.shape[2:], mode= "nearest").bool().repeat(1,128,1,1,1)
                    if cmask.sum() == 0:
                        continue

                    #print(cmask.sum(), l)
                    #print(x[:,:][cmask].shape)
                    #print(x[:,:][cmask].reshape(x.shape[0],x.shape[1],-1).shape)
                    self.class_token1[l] = self.class_token1[l] * (1-self.alpha) + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * (self.alpha)

        if self.use_cm[0]:

            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam84(x_t, self.class_token1.view(1, self.num_classes-1, 128).detach())
            if not self.deep_up:
                atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))
            else:
                atten_map.append(self.upsamplex4(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:]))))

        #print(x.shape) # 128 8 24 24 
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        deep_map.append(self.deepout2(x))

        if self.training:
            for l in range(self.num_classes):
                if (mask == (l+1)).sum() != 0:
                    cmask = torch.nn.functional.interpolate((mask == (l+1)).float(), x.shape[2:], mode="nearest").bool().repeat(1,64,1,1,1)
                    if cmask.sum() == 0:
                        continue
                    self.class_token2[l] = self.class_token2[l] * (1-self.alpha) + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * (self.alpha)

        if self.use_cm[1]:
            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam42(x_t, self.class_token2.view(1, self.num_classes-1, 64).detach())
            if not self.deep_up:
                atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))
            else:
                atten_map.append(self.upsamplex3(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))) 
            
        #print(x.shape) # 64 16 48 48
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        deep_map.append(self.deepout3(x))

        if self.training:
            for l in range(self.num_classes):
                if (mask == (l+1)).sum() != 0:
                    cmask = torch.nn.functional.interpolate((mask == (l+1)).float(), x.shape[2:], mode="nearest").bool().repeat(1,32,1,1,1)
                    if cmask.sum() == 0:
                        continue
                    self.class_token3[l] = self.class_token3[l] * (1-self.alpha) + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * (self.alpha)

        if self.use_cm[2]:
            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam21(x_t, self.class_token3.view(1, self.num_classes-1, 32).detach())
            if not self.deep_up:
                atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))
            else:
                atten_map.append(self.upsamplex2(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))) 

        #print(x.shape) # 32 32 96 96 
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        #print(x.shape) # 32 64 192 192

        logits = self.precls_conv(x)

        if self.training:
            return logits, atten_map, deep_map
        else:
            return logits

class unet3D_with_feam3(nn.Module):
    # to ema the result with the mask and prediction
    def __init__(self, layers, num_classes=12, weight_std = False, ema = False, use_cm = [True, True, True], deep_up = False):
        self.inplanes = 128
        self.weight_std = weight_std
        self.num_classes = num_classes
        self.use_cm = use_cm
        self.alpha = 0.01
        self.deep_up = deep_up
        super(unet3D_with_feam3, self).__init__()

        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.upsamplex3 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=8, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))

        self.deepout1 = nn.Sequential(
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(128, num_classes, kernel_size=1)
        )

        self.eam84 = EAM(128,  input_resolution = None, num_heads = 4)

        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))

        self.deepout2 = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(64, num_classes, kernel_size=1)
        )

        self.eam42 = EAM(64,  input_resolution = None, num_heads = 4)
        
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))

        self.deepout3 = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )


        self.eam21 = EAM(32,  input_resolution = None, num_heads = 4)

        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )

        # class token
        
        '''
        parameter = nn.Parameter(torch.randn(num_classes-1, 128))
        setattr(self, f'class_token1', parameter)
        parameter = nn.Parameter(torch.randn(num_classes-1, 64))
        setattr(self, f'class_token2', parameter)
        parameter = nn.Parameter(torch.randn(num_classes-1, 32))
        setattr(self, f'class_token3', parameter)
        '''
        parameter = torch.randn(num_classes-1, 128)
        setattr(self, f'class_token1', parameter)
        parameter = torch.randn(num_classes-1, 64)
        setattr(self, f'class_token2', parameter)
        parameter = torch.randn(num_classes-1, 32)
        setattr(self, f'class_token3', parameter)
        


        if ema:
            for param in self.parameters():
                param.detach_()

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def renew_token(self, features, mask):

        for index, x in enumerate(features):
            for l in range(self.num_classes):
                if (mask == (l+1)).sum() != 0:
                    cmask = torch.nn.functional.interpolate((mask == (l+1)).float(), x.shape[2:], mode= "nearest").bool().repeat(1,x.shape[1],1,1,1)
                    if cmask.sum() == 0:
                        continue

                    #print(cmask.sum(), l)
                    #print(x[:,:][cmask].shape)
                    #print(x[:,:][cmask].reshape(x.shape[0],x.shape[1],-1).shape)
                    if index == 0:
                        self.class_token1[l] = self.class_token1[l] * (1-self.alpha) + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * (self.alpha)
                    elif index == 1:
                        self.class_token2[l] = self.class_token2[l] * (1-self.alpha) + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * (self.alpha)
                    else:
                        self.class_token3[l] = self.class_token3[l] * (1-self.alpha) + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * (self.alpha)

    def renew_token2(self, features, mask):
        
        self.class_token1.require_grad = False
        self.class_token2.require_grad = False
        self.class_token3.require_grad = False


        for index, x in enumerate(features):
            for l in range(self.num_classes-1):
                if (mask[:, l]).sum() != 0:
                    cmask = torch.nn.functional.interpolate((mask[:, l:l+1]).float(), x.shape[2:], mode= "nearest").bool().repeat(1,x.shape[1],1,1,1)
                    if cmask.sum() == 0:
                        continue

                    #print(cmask.sum(), l)
                    #print(x[:,:][cmask].shape)
                    #print(x[:,:][cmask].reshape(x.shape[0],x.shape[1],-1).shape)
                    if index == 0:
                        self.class_token1[l] = self.class_token1[l] * (1-self.alpha) + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * (self.alpha)
                    elif index == 1:
                        self.class_token2[l] = self.class_token2[l] * (1-self.alpha) + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * (self.alpha)
                    else:
                        self.class_token3[l] = self.class_token3[l] * (1-self.alpha) + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * (self.alpha)


    def forward(self, input, mask = None):
        self.class_token1 = self.class_token1.to(input.device)
        self.class_token2 = self.class_token2.to(input.device)
        self.class_token3 = self.class_token3.to(input.device)
        atten_map = []
        deep_map = []
        feature_stored = []

        x = self.conv1(input) # 1-32 
        x = self.layer0(x) # 32 - 32
        skip0 = x

        x = self.layer1(x)  # 32 - 64   *2
        skip1 = x

        x = self.layer2(x)  # 64 - 128   * 2
        skip2 = x

        x = self.layer3(x)   # 128 - 256   * 2
        skip3 = x

        x = self.layer4(x)    # 256 - 256   * 2

        x = self.fusionConv(x)  # 256 - 256   

        #print(x.shape) # 256 4 12 12 

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x) # 1 128 8 24 24

        deep_map.append(self.deepout1(x))

        feature_stored.append(x.detach().clone())

        if self.use_cm[0]:

            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam84(x_t, self.class_token1.view(1, self.num_classes-1, 128).detach())
            if not self.deep_up:
                atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))
            else:
                atten_map.append(self.upsamplex4(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:]))))

        #print(x.shape) # 128 8 24 24 
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        deep_map.append(self.deepout2(x))

        feature_stored.append(x.detach().clone())

        if self.use_cm[1]:
            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam42(x_t, self.class_token2.view(1, self.num_classes-1, 64).detach())
            if not self.deep_up:
                atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))
            else:
                atten_map.append(self.upsamplex3(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))) 
            
        #print(x.shape) # 64 16 48 48
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        deep_map.append(self.deepout3(x))

        feature_stored.append(x.detach().clone())


        if self.use_cm[2]:
            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam21(x_t, self.class_token3.view(1, self.num_classes-1, 32).detach())
            if not self.deep_up:
                atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))
            else:
                atten_map.append(self.upsamplex2(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))) 

        #print(x.shape) # 32 32 96 96 
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        #print(x.shape) # 32 64 192 192

        logits = self.precls_conv(x)

        if self.training:
            return logits, atten_map, deep_map, feature_stored
        else:
            return logits


class unet3D_with_feam(nn.Module):
    def __init__(self, layers, num_classes=12, weight_std = False, ema = False, use_cm = [True, True, True]):
        self.inplanes = 128
        self.weight_std = weight_std
        self.num_classes = num_classes
        self.use_cm = use_cm
        super(unet3D_with_feam, self).__init__()

        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))

        self.eam84 = EAM(128,  input_resolution = None, num_heads = 4)

        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))

        self.eam42 = EAM(64,  input_resolution = None, num_heads = 4)
        
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))

        self.eam21 = EAM(32,  input_resolution = None, num_heads = 4)

        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )

        # class token
        parameter = nn.Parameter(torch.randn(num_classes-1, 128))
        setattr(self, f'class_token1', parameter)
        parameter = nn.Parameter(torch.randn(num_classes-1, 64))
        setattr(self, f'class_token2', parameter)
        parameter = nn.Parameter(torch.randn(num_classes-1, 32))
        setattr(self, f'class_token3', parameter)


        if ema:
            for param in self.parameters():
                param.detach_()

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def forward(self, input, mask = None):
        
        atten_map = []

        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        #print(x.shape) # 256 4 12 12 

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x) # 1 128 8 24 24

        if self.training:
            for l in range(13):
                if (mask == (l+1)).sum() != 0:
                    cmask = torch.nn.functional.interpolate((mask == (l+1)).float(), x.shape[2:], mode= "nearest").bool().repeat(1,128,1,1,1)
                    if cmask.sum() == 0:
                        continue
                    #print(x[:,:][cmask].shape)
                    #print(x[:,:][cmask].reshape(x.shape[0],x.shape[1],-1).shape)
                    self.class_token1[l] = self.class_token1[l] * 0.99 + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * 0.01

        if self.use_cm[0] and self.training:

            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam84(x_t, self.class_token1.view(1, self.num_classes-1, 128).detach())
            atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))

        #print(x.shape) # 128 8 24 24 
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        if self.training:
            for l in range(13):
                if (mask == (l+1)).sum() != 0:
                    cmask = torch.nn.functional.interpolate((mask == (l+1)).float(), x.shape[2:], mode="nearest").bool().repeat(1,64,1,1,1)
                    if cmask.sum() == 0:
                        continue
                    self.class_token2[l] = self.class_token2[l] * 0.99 + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * 0.01

        if self.training and self.use_cm[0] and self.use_cm[1]:
            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam42(x_t, self.class_token2.view(1, self.num_classes-1, 64).detach())
            atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))
            
        #print(x.shape) # 64 16 48 48
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        if self.training:
            for l in range(13):
                if (mask == (l+1)).sum() != 0:
                    cmask = torch.nn.functional.interpolate((mask == (l+1)).float(), x.shape[2:], mode="nearest").bool().repeat(1,32,1,1,1)
                    if cmask.sum() == 0:
                        continue
                    self.class_token3[l] = self.class_token3[l] * 0.99 + x[:,:][cmask].reshape(x.shape[1],-1).mean(-1).detach() * 0.01

        if self.training and self.use_cm[0] and self.use_cm[1] and self.use_cm[2]:
            x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
            cm, cattn = self.eam21(x_t, self.class_token3.view(1, self.num_classes-1, 32).detach())
            atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes-1, *x.shape[2:])))

        #print(x.shape) # 32 32 96 96 
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        #print(x.shape) # 32 64 192 192

        logits = self.precls_conv(x)

        if self.training:
            return logits, atten_map
        else:
            return logits


class unet3D_with_eam_baseline(nn.Module):
    def __init__(self, layers, num_classes=12, weight_std = False):
        self.inplanes = 128
        self.weight_std = weight_std
        self.num_classes = num_classes
        super(unet3D_with_eam_baseline, self).__init__()

        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))

        self.eam84 = EAM(128,  input_resolution = None, num_heads = 4)
        self.linear84_2_42 = nn.Linear(128, 64)

        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))

        self.eam42 = EAM(64,  input_resolution = None, num_heads = 4)
        
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))

        #self.eam21 = EAM(32,  input_resolution = None, num_heads = 4)

        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )

        # class token
        parameter = nn.Parameter(torch.randn(num_classes, 128))
        setattr(self, f'class_token', parameter)

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def forward(self, input, task_id):
        
        atten_map = []

        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        #print(x.shape) # 256 4 12 12 

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
        #print(x_t.shape)
        cm, cattn = self.eam84(x_t, self.class_token.view(1, self.num_classes, 128))
        cm = self.linear84_2_42(cm)
        #
        atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes, *x.shape[2:])))

        #print(x.shape) # 128 8 24 24 
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)


        x_t = x.view(x.shape[0], x.shape[1], -1).permute((0, 2, 1))
        cm, cattn = self.eam42(x_t, cm)
        atten_map.append(cattn.mean(1).reshape((x.shape[0], self.num_classes, *x.shape[2:])))
        #print(x.shape) # 64 16 48 48
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        #print(x.shape) # 32 32 96 96 
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        #print(x.shape) # 32 64 192 192

        logits = self.precls_conv(x)

        if self.training:
            return logits, cm, atten_map
        else:
            return logits


class unet3D_g(nn.Module):
    def __init__(self, layers, num_classes=3, weight_std = False, in_channel = 2, init_filter = 32):
        self.inplanes = 128
        self.weight_std = weight_std
        self.init_filter = init_filter
        super(unet3D_g, self).__init__()

        self.conv0 = conv3x3x3(in_channel, self.init_filter, stride=[2, 2, 2], weight_std=self.weight_std)

        self.conv1 = conv3x3x3(self.init_filter, self.init_filter, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, self.init_filter, self.init_filter, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, self.init_filter, self.init_filter * 2, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, self.init_filter * 2, self.init_filter * 4, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, self.init_filter * 4, self.init_filter * 8, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, self.init_filter * 8, self.init_filter * 8, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(self.init_filter // 2, self.init_filter * 8),
            nn.ReLU(inplace=in_place),
            conv3x3x3(self.init_filter * 8, self.init_filter * 8, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        #self.downsample = nn.Downsample(scale_factor = 2, mode='trilinear')

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, self.init_filter * 8, self.init_filter * 4, 1, stride=(1, 1, 1))
        self.x4_resb = self._make_layer(NoBottleneck, self.init_filter * 4, self.init_filter * 2, 1, stride=(1, 1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, self.init_filter * 2, self.init_filter, 1, stride=(1, 1, 1))
        self.x1_resb = self._make_layer(NoBottleneck, self.init_filter, self.init_filter, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(self.init_filter // 4, self.init_filter),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(self.init_filter, num_classes, kernel_size=1)
        )


    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(4, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std, group = 4))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std, group = 4))

        return nn.Sequential(*layers)

    def forward(self, input, _=None):

        #input = nn.functional.interpolate(input, scale_factor = 0.5)
        input = self.conv0(input)

        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        # generate conv filters for classification layer

        #print(x.shape) # 256 4 12 12 

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        #print(x.shape) # 128 8 24 24 
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        #print(x.shape) # 64 16 48 48
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        #print(x.shape) # 32 32 96 96 
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        #print(x.shape) # 32 64 192 192

        logits = self.precls_conv(x)

        logits = self.upsamplex2(logits)

        return logits

class unet3D(nn.Module):
    def __init__(self, layers, num_classes=3, weight_std = False, in_channel = 1, init_filter = 32):
        self.inplanes = 128
        self.weight_std = weight_std
        self.init_filter = init_filter
        super(unet3D, self).__init__()

        self.conv1 = conv3x3x3(in_channel, self.init_filter, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, self.init_filter, self.init_filter, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, self.init_filter, self.init_filter * 2, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, self.init_filter * 2, self.init_filter * 4, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, self.init_filter * 4, self.init_filter * 8, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, self.init_filter * 8, self.init_filter * 8, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, self.init_filter * 8, self.init_filter * 4, 1, stride=(1, 1, 1))
        self.x4_resb = self._make_layer(NoBottleneck, self.init_filter * 4, self.init_filter * 2, 1, stride=(1, 1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, self.init_filter * 2, self.init_filter, 1, stride=(1, 1, 1))
        self.x1_resb = self._make_layer(NoBottleneck, self.init_filter, self.init_filter, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, 8, kernel_size=1)
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.controller = nn.Conv3d(256+7, 162, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]]=1
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, input, task_id):

        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        # generate conv filters for classification layer
        task_encoding = self.encoding_task(task_id)
        task_encoding.unsqueeze_(2).unsqueeze_(2).unsqueeze_(2)
        x_feat = self.GAP(x)
        x_cond = torch.cat([x_feat, task_encoding], 1)
        params = self.controller(x_cond)
        params.squeeze_(-1).squeeze_(-1).squeeze_(-1)

        #print(x.shape) # 256 4 12 12 

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        #print(x.shape) # 128 8 24 24 
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        #print(x.shape) # 64 16 48 48
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        #print(x.shape) # 32 32 96 96 
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        #print(x.shape) # 32 64 192 192

        head_inputs = self.precls_conv(x)

        N, _, D, H, W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, D, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)
        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)

        logits = logits.reshape(-1, 2, D, H, W)

        return logits

def UNet3D(num_classes=1, weight_std=False):
    print("Using DynConv 8,8,2")
    model = unet3D([1, 2, 2, 2, 2], num_classes, weight_std)
    return model


def get_style_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 4, 1, kernel_size=(2,3,2), stride=1, padding=0)
    )

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    
    def forward(self, x):
        return x.view(x.shape[0], -1)

def get_style_discriminator_output(num_classes, ndf=32):
    return nn.Sequential(
        nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True), # new three
        nn.AdaptiveAvgPool3d(1),
        Reshape(),
        nn.Linear(ndf * 8, 1)
    )


class deep_style_discriminator_output(nn.Module):
    def __init__(self, num_classes, ndf = 32):
        
        super(deep_style_discriminator_output, self).__init__()
        self.ndf = ndf

        self.block1 = nn.Sequential(
            nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.min_block1 = nn.Sequential(
            nn.Conv3d(1, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block2 = nn.Sequential(
            nn.Conv3d(ndf*2, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.min_block2 = nn.Sequential(
            nn.Conv3d(1, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block3 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.min_block3 = nn.Sequential(
            nn.Conv3d(1, ndf * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block4 = nn.Sequential(
            nn.Conv3d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), # new three
            nn.AdaptiveAvgPool3d(1),
            Reshape(),
            nn.Linear(ndf * 8, 2)
        )

    def forward(self, x_in, f_m):

        x = self.block1(x_in)
        xm1 = self.min_block1(f_m[2])

        x = self.block2(torch.cat([x, xm1], 1))
        #for l in f_m:
        #    print(l.shape)
        xm2 = self.min_block2(f_m[1])       
        x = self.block3(torch.cat([x, xm2], 1))

        xm3 = self.min_block3(f_m[0]) 
        x = self.block4(torch.cat([x, xm3], 1)) 

        return x

class norm_style_discriminator_output(nn.Module):
    def __init__(self, num_classes, ndf = 32):
        
        super(norm_style_discriminator_output, self).__init__()
        self.ndf = ndf

        self.block1 = nn.Sequential(
            nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block2 = nn.Sequential(
            nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block3 = nn.Sequential(
            nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block4 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), # new three
            nn.AdaptiveAvgPool3d(1),
            Reshape(),
            nn.Linear(ndf * 8, 2)
        )

    def forward(self, x_in):

        x = self.block1(x_in)

        x = self.block2(torch.cat([x], 1))

        x = self.block3(torch.cat([x], 1))

        x = self.block4(torch.cat([x], 1)) 

        return x


def get_style_discriminator_linear(num_classes, ndf=64):
    return nn.Sequential(
        nn.Linear(num_classes, ndf),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Linear(ndf, ndf * 2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Linear(ndf * 2, 1)
    )