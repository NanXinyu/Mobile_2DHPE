import torch 
import torch.nn as nn

from typing import List, Sequence, Optional
from einops import rearrange
from functools import partial

import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Spatial_SE(nn.Module):
    def __init__(
        self,
        in_channels,
        squeeze_ratio = 64,
        activation = nn.ReLU,
        scale_activation = nn.Hardsigmoid):
        super(Spatial_SE, self).__init__()
        squeeze_channels = _make_divisible(in_channels // squeeze_ratio, 8)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.maxpool = nn.AdaptiveMaxPool2d(1) if SE_type == 'C' else nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv1d(squeeze_channels, in_channels, 1)

        self.activation = activation(inplace = True)
        self.scale_activation = scale_activation()

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        avg_x = self.avgpool(x)
        
        scale = self.fc2(self.activation(self.fc1(avg_x)))
        x = x * self.scale_activation(scale)
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        return x

class CA(nn.Module):
    def __init__(self, inp, reduction = 16):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (b,c,h,w)-->(b,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (b,c,h,w)-->(b,c,1,w)
 
        mip =  _make_divisible(inp // reduction, 8)  
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace = True)
 
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out

        
class Stem(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        groups = 1,
        dilation = 1,      
        norm_layer = nn.BatchNorm2d,
        activation = 'RE',
    ):
        super(Stem, self).__init__()
        self.activation = activation
        padding = (kernel_size - 1)//2 * dilation
    
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation, 
            groups = groups, bias = False)

        self.norm_layer = norm_layer(out_channels, eps=0.01, momentum=0.01)
        if activation == 'PRE':
            self.acti_layer = nn.PReLU()
        elif activation == 'HS':
            self.acti_layer = nn.Hardswish(inplace = True)  
        else:
            self.acti_layer = nn.ReLU(inplace = True)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        if self.activation is not None:
            x = self.acti_layer(x)
        return x


class BlockConfig:
    def __init__(
        self,
        in_channels,
        exp_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        activation,
        use_se
    ):
        self.in_channels = in_channels
        self.exp_channels = exp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.use_se = use_se
        
class Block(nn.Module):
    def __init__(
        self,
        cnf: BlockConfig,
        ):
        super(Block, self).__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.in_channels == cnf.out_channels

        layers = []
        # expand
        if cnf.exp_channels != cnf.in_channels:
            layers.append(
                Stem(
                    cnf.in_channels, 
                    cnf.exp_channels, 
                    kernel_size=1,
                    activation=cnf.activation
                )
            )

        # depthwise
        layers.append(
            Stem(
                cnf.exp_channels,
                cnf.exp_channels,
                kernel_size=cnf.kernel_size,
                stride=cnf.stride,
                groups=cnf.exp_channels,
                dilation=cnf.dilation,
                activation=cnf.activation,
            )
        )

        if cnf.use_se == True:
            layers.append(
                CA(cnf.exp_channels)
            )
        else:
            layers.append(
                nn.Identity()
            )
        
        layers.append(
            Stem(
                cnf.exp_channels,
                cnf.out_channels,
                kernel_size=1,
                activation=None))

        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect == True:
            result = result + x
        return result

class generate_location(nn.Module):
    def __init__(
            self,
            image_size,
            intermediate_size,
            embedding_channels
            
            ):
        super(generate_location, self).__init__()
        in_channels = intermediate_size[1] * intermediate_size[0]
        self.conv_h = nn.Sequential(
            nn.Conv1d(in_channels, image_size[1], 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv_w = nn.Sequential(
            nn.Conv1d(in_channels, image_size[0], 1, bias = True),
            nn.ReLU(inplace = True))
    def forward(self, x):
        x = x.flatten(2).permute(0, 2, 1)
        # print(x.shape)
        y = self.conv_h(x).permute(0, 2, 1)
        x = self.conv_w(x).permute(0, 2, 1)
        return x, y

class MobilePosNet(nn.Module):
    def __init__(
        self,
        BlockSetting: List[BlockConfig],
        num_joints,
        heatmaps_size,
        output_size,
        embedding_size = [192, 256]
    ):
        super(MobilePosNet, self).__init__()
        
        self.num_joints = num_joints
        layers = []
        # building first layer
        first_output_channels = BlockSetting[0].in_channels
        layers.append(
            Stem(
                3,
                first_output_channels,
                kernel_size=3,
                stride=2,
                activation='HS'
            )          
        )

        # building stage1 other blocks
        for cnf in BlockSetting:
            layers.append(Block(cnf))
        
        output_channel = BlockSetting[-1].out_channels
        last_channel = output_channel 
        self.Net = nn.Sequential(*layers)
     
        self.joints_classifier = nn.Sequential( 
            # Stem(output_channel, last_channel, 1),
            nn.Conv2d(last_channel, num_joints, kernel_size = 1),
            # nn.UpsamplingBilinear2d(scale_factor = 2), 
            nn.ReLU(inplace = True))
         
        self.coord_generator = generate_location(output_size, heatmaps_size, embedding_size)
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # nn.init.normal_(m.weight, 0, 0.01)
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            
    def forward(self, x):
      
        x = self.Net(x)
        joints = self.joints_classifier(x)
        # print(joints.shape)
        # print(joints.shape)
        x, y = self.coord_generator(joints)
        # print(x.shape)
        # print(y.shape)
        return x, y

def _mobileposnet_conf(arch: str):
    stage_conf = BlockConfig

    if arch == "SimCC_layers=13":
        block_setting = [
            stage_conf(32, 32, 16, 3, 1, 1, 'RE', True),
            stage_conf(16, 96, 24, 3, 2, 1, 'RE', True),
            stage_conf(24, 144, 24, 3, 1, 1, 'HS', True),
            stage_conf(24, 144, 32, 3, 2, 1, 'HS', True),
            stage_conf(32, 192, 32, 3, 1, 1, 'HS', True),
            stage_conf(32, 192, 32, 3, 1, 1, 'HS', True),
            stage_conf(32, 192, 64, 3, 2, 1, 'HS', True),
            stage_conf(64, 384, 64, 3, 1, 1, 'HS', True),
            stage_conf(64, 384, 64, 3, 1, 1, 'HS', True),
            stage_conf(64, 384, 64, 3, 1, 1, 'HS', True),
            stage_conf(64, 384, 96, 3, 1, 1, 'HS', True),
            stage_conf(96, 576, 96, 3, 1, 1, 'HS', True),
            stage_conf(96, 576, 96, 3, 1, 1, 'HS', True),
        ]
    
    elif arch == "SimCC_layers=14":
        block_setting = [
            stage_conf(16, 16, 16, 3, 2, 1, 'RE', True),
            stage_conf(16, 72, 24, 3, 2, 1, 'RE', True),
            # stage_conf(24, 88, 24, 3, 1, 1, 'RE', True),
            stage_conf(24, 88, 24, 3, 1, 1, 'HS', True),
            # stage_conf(24, 96, 24, 3, 1, 1, 'HS', True),
            stage_conf(24, 96, 40, 5, 1, 1, 'HS', True),
            stage_conf(40, 240, 40, 3, 1, 1, 'HS', True),
            stage_conf(40, 240, 40, 3, 1, 1, 'HS', True),
            stage_conf(40, 240, 40, 3, 1, 1, 'HS', True),
            stage_conf(40, 240, 40, 3, 1, 1, 'HS', True),
            stage_conf(40, 120, 40, 3, 1, 1, 'HS', True),
            stage_conf(40, 120, 48, 3, 1, 1, 'HS', True),
            stage_conf(48, 144, 48, 3, 1, 1, 'HS', True),
            stage_conf(48, 144, 48, 3, 1, 1, 'HS', True),
            stage_conf(48, 288, 48, 3, 1, 1, 'HS', True),
            stage_conf(48, 288, 96, 3, 1, 1, 'HS', True),
            # stage_conf(96, 576, 96, 5, 1, 1, 'HS', True),
            # stage_conf(96, 576, 96, 3, 1, 1, 'HS', True) 
        ]
    
    elif arch == "SimCC_MobileNetV3(8-32-256)":
        block_setting = [
            stage_conf(16, 16, 16, 3, 2, 1, 'RE', True),
            stage_conf(16, 72, 24, 3, 2, 1, 'RE', True),
            # stage_conf(24, 88, 24, 3, 1, 1, 'RE', True),
            stage_conf(24, 88, 24, 3, 1, 1, 'HS', True),
            # stage_conf(24, 96, 24, 3, 1, 1, 'HS', True),
            stage_conf(24, 96, 40, 5, 2, 1, 'HS', True),
            stage_conf(40, 240, 40, 5, 1, 1, 'HS', True),
            stage_conf(40, 240, 40, 5, 1, 1, 'HS', True),
            stage_conf(40, 120, 48, 5, 1, 1, 'HS', True),
            stage_conf(48, 144, 48, 5, 1, 1, 'HS', True),
            stage_conf(48, 288, 96, 5, 2, 1, 'HS', True),
            stage_conf(96, 576, 96, 5, 1, 1, 'HS', True),
            stage_conf(96, 576, 96, 5, 1, 1, 'HS', True) 
        ]
    
    elif arch == "MobileNet_V3_PosNet_32_":
        block_setting = [
            # stage_conf(16, 16, 16, 3, 1, 1, 'RE', True),
            stage_conf(16, 64, 24, 3, 2, 1, 'RE', True),
            stage_conf(24, 72, 24, 3, 1, 1, 'RE', True),
            stage_conf(24, 72, 40, 5, 2, 1, 'HS', True),
            stage_conf(40, 120, 40, 5, 1, 1, 'HS', True),
            # stage_conf(40, 120, 40, 3, 1, 1, 'HS', True),
            # stage_conf(40, 120, 40, 3, 1, 1, 'HS', True),
            stage_conf(40, 120, 40, 5, 1, 1, 'HS', True),
            stage_conf(40, 240, 80, 3, 1, 1, 'HS', True),
            stage_conf(80, 200, 80, 3, 1, 1, 'HS', True),
            stage_conf(80, 184, 80, 3, 1, 1, 'HS', True),
            stage_conf(80, 480, 112, 3, 1, 1, 'HS', True),
            # stage_conf(96, 576, 96, 5, 1, 1, 'HS', True),
            # stage_conf(96, 576, 96, 3, 1, 1, 'HS', True) 
        ]
    
    else:
        raise ValueError(f"Unsupported model type {arch}")
    return block_setting
    # return block_setting, Tblock_setting, last_channel

def get_pose_net(cfg, is_train):
    block_setting = _mobileposnet_conf(cfg.MODEL.NAME)
    # [6, 8]
    intermediate_size = cfg.MODEL.INTERMEDIATE_SIZE
    output_size = cfg.MODEL.IMAGE_SIZE
    # pmap_size = (cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.PATCH_SIZE[0], cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.PATCH_SIZE[1])
    # model = MobilePosNet(block_setting, cfg.MODEL.NUM_JOINTS, cfg.MODEL.PATCH_SIZE, pmap_size)
    model = MobilePosNet(block_setting, cfg.MODEL.NUM_JOINTS, intermediate_size, output_size)
    if is_train:
        model.init_weights()
    return model

    

             
        
        

        





        