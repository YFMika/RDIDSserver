import torch
import torch.nn as nn
import torch.nn.functional as F

from rdds.utils.misc import ACTIVATION_DICT

class ConvBN(nn.Module):
    """卷积 + 批归一化 + ReLU激活"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = ACTIVATION_DICT[act]()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PANBlock(nn.Module):
    """PANet中的特征金字塔聚合模块"""
    def __init__(self, in_channels_list, out_channels_list, act="silu"):
        """
        参数:
            in_channels_list: 输入特征图的通道数列表，从低层到高层 [C3, C4, C5]
            out_channels_list: 输出特征图的通道数列表，与输入对应 [P3, P4, P5]
        """
        super(PANBlock, self).__init__()
        assert len(in_channels_list) == len(out_channels_list), "输入和输出通道数列表长度必须相同"
        self.num_levels = len(in_channels_list)
        
        # 自顶向下路径 (Top-down pathway)
        self.toplayer = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_levels - 1, -1, -1):  # 从高层到低层 (C5→C4→C3)
            self.toplayer.append(ConvBN(in_channels_list[i], out_channels_list[i], 1, act=act))
            self.lateral_convs.append(ConvBN(out_channels_list[i] * 2, out_channels_list[i - 1], 3, padding=1, act=act))
        
        # 自底向上路径 (Bottom-up pathway)
        self.bottom_up_convs = nn.ModuleList()
        for i in range(self.num_levels):  # 从低层到高层 (P2→P3→P4)
            self.bottom_up_convs.append(ConvBN(out_channels_list[i], out_channels_list[i], 3, stride=2, padding=1, act=act))
    
    def forward(self, inputs):
        """
        参数:
            inputs: 输入特征图列表，从低层到高层 [C3, C4, C5]
        
        返回:
            outputs: 增强后的特征图列表，与输入对应 [P3, P4, P5]
        """
        assert len(inputs) == self.num_levels, "输入特征图数量与初始化时不一致"
        
        # 自顶向下路径
        laterals = [self.toplayer[0](inputs[-1])]  # 最高层特征
        for i in range(1, self.num_levels):
            # 上采样并与下一层特征融合
            upsample = F.interpolate(laterals[-1], size=inputs[self.num_levels - i - 1].shape[2:], mode='nearest')
            lateral = self.toplayer[i](inputs[self.num_levels - i - 1])
            laterals.append(self.lateral_convs[i - 1](torch.concat([upsample, lateral], dim=1)))

        laterals = laterals[::-1]  # 反转列表，使其顺序为 [P3, P4, P5]
        
        # 自底向上路径
        outputs = [laterals[0]]  # 最低层特征无需处理
        for i in range(self.num_levels - 1):
            # 下采样并与上一层特征融合
            bottom_up = self.bottom_up_convs[i](outputs[-1])
            outputs.append(torch.concat([bottom_up + laterals[i + 1]], dim=1))
        
        return outputs