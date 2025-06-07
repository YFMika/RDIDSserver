import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from rdds.models.fpn import PANBlock
from rdds.utils.misc import ACTIVATION_DICT
from rdds.models.transformers import TransformerEncoderLayer


class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shortcut, expansion=1, use_bias=False, act='relu', variant='d'):
        super().__init__()

        self.expansion = expansion

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        self.branch2a = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride1, padding=0, bias=use_bias
        )
        self.branch2b = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride2, padding=1, bias=use_bias
        )
        self.branch2c = nn.Conv2d(
            in_channels, out_channels * expansion, kernel_size=1, stride=1, padding=0, bias=use_bias
        )

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)),
                    ('conv', nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0, bias=use_bias))
                ]))
            else:
                self.short = nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0, bias=use_bias)

        self.act = nn.Identity() if act is None else ACTIVATION_DICT[act]()

    def forward(self, input):
        '''
            input: torch.tensor, (b, c, h, w)
        '''
        out = self.branch2a(input)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = input
        else:
            short = self.short(input)

        out = out + short
        out = self.act(out)

        return out
    

class BackBone(nn.Module):

    def __init__(self, num_blocks, in_channels, hidden_channels, out_channels, stride, shortcut, expansion=2, use_bias=False, act='relu', variant='d', double=False):
        super().__init__()

        if double:
            self.conv1 = nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=use_bias
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=use_bias
            )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bottlenecks = nn.Sequential(*[
            BottleNeck(hidden_channels * i, hidden_channels * i, stride=stride, expansion=expansion, shortcut=shortcut, act=act, variant=variant) 
            for i in range(1, num_blocks + 1)
        ])
        self.conv2 = nn.Conv2d(
            hidden_channels * pow(2, num_blocks), out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias
        )

    def forward(self, input):
        input = self.maxpool(self.conv1(input))
        input = self.bottlenecks(input)
        input = self.conv2(input)

        return input
    

class LargeBackBone(nn.Module):

    def __init__(self, num_blocks, in_channels, hidden_channels, out_channels, stride, shortcut, expansion=2, use_bias=False, act='relu', variant='d'):
        super().__init__()

        self.backbone = BackBone(num_blocks, in_channels, hidden_channels, out_channels, stride, 
                                 shortcut, expansion, use_bias, act, variant, double=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, input):

        return self.norm(self.backbone(input))
    

class OrdinalRegressionHead(nn.Module):
    def __init__(self, in_features, num_classes, bias=False):
        """
        有序回归头，将输入特征映射到多个二元分类器
        参数:
            in_features: 输入特征维度
            num_classes: 有序类别数（如严重程度1-5，num_classes=5）
        """
        super(OrdinalRegressionHead, self).__init__()
        self.num_classes = num_classes
        # 创建 num_classes-1 个二元分类器
        self.logits = nn.Linear(in_features, num_classes - 1, bias)
    
    def forward(self, x):
        """
        返回:
            每个分割点的概率
        """
        return self.logits(x)
    

class SimpleMLP(nn.Module):
    
    def __init__(self, in_dims, hidden_dims, out_dims, act='relu'):
        super().__init__()

        self.linear1 = nn.Linear(in_dims, hidden_dims, bias=True)
        self.act = nn.Identity() if act is None else ACTIVATION_DICT[act]()
        self.linear2 = nn.Linear(hidden_dims, out_dims, bias=True)

    def forward(self, input):
        return self.linear2(self.act(self.linear1(input)))
    

class SimpleOrdinalMLP(nn.Module):
    
    def __init__(self, in_dims, hidden_dims, out_dims, act='relu'):
        super().__init__()

        self.linear1 = nn.Linear(in_dims, hidden_dims, bias=True)
        self.act = nn.Identity() if act is None else ACTIVATION_DICT[act]()
        self.linear2 = OrdinalRegressionHead(hidden_dims, out_dims, bias=False)

    def forward(self, input):
        return self.linear2(self.act(self.linear1(input)))
    

class Classify_head(nn.Module):

    def __init__(self, args, act='relu'):
        super().__init__()

        self.mlp = SimpleMLP(args.in_dim, args.hidden_dims, args.num_classes, act)

    def forward(self, input):
        return self.mlp(input)
    

class Classify_severity(nn.Module):

    def __init__(self, args, act='relu'):
        super().__init__()

        self.mlp = SimpleOrdinalMLP(args.in_dim + args.num_classes, args.hidden_dims, args.num_severity, act)

    def forward(self, input):
        return self.mlp(input)


class Classification(nn.Module):

    def __init__(self, num_blocks, in_channels, hidden_channels, out_channels, stride, shortcut, hidden_dims,
                 expansion=2, use_bias=False, act='relu', variant='d'):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.block0 = LargeBackBone(num_blocks, in_channels, hidden_channels, in_channels, stride, shortcut, expansion, use_bias, act, variant)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block1 = BackBone(num_blocks, in_channels, hidden_channels, out_channels, stride, shortcut, expansion, use_bias, act, variant)
        self.block2 = BackBone(num_blocks, out_channels, hidden_channels, out_channels * 2, stride, shortcut, expansion, use_bias, act, variant)
        self.block3 = BackBone(num_blocks, out_channels * 2, hidden_channels * 2, out_channels * 4, stride, shortcut, expansion, use_bias, act, variant)

        # self.tranformer = TransformerEncoderLayer(
        #     d_model=256,
        #     nhead=4,
        #     dim_feedforward=128,
        #     dropout=0.1,
        # )
        self.fpn = PANBlock([out_channels, out_channels * 2, out_channels * 4], [out_channels, out_channels, out_channels])

        self.flatten = nn.Flatten()

    def forward(self, image):
        out0 = self.block0(image)
        # out0 = self.pool(out0)
        out1 = self.block1(out0)
        out2 = self.block2(out1)
        out3 = self.block3(out2)

        # B, C, H, W = out3.shape

        # out3 = out3.permute(0, 2, 3, 1)
        # # 展平空间维度：[B, H, W, C] -> [B, S, C]
        # out3 = out3.reshape(out3.size(0), -1, out3.size(-1))
        # # [B, S, C] -> [S, B, C]
        # out3 = out3.transpose(0, 1)
        # out3 = out3.transpose(0, 1).reshape(B, C, H, W)

        out = self.fpn([out1, out2, out3])
        out = [out1, out2, out3]

        for i in range(len(out)):
            out[i] = self.flatten(out[i])
        out = torch.concat(out, dim=1)
        # out = self.flatten(out3)

        return out
