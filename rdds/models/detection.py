import torch
import torch.nn as nn

from rdds.models.classify import Classification, Classify_head, Classify_severity
from rdds.models.criterion import OverallCriterion
from clip import clip


class Detection(nn.Module):

    def __init__(self, args, expansion=2, use_bias=False, act='relu', variant='d', is_eval=False):
        super().__init__()

        self.is_pretrain = args.is_pretrain
        
        if self.is_pretrain:
            self.classify, _ = clip.load("ViT-B/32")
            for param in self.classify.parameters():
                param.requires_grad = False
        else:
            self.classify = Classification(
                num_blocks=args.num_blocks, 
                in_channels=args.in_channels, 
                hidden_channels=args.hidden_channels, 
                out_channels=args.out_channels, 
                stride=args.stride, 
                shortcut=args.shortcut, 
                hidden_dims=args.hidden_dims,
                expansion=expansion, 
                use_bias=use_bias, 
                act=act, 
                variant=variant
            )
            self.dtype = None

        self.num_severity = args.num_severity

        self.classify_head = Classify_head(args)
        self.classify_severity = Classify_severity(args)
        
        if is_eval is False:
            self.criterion = OverallCriterion(args)
        
        self.train()

    def forward(self, image, target=None, is_eval=False):
        output = {}
        if self.is_pretrain:
            out = self.classify.encode_image(image)
            out = out.float()
        else:
            out = self.classify(image)

        classes = self.classify_head(out)
        severity = self.classify_severity(torch.concat([classes, out], dim=-1))

        output["out"] = out
        output["classes"] = classes
        output["severity"] = severity
        
        if is_eval is False and target is not None:
            output["loss"] = self.criterion(output, target)[0]
        return output
    
    def predict(self, logits):
        """将模型输出转换为类别预测"""
        # 计算累积概率 P(Y > k)
        prob = torch.sigmoid(logits)
        
        # 计算每个类别的概率 P(Y = k)
        p = torch.zeros((logits.size(0), self.num_severity), device=logits.device)
        p[:, 0] = 1.0 - prob[:, 0]  # P(Y=0) = 1 - P(Y>0)
        
        for k in range(1, self.num_severity - 1):
            p[:, k] = prob[:, k-1] - prob[:, k]  # P(Y=k) = P(Y>k-1) - P(Y>k)
        
        p[:, self.num_severity - 1] = prob[:, self.num_severity - 2]  # P(Y=3) = P(Y>2)
        
        # 返回概率最大的类别
        return torch.argmax(p, dim=1)

