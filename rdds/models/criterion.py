import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict


class Criterion(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.criterion_classes = nn.BCEWithLogitsLoss()
        self.criterion_severity = nn.CrossEntropyLoss()
        
        self.loss_dict = {
            'loss_classes': (self.loss_classes, args.loss_classes_weight),
            'loss_severity': (self.loss_severity, args.loss_severity_weight),
        }


    def loss_classes(self, output, target) -> Dict:
        loss = self.criterion_classes(output['classes'], target['type_label'])
    
        return {'loss_classes': loss}
    

    def loss_severity(self, output, target) -> Dict:
        """
        有序回归损失函数
        
        参数:
            logits: 模型输出的logits，形状为 [batch_size, num_classes - 1]
            targets: 真实类别标签，形状为 [batch_size]
            weight: 每个样本的权重（可选）
        """
        logits = output['severity']
        severity_true = target['yanzhong_label']
        batch_size, num_thresholds = logits.size()
        
        # 创建目标标签的二进制表示
        # 例如：target=2 对应 [1, 1, 0]（类别>0 和 类别>1 为真）
        ordinal_targets = torch.zeros((batch_size, num_thresholds), device=logits.device)
        
        for i in range(batch_size):
            ordinal_targets[i, :severity_true[i]] = 1.0
        
        loss = F.binary_cross_entropy_with_logits(logits, ordinal_targets)
        
        return {'loss_severity': loss}

    
    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        # assignments = self.compute_label_assignment(outputs, targets)
        
        loss = torch.zeros(1)[0].to(outputs['out'].device)
        loss_dict = {}
        loss_intermidiate = {}
        for loss_name, (loss_fn, loss_weight) in self.loss_dict.items():
            
            # loss_intermidiate = loss_fn(outputs, targets, assignments)
            loss_intermidiate = loss_fn(outputs, targets)
            loss_dict.update(loss_intermidiate)
            
            loss += loss_weight * loss_intermidiate[loss_name]
             
        return loss, loss_intermidiate


class OverallCriterion(nn.Module):
    def __init__(self, args):
        super(OverallCriterion, self).__init__()
        self.loss = Criterion(args)

    def forward(self, output, target) -> Dict:
        loss_dict = {}
        loss, loss_dict = self.loss(output, target)
        
        loss_dict.update(loss_dict)
        
        return loss, loss_dict
    