import argparse
import torch
from torch import nn

from rdds.models.detection import Detection
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .utils.io import resume_if_possible, resume_best_if_possible


def make_args_parser():
    parser = argparse.ArgumentParser(
        "SlaytheSpire-AIAgent-Classifcation", 
        add_help=False
    )

    ##### Model #####
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--hidden_channels", default=128, type=int)
    parser.add_argument("--out_channels", default=64, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--shortcut", default=False, type=bool)
    parser.add_argument("--hidden_dims", default=128, type=int)
    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--num_severity", default=4, type=int)
    parser.add_argument("--is_pretrain", default=False, type=bool)
    parser.add_argument("--in_dim", default=351232, type=int)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=1e-7, type=float)
    parser.add_argument("--loss_severity_weight", default=5, type=float)
    parser.add_argument("--loss_classes_weight", default=5, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default="./rdds/results", type=str)
    
    args = parser.parse_args()
    
    return args


def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def test(args=None, image_path=None):
    if not args:
        args = make_args_parser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),                  # 归一化到 [0,1]
        transforms.Normalize(                   # 标准化
            mean=[0.5296, 0.5323, 0.5292],
            std=[0.2044, 0.2043, 0.2183]
        )
    ])

    if args.checkpoint_dir is not None:
        pass
    else:
        raise AssertionError(
            'Checkpoint_dir should be presented!'
        )
    
    args = make_args_parser()

    model = Detection(args, is_eval=True)

    optimizer = torch.optim.AdamW(
        filter(lambda params: params.requires_grad, model.parameters()), 
        lr=args.base_lr, 
        weight_decay=args.weight_decay
    )   
    _, best_val_metrics = resume_best_if_possible(
        args.checkpoint_dir, model, optimizer
    )
    
    if image_path is None:
        images = Image.open("./test.jpg").convert("RGB")
    else:
        images = Image.open(image_path).convert("RGB")

    model.eval()
    model = model.to(device) 

    images = transform(images)
    images = images.to(device)
    images = images.unsqueeze(0)     
    
    outputs = model(images, is_eval=True)

    class_logits = outputs['classes']

    # 应用 sigmoid 并设置阈值
    probs = torch.sigmoid(class_logits)
    predicted = (probs > 0.5).long()

    severity_pred = outputs['severity'].squeeze()
    severity_pred = F.softmax(severity_pred, dim=0)
    max_prob, severity_pred = torch.max(severity_pred, dim=0)

    print(f"预测种类: {predicted}, 预测严重程度: {severity_pred}, 置信度: {max_prob}")

    return predicted, severity_pred

if __name__ == "__main__":
    args = make_args_parser()
    test(args)