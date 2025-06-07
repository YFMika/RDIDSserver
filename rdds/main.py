import argparse, os
import torch
from torch import nn

from rdds.dataset.rdds import getData
from rdds.models.detection import Detection

from engine import do_train, do_eval
from rdds.utils.io import resume_if_possible

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
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, 
        help="Max L2 norm of the gradient"
    )
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=200, type=int)
    parser.add_argument("--eval_every_iteration", default=300, type=int)

    parser.add_argument("--gpu", default='0', type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default="results", type=str)
    parser.add_argument("--log_every", default=10, type=int)
    
    args = parser.parse_args()
    
    return args


def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def main(args):
    if args.checkpoint_dir is not None:
        pass
    else:
        raise AssertionError(
            'Checkpoint_dir should be presented!'
        )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
      
    datasets, dataloaders = getData(args)
    model = Detection(args).cuda()
    model.apply(xavier_init)

    optimizer = torch.optim.AdamW(
        filter(lambda params: params.requires_grad, model.parameters()), 
        lr=args.base_lr, 
        weight_decay=args.weight_decay
    )
        
    print('certain parameters are not trained:')
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            print(name)
        
    loaded_epoch, best_val_metrics = resume_if_possible(
        args.checkpoint_dir, model, optimizer
    )
    args.start_epoch = loaded_epoch + 1
    do_train(
        args,
        model,
        optimizer,
        dataloaders,
        best_val_metrics,
    )


if __name__ == "__main__":
    args = make_args_parser()
    
    print(f"Called with args: {args}")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)