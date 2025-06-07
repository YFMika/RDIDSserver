import os, time, math, sys
import torch
import datetime
from rdds.utils.misc import SmoothedValue
from rdds.utils.io import save_checkpoint

import torch.nn.functional as F

class Logger:
    def __init__(self, args):
        self.logger = open(os.path.join(args.checkpoint_dir, 'logger.out'), 'a')
    def __call__(self, info_str):
        self.logger.write(info_str + "\n")
        self.logger.flush()
        print(info_str)


def do_train(
    args,
    model,
    optimizer,
    dataloaders,
    best_val_metrics=dict()
):    
    logout = Logger(args)
    logout(f"call with args: {args}")
    logout(f"{model}")
    
    curr_iter = args.start_epoch * len(dataloaders['train'])
    max_iters = args.max_epoch * len(dataloaders['train'])
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    for curr_epoch in range(args.start_epoch, args.max_epoch):
        
        for batch_idx, batch_data_label in enumerate(dataloaders['train']):
            
            curr_time = time.time()
            
            curr_iter = curr_epoch * len(dataloaders['train']) + batch_idx
            curr_lr = args.base_lr
            for key in batch_data_label:
                if isinstance(batch_data_label[key], dict):
                    for keys in batch_data_label[key]:
                        batch_data_label[key][keys] = batch_data_label[key][keys].to(net_device)
                else:
                    batch_data_label[key] = batch_data_label[key].to(net_device)
    
            # Forward pass
            optimizer.zero_grad()
    
            outputs = model(batch_data_label['image'], batch_data_label['label'], is_eval=False)
            loss = outputs['loss']
    
            if not math.isfinite(loss.item()):
                logout("Loss in not finite. Training will be stopped.")
                sys.exit(1)
    
            loss.backward()
            if args.clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()
    
            time_delta.update(time.time() - curr_time)
            loss_avg.update(loss.item())
    
            # logging
            if curr_iter % args.log_every == 0:
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                eta_seconds = (max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logout(
                    f"Epoch [{curr_epoch}/{args.max_epoch}]; "
                    f"Iter [{curr_iter}/{max_iters}]; "
                    f"Loss {loss_avg.avg:0.2f}; "
                    f"LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; "
                    f"ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                )
            
            # eval
            if (curr_iter + 1) % args.eval_every_iteration == 0:
                eval_metrics = do_eval(
                    args,
                    curr_epoch,
                    model,
                    dataloaders['test'],
                    logout,
                )
                model.train()
                if not best_val_metrics or (
                    best_val_metrics['classification_accuracy'] + best_val_metrics['severity_accuracy'] 
                    < eval_metrics['classification_accuracy'] + eval_metrics['severity_accuracy']
                ):
                    best_val_metrics = eval_metrics
                    filename = "checkpoint_best.pth"
                    save_checkpoint(
                        args.checkpoint_dir,
                        model,
                        optimizer,
                        curr_epoch,
                        args,
                        best_val_metrics,
                        filename="checkpoint_best.pth",
                    )
                    logout(
                        f"Epoch [{curr_epoch}/{args.max_epoch}] "
                        f"saved current best val checkpoint at {filename}; "
                        f"{'classification_accuracy'} {eval_metrics['classification_accuracy']}; "
                        f"{'severity_accuracy'} {eval_metrics['severity_accuracy']} "
                    )
            # end of an iteration
        # end of an epoch
        save_checkpoint(
            args.checkpoint_dir,
            model,
            optimizer,
            curr_epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )
    # end of training
    do_eval(
        args,
        curr_epoch,
        model,
        dataloaders['test'],
        logout,
    )
    return 


# def do_eval(
#     args,
#     curr_epoch,
#     model,
#     dataset_loader,
#     logout=print,
#     curr_train_iter=-1,
# ):
    
#     # 初始化计算精度的计数器
#     correct = 0
#     total = 0

#     # 设置模型为评估模式
#     model.eval()
#     net_device = next(model.parameters()).device
#     num_batches = len(dataset_loader)
    
#     time_delta = SmoothedValue(window_size=10)
#     epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
#     for curr_iter, batch_data_label in enumerate(dataset_loader):
        
#         curr_time = time.time()
        
#         # 将数据传输到相应设备
#         for key in batch_data_label:
#                 if isinstance(batch_data_label[key], dict):
#                     for keys in batch_data_label[key]:
#                         batch_data_label[key][keys] = batch_data_label[key][keys].to(net_device)
#                 else:
#                     batch_data_label[key] = batch_data_label[key].to(net_device)
        
#         # 获取输入图像和标签
#         images = batch_data_label["image"]
#         labels = batch_data_label["label"]
        
#         # 获取模型的输出（比如分类概率）
#         outputs = model(images, labels, is_eval=True)
        
#         # 对模型输出应用softmax，得到每个类别的概率
#         probs = torch.nn.functional.softmax(outputs['out'], dim=1)  # [B, num_classes]
        
#         # 预测类别是概率最大的类别
#         _, predicted = torch.max(probs, 1)  # [B]
        
#         # 计算正确预测的数量
#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)
        
#         time_delta.update(time.time() - curr_time)

#         # 打印训练日志
#         if curr_iter % args.log_every == 0:
#             mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
#             logout(
#                 f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; "
#                 f"Evaluating on iter: {curr_train_iter}; "
#                 f"Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
#             )

#     # 计算准确率
#     accuracy = correct / total

#     # 记录评估指标
#     eval_metrics = {
#         'accuracy': accuracy
#     }

#     logout("==" * 10)
#     logout(f"Evaluate Epoch [{curr_epoch}/{args.max_epoch}]")
#     logout(f"Accuracy: {accuracy * 100:.2f}%")
#     logout("==" * 10)

#     return eval_metrics
    

def do_eval(
    args,
    curr_epoch,
    model,
    dataset_loader,
    logout=print,
):
    # 初始化计数器（分类和回归分开）
    # 多标签分类：正确预测的标签数 / 总标签数
    class_correct = 0
    class_total = 0
    # 回归任务：均方误差（MSE）和样本数
    severity_correct = 0
    severity_total = 0

    model.eval()
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)
    time_delta = SmoothedValue(window_size=10)
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    with torch.no_grad():
        for curr_iter, batch_data_label in enumerate(dataset_loader):
            curr_time = time.time()

            # 数据移到设备
            for key in batch_data_label:
                if isinstance(batch_data_label[key], dict):
                    for keys in batch_data_label[key]:
                        batch_data_label[key][keys] = batch_data_label[key][keys].to(net_device)
                else:
                    batch_data_label[key] = batch_data_label[key].to(net_device)
            
            images = batch_data_label["image"]
            labels = batch_data_label["label"]
            outputs = model(images, is_eval=True)

            # ---------------------- 处理多类别预测 ----------------------
            class_logits = outputs['classes']
            class_labels = labels['type_label']  # 真实标签（多标签：[B, C] 二进制；单标签：[B] 索引）

            # 应用 sigmoid 并设置阈值（如 0.5）
            probs = torch.sigmoid(class_logits)
            predicted = (probs > 0.5).long()
            # 计算每个样本的正确标签数（汉明距离）
            correct_per_sample = (predicted == class_labels).sum(dim=1)
            class_correct += correct_per_sample.sum().item()
            class_total += class_labels.numel()  # 总标签数（B*C）

            # ---------------------- 处理严重性回归 ----------------------
            
            severity_true = labels['yanzhong_label']
            severity_pred = model.predict(outputs['severity'])  # [B]
        
            # 计算正确预测的数量
            severity_correct += (severity_pred == severity_true).sum().item()
            severity_total += severity_true.size(0)

            time_delta.update(time.time() - curr_time)

            # 打印日志
            if curr_iter % args.log_every == 0:
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                logout(
                    f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; "
                    f"Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
                )

    # ---------------------- 计算最终指标 ----------------------
    # 分类指标
    if class_total == 0:
        classification_accuracy = 0.0
    else:
        classification_accuracy = class_correct / class_total  # 多标签正确率（每个标签的准确率）

    # 回归指标（均方误差 MSE 和 RMSE）
    if severity_total == 0:
        severity_accuracy = 0.0
    else:
        severity_accuracy = severity_correct / severity_total

    # 记录评估结果
    eval_metrics = {
        'classification_accuracy': classification_accuracy,
        'severity_accuracy': severity_accuracy,
    }

    logout("==" * 20)
    logout(f"Evaluate Epoch [{curr_epoch}/{args.max_epoch}]")
    if class_labels.dim() == 1:
        logout(f"单标签分类准确率: {classification_accuracy * 100:.2f}%")
    else:
        logout(f"多标签分类正确率（每个标签）: {classification_accuracy * 100:.2f}%")
    logout(f"严重性分类准确率: {severity_accuracy * 100:.2f}%")
    logout("==" * 20)

    return eval_metrics
