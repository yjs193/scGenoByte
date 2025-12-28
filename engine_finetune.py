import math
import sys
from scipy import stats
from typing import Iterable, Optional
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from timm.data import Mixup
from timm.utils import accuracy
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
def train_one_epoch(model: torch.nn.Module,
                    criterion_class: torch.nn.Module, criterion_recon: torch.nn.Module,  # <-- 接收两个 criterion
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, wandb_logger=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_class', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_recon', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    for data_iter_step, (samples, targets_class, targets_pathway) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets_class = targets_class.to(device, non_blocking=True)
        targets_pathway = targets_pathway.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            outputs_class, outputs_recon = model(samples)
            loss_class = criterion_class(outputs_class, targets_class)
            loss_recon = criterion_recon(outputs_recon, targets_pathway)
            loss = loss_class + args.pathway_loss_beta * loss_recon
            softmax = nn.Softmax(dim=-1)
            logit = softmax(outputs_class)  # <-- 使用 outputs_class
            predLabel = logit.argmax(dim=-1)

        acc, _ = accuracy(outputs_class, targets_class, topk=(1, 5))  # <-- 使用 outputs_class

        loss_value = loss.item()
        loss_class_value = loss_class.item()
        loss_recon_value = loss_recon.item()
        f1 = f1_score(np.array(targets_class.cpu()), np.array(predLabel.cpu()), average='macro')  # <-- 使用 targets_class


        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_class=loss_class_value)
        metric_logger.update(loss_recon=loss_recon_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        acc_reduce = misc.all_reduce_mean(acc.item())
        f1_reduce = misc.all_reduce_mean(f1)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_class_reduce = misc.all_reduce_mean(loss_class_value)
        loss_recon_reduce = misc.all_reduce_mean(loss_recon_value)
        if wandb_logger is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            prefix = f"{args.dataset_name}/"
            dict = {
                prefix + "epoch_1000x": epoch_1000x,
                prefix + "train_loss": loss_value_reduce,
                prefix + "train_loss_class": loss_class_reduce,  # <-- 新增
                prefix + "train_loss_recon": loss_recon_reduce,  # <-- 新增
                prefix + "train_acc": acc_reduce,
                prefix + "train_f1score": f1_reduce,
                prefix + "lr": max_lr,
            }
            wandb_logger.log(dict, step=epoch_1000x)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, criterion_class, criterion_recon, device, args):
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_class', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_recon', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Test:'

    model.eval()
    truths = []
    pred_finals = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        cells = batch[0]
        targets_class = batch[1]
        targets_pathway = batch[2]

        cells = cells.to(device, non_blocking=True)
        targets_class = targets_class.to(device, non_blocking=True)
        targets_pathway = targets_pathway.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            outputs_class, outputs_recon = model(cells)

            loss_class = criterion_class(outputs_class, targets_class)  # 细胞注释损失
            loss_recon = criterion_recon(outputs_recon, targets_pathway)  # 通路预测损失

            loss = loss_class + args.pathway_loss_beta * loss_recon

            # Acc 和 F1 仅基于分类任务
            softmax = nn.Softmax(dim=-1)
            logit = softmax(outputs_class)
            pred_final = logit.argmax(dim=-1)

            truths.extend(targets_class.detach().cpu().numpy().tolist())  # <-- 使用 targets_class
            pred_finals.extend(pred_final.detach().cpu().numpy().tolist())

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_class=loss_class.item())
        metric_logger.update(loss_recon=loss_recon.item())

    print(len(truths))
    print(len(pred_finals))
    acc = accuracy_score(truths, pred_finals)
    f1 = f1_score(truths, pred_finals, average='macro')

    metric_logger.synchronize_between_processes()

    print(
        '* Acc {top1:.3f} F1 score {f1:.3f} Total Loss {losses.global_avg:.3f} Class Loss {loss_class.global_avg:.3f} Recon Loss {loss_recon.global_avg:.3f}'
        .format(top1=acc, f1=f1,
                losses=metric_logger.loss,
                loss_class=metric_logger.loss_class,
                loss_recon=metric_logger.loss_recon))

    model.train(True)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, f1, acc, truths, pred_finals
