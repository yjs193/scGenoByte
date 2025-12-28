
import math
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, wandb_logger=None, args=None, max_norm=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_recon', misc.SmoothedValue(window_size=20, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_sim', misc.SmoothedValue(window_size=20, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, protein_targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)  #
        protein_targets = protein_targets.to(device, non_blocking=True)  # <--- 新增

        with torch.cuda.amp.autocast():
            loss_dict = model(samples, protein_targets, mask_ratio=args.mask_ratio)

        loss = loss_dict["loss"]  # 总损失
        loss_value = loss.item()

        loss_recon_value = loss_dict["loss_recon"].item()
        loss_sim_value = loss_dict["loss_sim"].item()

        loss /= accum_iter  #
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)  #

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()  #

        torch.cuda.synchronize()  #

        metric_logger.update(loss=loss_value)  #
        metric_logger.update(loss_recon=loss_recon_value)
        metric_logger.update(loss_sim=loss_sim_value)

        lr = optimizer.param_groups[0]["lr"]  #
        metric_logger.update(lr=lr)  #

        loss_value_reduce = misc.all_reduce_mean(loss_value)  #
        loss_recon_reduce = misc.all_reduce_mean(loss_recon_value)
        loss_sim_reduce = misc.all_reduce_mean(loss_sim_value)

        if wandb_logger and (data_iter_step + 1) % accum_iter == 0:  #
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  #

            wandb_logger.log({
                "pretrain/train_loss": loss_value_reduce,
                "pretrain/train_loss_recon": loss_recon_reduce,
                "pretrain/train_loss_sim": loss_sim_reduce,
                "pretrain/lr": lr
            }, step=epoch_1000x)  #

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()  #
    print("Averaged stats:", metric_logger)  #
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  #

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for (cells, _) in metric_logger.log_every(data_loader, 10, header):

        cells = cells.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss = model(cells, mask_ratio=args.mask_ratio)
        metric_logger.update(loss=loss.item())
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f} '.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
