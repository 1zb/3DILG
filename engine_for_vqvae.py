import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.utils import ModelEma

import utils

def train_batch(model, surface, points, labels, criterion):
    outputs, z_e_x, z_q_x, sigma, loss_commit, perplexity = model(surface, points)

    loss_vol = criterion(outputs[:, :1024], labels[:, :1024])
    loss_near = criterion(outputs[:, 1024:], labels[:, 1024:])

    loss_sigma = sigma.mean()

    loss = loss_vol + 0.1 * loss_near + loss_commit + 0.0001 * loss_sigma 

    return loss, outputs, loss_vol.item(), loss_near.item(), loss_commit.item(), loss_sigma.item()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (points, labels, surface, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        surface = surface.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if loss_scaler is None:
            raise NotImplementedError
        else:
            with torch.cuda.amp.autocast(enabled=True):

                loss, output, loss_vol, loss_near, loss_commit, loss_sigma = train_batch(model, surface, points, labels, criterion)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            raise NotImplementedError
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

                    
        torch.cuda.synchronize()


        pred = torch.zeros_like(output[:, :1024])
        pred[output[:, :1024]>=0] = 1

        accuracy = (pred==labels[:, :1024]).float().sum(dim=1) / labels[:, :1024].shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels[:, :1024]).sum(dim=1)
        union = (pred + labels[:, :1024]).gt(0).sum(dim=1) + 1e-5
        iou = intersection * 1.0 / union
        iou = iou.mean()

        metric_logger.update(loss=loss_value)
        metric_logger.update(iou=iou)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(loss_vol=loss_vol)
        metric_logger.update(loss_near=loss_near)
        metric_logger.update(loss_commit=loss_commit)
        metric_logger.update(loss_sigma=loss_sigma)


        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(loss_vol=loss_vol, head="opt")
            log_writer.update(loss_near=loss_near, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 200, header):
        points, labels, surface, _ = batch
        surface = surface.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            N = 100000

            output, _, _, _, _, perplexity = model(surface, points)
            loss = criterion(output, labels)


        pred = torch.zeros_like(output)
        pred[output>=0] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()

        batch_size = surface.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['iou'].update(iou.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    print('* iou{iou.global_avg:.4f} loss {losses.global_avg:.3f}'
          .format(iou=metric_logger.iou, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}