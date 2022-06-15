
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.utils import ModelEma

import utils

def sort(centers_quantized, encodings):
    ind3 = torch.argsort(centers_quantized[:, :, 2], dim=1)
    centers_quantized = torch.gather(centers_quantized, 1, ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind3)

    _, ind2 = torch.sort(centers_quantized[:, :, 1], dim=1, stable=True)
    centers_quantized = torch.gather(centers_quantized, 1, ind2[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind2)

    _, ind1 = torch.sort(centers_quantized[:, :, 0], dim=1, stable=True)
    centers_quantized = torch.gather(centers_quantized, 1, ind1[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind1)
    return centers_quantized, encodings

def train_batch(model, vqvae, surface, categories, criterion):
    with torch.no_grad():
        _, _, centers_quantized, _, _, encodings = vqvae.encode(surface)

    centers_quantized, encodings = sort(centers_quantized, encodings)

    x_logits, y_logits, z_logits, latent_logits = model(centers_quantized, encodings, categories)

    loss_x = criterion(x_logits, centers_quantized[:, :, 0])
    loss_y = criterion(y_logits, centers_quantized[:, :, 1])
    loss_z = criterion(z_logits, centers_quantized[:, :, 2])
    loss_latent = criterion(latent_logits, encodings)
    loss = loss_x + loss_y + loss_z + loss_latent

    return loss, loss_x.item(), loss_y.item(), loss_z.item(), loss_latent.item(), 

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, vqvae: torch.nn.Module,
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

    for data_iter_step, (_, _, surface, categories) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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
        categories = categories.to(device, non_blocking=True)

        if loss_scaler is None:
            raise NotImplementedError
        else:
            with torch.cuda.amp.autocast():
                loss, loss_x, loss_y, loss_z, loss_latent = train_batch(model, vqvae, surface, categories, criterion)
        
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

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(loss_x=loss_x)
        metric_logger.update(loss_y=loss_y)
        metric_logger.update(loss_z=loss_z)
        metric_logger.update(loss_latent=loss_latent)

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

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, vqvae, device):

    criterion = torch.nn.NLLLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 1000, header):
        _, _, surface, categories = batch
        surface = surface.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():

            with torch.no_grad():
                _, _, centers_quantized, _, _, encodings = vqvae.encode(surface)

            centers_quantized, encodings = sort(centers_quantized, encodings)

            x_logits, y_logits, z_logits, latent_logits = model(centers_quantized, encodings, categories)

            loss_x = criterion(x_logits, centers_quantized[:, :, 0])
            loss_y = criterion(y_logits, centers_quantized[:, :, 1])
            loss_z = criterion(z_logits, centers_quantized[:, :, 2])

            loss_latent = criterion(latent_logits, encodings)
            loss = loss_x + loss_y + loss_z + loss_latent

        batch_size = surface.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_x=loss_x.item())
        metric_logger.update(loss_y=loss_y.item())
        metric_logger.update(loss_z=loss_z.item())
        metric_logger.update(loss_latent=loss_latent.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f} '
          .format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
