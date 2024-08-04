import torch
import time
import logging
import numpy as np
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)

from dynasiam.train_utils.dice_coefficient_loss import ConfusionMatrix
from dynasiam.train_utils.dice_coefficient_loss import dice_loss, structure_loss
from dynasiam.train_utils.distributed_utils import MetricLogger, SmoothedValue


def train_one_epoch(model,
                    optimizer,
                    train_loader,
                    writer,
                    epoch,
                    total_epochs,
                    num_classes,
                    lr_schedule,
                    print_freq=10,
                    ):
    global train_msg
    model.train()

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    iter_per_epoch = len(train_loader)

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], device="cuda")
    else:
        loss_weight = None

    train_confmat = ConfusionMatrix(num_classes=num_classes)

    b = time.time()

    i = 0
    for normal, surface, mucosal, tone, target in metric_logger.log_every(train_loader, print_freq, header):
        step = (i + 1) + epoch * iter_per_epoch
        normal = normal.cuda()
        surface = surface.cuda()
        mucosal = mucosal.cuda()
        tone = tone.cuda()
        target = target.cuda()

        fuse_pred, sep_preds, intra_preds, prm_preds = model(normal, surface, mucosal, tone)

        fuse_structure_loss = structure_loss(fuse_pred, target, loss_weight)
        fuse_dice_loss = dice_loss(fuse_pred, target)
        fuse_total_loss = fuse_structure_loss + fuse_dice_loss

        sep_structure_loss = torch.zeros(1).cuda().float()
        sep_dice_loss = torch.zeros(1).cuda().float()
        for sep_pred in sep_preds:
            sep_structure_loss += structure_loss(sep_pred, target, loss_weight)
            sep_dice_loss += dice_loss(sep_pred, target)
        sep_total_loss = sep_structure_loss + sep_dice_loss

        intra_structure_loss = torch.zeros(1).cuda().float()
        intra_dice_loss = torch.zeros(1).cuda().float()
        for intra_pred in intra_preds:
            intra_structure_loss += structure_loss(intra_pred, target, loss_weight)
            intra_dice_loss += dice_loss(intra_pred, target)
        intra_total_loss = intra_structure_loss + intra_dice_loss

        prm_structure_loss = torch.zeros(1).cuda().float()
        prm_dice_loss = torch.zeros(1).cuda().float()
        for prm_pred in prm_preds:
            prm_structure_loss += structure_loss(prm_pred, target, loss_weight)
            prm_dice_loss += dice_loss(prm_pred, target)
        prm_total_loss = prm_structure_loss + prm_dice_loss

        total_loss = fuse_total_loss + sep_total_loss + intra_total_loss + prm_total_loss

        train_confmat.get_confusion_matrix(pred_label=fuse_pred.argmax(1).flatten(),
                                           label=target.flatten())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_schedule.step()
        i += 1

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=total_loss.item(), lr=lr)

        writer.add_scalar('total_loss', total_loss.item(), global_step=step)
        writer.add_scalar('fuse_structure_loss', fuse_structure_loss.item(), global_step=step)
        writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
        writer.add_scalar('sep_structure_loss', sep_structure_loss.item(), global_step=step)
        writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
        writer.add_scalar('intra_structure_loss', intra_structure_loss.item(), global_step=step)
        writer.add_scalar('intra_dice_loss', intra_dice_loss.item(), global_step=step)
        writer.add_scalar('prm_structure_loss', prm_structure_loss.item(), global_step=step)
        writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)

        train_msg = 'Epoch {}/{}, Total_loss {:.4f},\n'.format((epoch + 1), total_epochs, total_loss.item())
        train_msg += 'fuse_total_loss:{:.4f}\n'.format(fuse_total_loss.item())
        train_msg += 'fuse_structure_loss:{:.4f}, fuse_dice_loss:{:.4f},'.format(fuse_structure_loss.item(),
                                                                                 fuse_dice_loss.item())
        train_msg += 'sep_structure_loss:{:.4f}, sep_dice_loss:{:.4f},'.format(sep_structure_loss.item(),
                                                                               sep_dice_loss.item())
        train_msg += 'intra_structure_loss:{:.4f}, intra_dice_loss:{:.4f},'.format(intra_structure_loss.item(),
                                                                               intra_dice_loss.item())
        train_msg += 'prm_structure_loss:{:.4f}, prm_dice_loss:{:.4f},\n\n'.format(prm_structure_loss.item(),
                                                                                   prm_dice_loss.item())
        train_msg += 'train_evaluating_indicator:\n'
        train_msg += str(train_confmat)

        logging.info(train_msg)

    logging.info('normal time per epoch: {}'.format(time.time() - b))

    return train_msg


def evaluate(model,
             valid_loader,
             num_classes,
             is_valid = True):
    model.eval()
    test_confmat = ConfusionMatrix(num_classes=num_classes)

    metric_logger = MetricLogger(delimiter=" ")
    if is_valid:
        header = 'Evaluate：'
    else:
        header = 'Test:'

    with torch.no_grad():
        for normal, surface, mucosal, tone, target in metric_logger.log_every(valid_loader, 10, header):
            normal = normal.cuda()
            surface = surface.cuda()
            mucosal = mucosal.cuda()
            tone = tone.cuda()
            target = target.cuda()

            output = model(normal, surface, mucosal, tone)

            test_confmat.get_confusion_matrix(pred_label=output[0].argmax(1).flatten(),
                                              label=target.flatten())

            loss_weight = torch.as_tensor([1.0, 2.0], device="cuda")
            fuse_structure_loss = structure_loss(output[0], target, loss_weight)
            fuse_dice_loss = dice_loss(output[0], target)
            fuse_total_loss = fuse_structure_loss + fuse_dice_loss

            if is_valid:
                test_msg = 'valid_evaluating_indicator:\n'
            else:
                test_msg = 'test_evaluating_indicator:\n'

            test_msg += 'fuse_total_loss:{:.4f}\n'.format(fuse_total_loss.item())
            test_msg += str(test_confmat)
            logging.info(test_msg)

    return test_msg
