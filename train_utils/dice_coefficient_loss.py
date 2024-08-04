import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def dice_loss(output: torch.Tensor,
              target: torch.Tensor,
              num_classes: int = 2,
              ):
    smooth = 1
    p = 2
    output = F.softmax(output, dim=1)
    predict = output.contiguous().view(output.shape[0], -1)

    dice_target = target.clone()
    dice_target = F.one_hot(dice_target.to(torch.int64), num_classes).float()
    dice_target = dice_target.permute(0, 3, 1, 2)

    valid_mask = torch.ones_like(dice_target)
    target = dice_target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth

    loss = 1 - num / den

    return loss.mean()


def structure_loss(output: torch.Tensor,
                   target: torch.Tensor,
                   loss_weight: torch.Tensor,
                   num_classes: int = 2):
    oh_target = target.clone()
    bce = F.cross_entropy(output, oh_target, weight=loss_weight, reduction='mean')

    # output = torch.softmax(output, dim=1)
    # output = output.contiguous()
    # inter = (output * oh_target).sum()
    # union = (output + oh_target).sum()
    # iou = 1 - (inter + 1) / (union - inter + 1)
    output = torch.softmax(output, dim=1).contiguous().view(-1, num_classes)
    alpha = 0.25
    gamma = 2
    epsilon = 1e-10
    target = target.view(-1, 1)
    idx = target.cpu().long()
    one_hot_key = torch.FloatTensor(target.size(0), num_classes).zero_()
    one_hot_key = one_hot_key.scatter_(1, idx, 1)
    pt = (one_hot_key * output.cpu()).sum(1) + epsilon
    logpt = pt.log()
    loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
    focal = loss.mean()

    return (bce + focal).mean()


class ConfusionMatrix(object):
    def __init__(self,
                 num_classes: int):
        self.num_classes = num_classes
        self.total_mat = None

    def get_confusion_matrix(self, pred_label, label):
        n = self.num_classes  # 2

        if self.total_mat is None:
            self.total_mat = np.zeros((n, n), dtype=np.float32)  # 2*2

        with torch.no_grad():
            mask = (label >= 0) & (label < n)   #224*224[bool]

            pred_label = pred_label[mask]
            label = label[mask]

            inds = n * label + pred_label

            mat = np.bincount(inds.cpu(), minlength=n ** 2)  # 指定 minlength=4，现在的索引值为0->gas  # [TN, FP, FN, TP]
            mat = mat.reshape(n, n)  # [[TN, FP],[FN, TP]]
            # print(mat)

            self.total_mat += mat

    def compute(self, beta=1):
        # print(self.total_mat)
        all_acc = np.diag(self.total_mat).sum() / self.total_mat.sum()

        # np.diag(self.total_mat)：取对角线上的数
        acc = np.diag(self.total_mat) / self.total_mat.sum(axis=0)

        iou = np.diag(self.total_mat) / (self.total_mat.sum(axis=1) + self.total_mat.sum(axis=0) - np.diag(self.total_mat))

        dice = (2 * np.diag(self.total_mat) / (self.total_mat.sum(axis=1) + self.total_mat.sum(axis=0)))[1]

        specificity = (np.diag(self.total_mat) / self.total_mat.sum(axis=1))[0]
        recall = (np.diag(self.total_mat) / self.total_mat.sum(axis=1))[1]

        precision = (np.diag(self.total_mat) / self.total_mat.sum(axis=0))[1]

        f1_score = (2 * precision * recall) / (precision + recall)
        f2_score = (5 * precision * recall) / (4 * precision + recall)
        g_score = np.sqrt(precision * recall)

        return all_acc, acc, iou, dice, specificity, recall, precision, f1_score, f2_score, g_score

    def reset(self):
        if self.total_mat is not None:
            self.total_mat.zero_()

    def __str__(self):
        all_acc, acc, iou, dice, specificity, recall, precision, f1_score, f2_score, g_score = self.compute()
        return (
            'all_acc: {:.4f}\n'
            'mean acc: {:.4f}\n'
            'iou:{}\n'
            'mean iou:{:.4f}\n'
            'dice:{:.4f}\n'
            'specificity:{:.4f}\n'
            'recall:{:.4f}\n'
            'precision:{:.4f}\n'
            'F1score:{:.4f}\n'
            'F2score:{:.4f}\n'
            'Gscore:{:.4f}\n'
        ).format(
            all_acc.item() * 100,
            acc.mean().item() * 100,
            ['{:.4f}'.format(i) for i in (iou * 100).tolist()],
            iou.mean().item() * 100,
            dice.item() * 100,
            specificity.item() * 100,
            recall.item() * 100,
            precision.item() * 100,
            f1_score.item() * 100,
            f2_score.item() * 100,
            g_score.item() * 100,
        )
