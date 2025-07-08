import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F

def ssim3d(x, y, window_size=11, data_range=4096, K1=0.01, K2=0.03):

    C1 = K1 / data_range ** 2
    C2 = K2 / data_range ** 2

    pad = window_size // 2
    mu_x = F.avg_pool3d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool3d(y, window_size, stride=1, padding=pad)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool3d(x * x, window_size, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool3d(y * y, window_size, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool3d(x * y, window_size, stride=1, padding=pad) - mu_xy

    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim = ssim_n / (ssim_d + 1e-8)
    return torch.clamp((1 - ssim) / 2, 0, 1)  # SSIM loss in [0,1]

class SSIM_L1Loss(nn.Module):
    def __init__(self, alpha=0.85, window_size=11, weights=None):
        """
        alpha: weighted coefficients of SSIM and L1 loss
        weights: optional, list [float], representing the loss weight of each layer
        """
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.weights = weights  # None represents uniform average

    def forward(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = [pred]
        if isinstance(target, torch.Tensor):
            target = [target]
            if len(target) != len(pred):
                for ii, item in enumerate(pred):
                    if ii == 1:
                        target.append(target[-1])
                    if ii > 1:
                        target.append(F.interpolate(target[-1], size=item.shape[2:], mode='trilinear', align_corners=True))

        assert len(pred) == len(target), "Mismatch in number of outputs"

        num_outputs = len(pred)
        weights = self.weights or [1.0] * num_outputs
        weights = torch.tensor(weights, dtype=pred[0].dtype, device=pred[0].device)
        weights = weights / weights.sum()  # Normalized weights
        
        total_loss = 0.0
        for i, (p, t) in enumerate(zip(pred, target)):
            # l1 = F.l1_loss(p, t)
            l1 = F.smooth_l1_loss(p, t, beta=1.0, reduction='mean')
            ssim = ssim3d(p, t, window_size=self.window_size).mean()
            loss = self.alpha * ssim + (1 - self.alpha) * l1
            total_loss += weights[i] * loss

        return total_loss

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
