from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
import torch.nn.functional as F
from typing import Union, List
import torch.distributed as dist


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


def ssim_tensor_safe(pred: Union[torch.Tensor, List[torch.Tensor]],
                     target: Union[torch.Tensor, List[torch.Tensor]],
                     window_size=11,
                     data_range = 4096,
                     K1=0.01,
                     K2=0.03,
                     eps=1e-8,
                     reduction='mean',
                     ddp: bool = False,
                     layer_wise_weights: Union[List[float], None] = None):
    """
    SSIM for 2D/3D tensors or list of tensors. DDP-aware.
    
    Args:
        pred, target: Tensor [B, 1, H, W] / [B, 1, D, H, W] or list of such tensors
        reduction: 'mean' or 'none'
        ddp: whether in distributed mode (DDP), uses all_reduce
        layer_wise_weights: optional weights for each layer in list
    Returns:
        SSIM score (scalar if reduction='mean', tensor of shape [B] if 'none')
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)
    if isinstance(pred, torch.Tensor):
        pred = [pred]
    if isinstance(target, torch.Tensor):
        target = [target]

    assert isinstance(pred, list) and isinstance(target, list)
    assert len(pred) == len(target)
    C1 = (K1*data_range) ** 2
    C2 = (K2*data_range) ** 2
    
    if layer_wise_weights is None:
        layer_wise_weights = [1.0 / len(pred)] * len(pred)
    else:
        assert len(layer_wise_weights) == len(pred), "Mismatch with number of prediction layers"
        layer_wise_weights = torch.tensor(layer_wise_weights, device=pred[0].device, dtype=pred[0].dtype)
        layer_wise_weights = layer_wise_weights / layer_wise_weights.sum()  # normalize

    ssim_all = []
    for i, (p, t) in enumerate(zip(pred, target)):
        assert p.shape == t.shape and p.ndim in [4, 5]

        dims = [2, 3] if p.ndim == 4 else [2, 3, 4]
        conv = F.conv2d if p.ndim == 4 else F.conv3d

        channel = p.size(1)
        kernel = torch.ones(1, 1, *([window_size] * len(dims)), device=p.device, dtype=p.dtype)
        kernel = kernel / kernel.numel()
        kernel = kernel.expand(channel, 1, *([-1] * len(dims)))

        pad = window_size // 2
        pad_shape = [pad] * len(dims) * 2
        p = F.pad(p, pad_shape, mode='replicate')
        t = F.pad(t, pad_shape, mode='replicate')

        mu1 = conv(p, kernel, groups=channel)
        mu2 = conv(t, kernel, groups=channel)

        mu1_sq, mu2_sq = mu1 * mu1, mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = conv(p * p, kernel, groups=channel) - mu1_sq
        sigma2_sq = conv(t * t, kernel, groups=channel) - mu2_sq
        sigma12 = conv(p * t, kernel, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)

        ssim_val = ssim_map.mean(dim=dims)  # [B]
        ssim_all.append(ssim_val * layer_wise_weights[i])

    ssim_all = torch.stack(ssim_all, dim=0).sum(dim=0)  # shape: [B]

    if ddp and dist.is_available() and dist.is_initialized():
        dist.all_reduce(ssim_all, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        ssim_all = ssim_all / world_size

    if reduction == 'mean':
        return ssim_all.mean()
    elif reduction == 'none':
        return ssim_all
    else:
        raise ValueError("reduction must be 'mean' or 'none'")


def psnr_tensor_flexible(preds: Union[torch.Tensor, List[torch.Tensor]],
                         targets: Union[torch.Tensor, List[torch.Tensor]],
                         data_range: Union[float, None] = None,
                         eps: float = 1e-8,
                         ddp: bool = False,
                         reduction: str = 'mean',
                         layer_wise_weights: Union[List[float], None] = None):
    """
    Computes PSNR over tensor or list of tensors (2D/3D), with optional DDP support.

    Args:
        preds: Tensor or list of [B, 1, H, W] or [B, 1, D, H, W]
        targets: Same structure as preds
        data_range: Range for normalization, if None, computed from each target
        ddp: if True and DDP initialized, performs all_reduce
        reduction: 'mean' (scalar) or 'none' (tensor per sample)
        layer_wise_weights: optional weights for each pred-target pair

    Returns:
        PSNR value (scalar or tensor)
    """
    if isinstance(preds, torch.Tensor):
        preds = [preds]
    if isinstance(targets, torch.Tensor):
        targets = [targets]

    assert len(preds) == len(targets), "Mismatch in number of pred/target"

    if layer_wise_weights is None:
        layer_wise_weights = [1.0 / len(preds)] * len(preds)
    else:
        assert len(layer_wise_weights) == len(preds), "Mismatch in layer weights"
        layer_wise_weights = torch.tensor(layer_wise_weights, device=preds[0].device)
        layer_wise_weights = layer_wise_weights / layer_wise_weights.sum()

    psnr_vals = []
    for i, (pred, target) in enumerate(zip(preds, targets)):
        pred = pred.float()
        target = target.float()

        if data_range is None:
            range_val = (target.max() - target.min()).clamp(min=eps)
        else:
            range_val = torch.tensor(data_range, device=target.device)

        mse = F.mse_loss(pred, target, reduction='mean')
        mse = torch.clamp(mse, min=eps)
        psnr = 10.0 * torch.log10(range_val ** 2 / mse)
        psnr_vals.append(psnr * layer_wise_weights[i])

    psnr_tensor = torch.stack(psnr_vals).sum()

    if ddp and dist.is_available() and dist.is_initialized():
        dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        psnr_tensor = psnr_tensor / world_size

    if reduction == 'mean':
        return psnr_tensor
    elif reduction == 'none':
        return torch.stack(psnr_vals)
    else:
        raise ValueError("reduction must be 'mean' or 'none'")

if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
