#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn


def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv


def l1_loss_weighted(network_output, gt, drop_region=20):
    """
    Weighted L1 loss with an exponential drop-off near the width edges.

    Args:
        network_output (torch.Tensor): Model output, shape [B, H, W].
        gt (torch.Tensor): Ground truth, shape [B, H, W].
        drop_region (int): Number of pixels near the edge to drop weight exponentially.

    Returns:
        torch.Tensor: Weighted L1 loss.
    """
    assert network_output.shape == gt.shape, "Input and ground truth must have the same shape"

    B, H, W = network_output.shape

    # Create a weight vector for the width dimension
    weight = torch.ones(W)
    decay_region = torch.arange(drop_region)

    # Exponential decay near the edges
    decay_left = torch.exp(-decay_region / (drop_region / 5))  # Adjust scale for exponential drop
    decay_right = torch.exp(-(drop_region - decay_region - 1) / (drop_region / 5))

    weight[:drop_region] = decay_left
    weight[-drop_region:] = decay_right

    # Reshape weight to [1, 1, W] to broadcast across batch and height dimensions
    weight = weight.view(1, 1, W).to(network_output.device)

    # Apply the weight
    weighted_l1 = torch.abs(network_output - gt) * weight

    return weighted_l1.mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def normalized_cross_correlation(img1, img2):
    # Flatten images
    img1_flat = img1.view(-1)
    img2_flat = img2.view(-1)

    # Normalize
    img1_mean = img1_flat - img1_flat.mean()
    img2_mean = img2_flat - img2_flat.mean()

    # Compute cross-correlation
    correlation = torch.dot(img1_mean, img2_mean) / (torch.norm(img1_mean) * torch.norm(img2_mean))

    return correlation
