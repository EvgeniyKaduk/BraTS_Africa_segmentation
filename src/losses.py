# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

def extract_boundary(mask):
    """
    mask: [B,1,D,H,W] бинарная
    return: boundary map
    """
    dilated = F.max_pool3d(mask.float(), 3, stride=1, padding=1)
    eroded  = -F.max_pool3d(-mask.float(), 3, stride=1, padding=1)
    boundary = (dilated - eroded) > 0
    return boundary.float()

class FastBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)

        # берём опухоль целиком (WT)
        tumor_prob = probs[:,1] + probs[:,2] + probs[:,3]
        tumor_prob = tumor_prob.unsqueeze(1)

        tumor_gt = (target > 0).unsqueeze(1).float()

        boundary_gt = extract_boundary(tumor_gt)

        loss = ((1 - tumor_prob) * boundary_gt).mean()

        return loss

class RegionDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def dice(self, p, g):
        inter = (p * g).sum(dim=(2,3,4))
        union = p.sum(dim=(2,3,4)) + g.sum(dim=(2,3,4))
        return (2*inter + self.eps) / (union + self.eps)

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        # one-hot
        targets_oh = F.one_hot(targets, 4).permute(0,4,1,2,3).float()

        # --- регионы ---
        wt_p = probs[:,1:].sum(1, keepdim=True)
        wt_g = targets_oh[:,1:].sum(1, keepdim=True)

        tc_p = probs[:,[1,3]].sum(1, keepdim=True)
        tc_g = targets_oh[:,[1,3]].sum(1, keepdim=True)

        et_p = probs[:,3:4]
        et_g = targets_oh[:,3:4]

        dice_wt = self.dice(wt_p, wt_g)
        dice_tc = self.dice(tc_p, tc_g)
        dice_et = self.dice(et_p, et_g)

        loss = 1 - (dice_wt + 1.5*dice_tc + 1.5*dice_et) / 4
        return loss.mean()

class TCFocusedLoss(nn.Module):
    def __init__(self, max_epochs):
        super().__init__()
        self.focal = FocalLoss(gamma=1.5)
        self.region = RegionDiceLoss()
        self.boundary = FastBoundaryLoss()
        self.max_epochs = max_epochs
        self.epoch = 0
        self.eps = 1e-6

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_boundary_weight(self):
        progress = self.epoch / self.max_epochs
        if self.epoch <= 10:
            loss = 0
        else:
            loss = 0.4 * (1 - math.cos(math.pi * progress)) / 2
        return loss

    def dice_class1(self, probs, target):
        p = probs[:,1]
        g = (target==1).float()
        inter = (p*g).sum()
        union = p.sum() + g.sum()
        return 1 - (2*inter + self.eps)/(union + self.eps)

    def dice_et(self, probs, target):
        p = probs[:,3]
        g = (target==3).float()
        inter = (p*g).sum()
        union = p.sum() + g.sum()
        return 1 - (2*inter + self.eps)/(union + self.eps)

    def forward(self, logits, target):
        probs = torch.softmax(logits,1)

        loss_focal = self.focal(logits,target)
        loss_region = self.region(logits,target)
        loss_boundary = self.boundary(logits,target)
        loss_tc = self.dice_class1(probs, target)
        loss_et = self.dice_et(probs, target)

        w_b = self.get_boundary_weight()

        return (
            0.25*loss_focal +
            0.45*loss_region +
            0.25*loss_et+
            0.15*loss_tc+
            w_b * loss_boundary
        )