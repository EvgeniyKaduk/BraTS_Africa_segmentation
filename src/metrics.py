# -*- coding: utf-8 -*-

import torch

@torch.no_grad()
def dice_score_wt_tc_et(logits, targets, eps=1e-6):
    """
    logits:  [B, C, D, H, W]
    targets: [B, D, H, W]
    return: dict {WT, TC, ET}
    """
    preds = torch.argmax(logits, dim=1)

    # --- регионы ---
    wt_pred = (preds > 0)
    wt_gt   = (targets > 0)

    tc_pred = (preds == 1) | (preds == 3)
    tc_gt   = (targets == 1) | (targets == 3)

    et_pred = (preds == 3)
    et_gt   = (targets == 3)

    def dice(p, g):
        inter = (p & g).sum().float()
        union = p.sum() + g.sum()
        if union == 0:
            return torch.tensor(1.0, device=logits.device)
        return (2 * inter + eps) / (union + eps)

    return {
        "WT": dice(wt_pred, wt_gt).item(),
        "TC": dice(tc_pred, tc_gt).item(),
        "ET": dice(et_pred, et_gt).item()
    }