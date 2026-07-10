# -*- coding: utf-8 -*-

import torch
from src.metrics import dice_score_wt_tc_et
from src.inference import sliding_window_fast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    dice_scores = {'WT':0, 'TC':0, 'ET':0}

    for images, masks in loader:
        images = images.contiguous().to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # DICE scores
        scores = dice_score_wt_tc_et(outputs, masks)
        for k in dice_scores:
            dice_scores[k] += scores[k]

    mean_loss = total_loss / len(loader)
    mean_dice = {k: dice_scores[k]/len(loader) for k in dice_scores}

    return mean_loss, mean_dice

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, patch_size=(128,128,128),
                       stride=(96,96,96)):

    model.eval()
    total_loss = 0
    dice_scores = {'WT':0, 'TC':0, 'ET':0}

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        logits = sliding_window_fast(model, images, patch_size, stride, sw_batch_size=4)

        loss = criterion(logits, masks)

        total_loss += loss.item()

        scores = dice_score_wt_tc_et(logits, masks)

        for k in dice_scores:
            dice_scores[k] += scores[k]

    mean_loss = total_loss / len(loader)
    mean_dice = {k: dice_scores[k]/len(loader) for k in dice_scores}
    return mean_loss, mean_dice