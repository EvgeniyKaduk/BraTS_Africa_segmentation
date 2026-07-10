# -*- coding: utf-8 -*-

import torch

gaussian_cache = {}

def get_gaussian_weight(patch_size, device):
    key = (patch_size, device)
    if key in gaussian_cache:
        return gaussian_cache[key]

    dz, dy, dx = patch_size
    z = torch.linspace(-1, 1, dz, device=device)
    y = torch.linspace(-1, 1, dy, device=device)
    x = torch.linspace(-1, 1, dx, device=device)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    g = torch.exp(-(xx**2 + yy**2 + zz**2) / 0.5)
    g = g / g.max()
    g = g.half().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

    gaussian_cache[key] = g
    return g

@torch.no_grad()
def sliding_window_fast(model, image, patch_size, stride, sw_batch_size=4):
    """
    Ускоренная sliding-window inference.
    Без вложенных циклов.
    """

    B, C, D, H, W = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    device = image.device

    # координаты начала окон
    ds = list(range(0, D - pd + 1, sd))
    hs = list(range(0, H - ph + 1, sh))
    ws = list(range(0, W - pw + 1, sw))

    if ds[-1] != D - pd: ds.append(D - pd)
    if hs[-1] != H - ph: hs.append(H - ph)
    if ws[-1] != W - pw: ws.append(W - pw)

    coords = [(d, h, w) for d in ds for h in hs for w in ws]

    num_classes = model.out_channels

    output = torch.zeros((B, num_classes, D, H, W),
                         device=device, dtype=torch.float32)
    norm_map = torch.zeros((1, 1, D, H, W),
                           device=device, dtype=torch.float32)

    weight = get_gaussian_weight(patch_size, device).float()

    # === батчевый прогон ===
    for i in range(0, len(coords), sw_batch_size):

        batch_coords = coords[i:i+sw_batch_size]

        patches = torch.cat([
            image[..., d:d+pd, h:h+ph, w:w+pw]
            for (d, h, w) in batch_coords
        ], dim=0)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                preds = model(patches)
        else:
            preds = model(patches)

        preds = preds.float()

        for j, (d, h, w) in enumerate(batch_coords):
            output[..., d:d+pd, h:h+ph, w:w+pw] += preds[j:j+1] * weight
            norm_map[..., d:d+pd, h:h+ph, w:w+pw] += weight

    return output / norm_map.clamp_min(1e-6)

def logits_to_segmentation(logits):
    return torch.argmax(logits, dim=1)

def visualize_case_overlay_color(
    image,
    mask,
    prediction,
    slice_id=None,
    save_path=None,
    show=True,
):
    """
    image: [C, D, H, W]
    mask:  [D, H, W]
    prediction: [D, H, W]

    slice_id: если None — автоматически выбирается срез с максимальной
              площадью опухоли по ground truth (а если опухоли нет — по
              предсказанию; если и там пусто — середина объёма).
    save_path: если задан — картинка сохраняется в файл (PNG).
    show: показывать ли окно на экране.
    """

    image = image.cpu()
    mask = mask.cpu()
    prediction = prediction.cpu()

    _, D, H, W = image.shape

    # === Автовыбор среза с максимальной площадью опухоли ===
    if slice_id is None:
        # число опухолевых вокселей в каждом срезе (по осям H, W)
        gt_area = (mask > 0).sum(dim=(1, 2))     # [D]
        pred_area = (prediction > 0).sum(dim=(1, 2))  # [D]

        if gt_area.max() > 0:
            slice_id = int(gt_area.argmax().item())
            src = "GT"
        elif pred_area.max() > 0:
            slice_id = int(pred_area.argmax().item())
            src = "Pred"
        else:
            slice_id = D // 2
            src = "middle (no tumor found)"
        print(f"  Selected slice {slice_id}/{D} by {src}")

    gt = mask[slice_id].numpy()
    pred = prediction[slice_id].numpy()
    mri = image[2, slice_id].numpy()  # T2

    # нормализация
    mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-6)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Ground Truth
    ax[0].imshow(mri, cmap="gray")
    ax[0].imshow(gt == 1, alpha=0.35, cmap="Reds")
    ax[0].imshow(gt == 2, alpha=0.35, cmap="Greens")
    ax[0].imshow(gt == 3, alpha=0.35, cmap="Blues")
    ax[0].set_title(f"Ground Truth (slice {slice_id})")

    # Prediction
    ax[1].imshow(mri, cmap="gray")
    ax[1].imshow(pred == 1, alpha=0.35, cmap="Reds")
    ax[1].imshow(pred == 2, alpha=0.35, cmap="Greens")
    ax[1].imshow(pred == 3, alpha=0.35, cmap="Blues")
    ax[1].set_title(f"Prediction (slice {slice_id})")

    # Errors
    error = (gt != pred)
    ax[2].imshow(mri, cmap="gray")
    ax[2].imshow(error, cmap="cool", alpha=0.5)
    ax[2].set_title("Errors")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)