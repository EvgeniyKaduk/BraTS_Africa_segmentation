# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random

class BraTSAfricaDataset(Dataset):
    def __init__(
        self,
        data_path,
        subjects,
        mode="train",
        cache_in_ram=True,
        patch_size=(96, 96, 96),
        p_foreground=0.75,
        augment=False
    ):
        self.data_path = data_path
        self.subjects = subjects
        self.mode = mode
        self.patch_size = patch_size
        self.p_foreground = p_foreground
        self.augment = augment
        self.cache_in_ram = cache_in_ram if mode=="train" else False


        self.data = []       # (image, mask)
        self.fg_cache = []   # bbox per subject

        if self.cache_in_ram:
            print(f"[{mode}] Caching dataset in RAM...")
            print(f"[{mode}] Cached {len(self.data)} subjects ✅")

            for subj_path in self.subjects:
                image, mask, fg = self._load_subject(subj_path)
                self.data.append((image, mask))
                self.fg_cache.append(fg)
        else:
            print(f"[{mode}] Using on-the-fly loading")

    # ---------------------------------------------------------
    # Загрузка одного пациента
    # ---------------------------------------------------------
    def _load_subject(self, subj_path):

        files = os.listdir(subj_path)

        def find_file(keyword):
            for f in files:
                if keyword in f and f.endswith((".nii", ".nii.gz")):
                    return os.path.join(subj_path, f)
            return None

        modality_names = ["t1n", "t1c", "t2w", "t2f"]
        modality_paths = {m: find_file(m) for m in modality_names}
        seg_path = find_file("seg")
        if seg_path is None:
            raise RuntimeError(f"Segmentation not found in {subj_path}")

        loaded = {}
        for m, p in modality_paths.items():
            if p is not None:
                img = nib.load(p)
                if not np.allclose(img.affine, np.diag(np.diag(img.affine))):
                    img = nib.as_closest_canonical(img)
                loaded[m] = np.asarray(img.dataobj, dtype=np.float32)

        if len(loaded) == 0:
            raise RuntimeError(f"No modalities found in {subj_path}")

        ref = next(iter(loaded.values()))
        ref_shape = ref.shape

        channels = []
        for m in modality_names:
            if m in loaded:
                vol = loaded[m]
                if vol.shape != ref_shape:
                    vol = self._match_shape(vol, ref_shape)
                channels.append(vol)
            else:
                channels.append(np.zeros(ref_shape, dtype=np.float32))

        image = np.stack(channels, axis=0)  # [4, H, W, D]

        # ---------- MASK ----------
        seg_img = nib.load(seg_path)
        seg_img = nib.as_closest_canonical(seg_img)
        mask = np.asarray(seg_img.dataobj, dtype=np.int16)

        if mask.shape != ref_shape:
            mask = self._match_shape(mask, ref_shape)

        # ---------- TO TORCH CPU ----------
        image = torch.from_numpy(image).permute(0, 3, 1, 2).float().contiguous()
        mask  = torch.from_numpy(mask).permute(2, 0, 1).long().contiguous()

        # ----------Нормализация--------------
        image = self.normalize_brain(image)

        # ---------- FG BBOX CACHE ----------
        def get_bbox(label):
            vox = torch.nonzero(mask == label, as_tuple=False)
            if len(vox) == 0:
                return None
            mins = vox.min(0).values.cpu().numpy()
            maxs = vox.max(0).values.cpu().numpy()

            return (mins.tolist(), maxs.tolist())

        fg_cache = {
            3: get_bbox(3),
            1: get_bbox(1),
            2: get_bbox(2),
        }

        return image, mask, fg_cache

    #----Нормализация__________________________________________
    def normalize_brain(self, image):
        # image: torch [C,D,H,W]

        for c in range(image.shape[0]):
            channel = image[c]

            mask = channel != 0
            if mask.sum() == 0:
                continue

            vox = channel[mask]

            mean = vox.mean()
            std = vox.std()

            std = torch.clamp(std, min=1e-6)

            channel[mask] = (vox - mean) / std
            channel[~mask] = 0

        return image

    # ---------------------------------------------------------
    def _match_shape(self, vol, target_shape):
        Ht, Wt, Dt = target_shape
        H, W, D = vol.shape

        vol = vol[
            max(0,(H-Ht)//2):max(0,(H-Ht)//2)+min(Ht,H),
            max(0,(W-Wt)//2):max(0,(W-Wt)//2)+min(Wt,W),
            max(0,(D-Dt)//2):max(0,(D-Dt)//2)+min(Dt,D)
        ]

        pad_h = Ht - vol.shape[0]
        pad_w = Wt - vol.shape[1]
        pad_d = Dt - vol.shape[2]

        vol = np.pad(vol, (
            (pad_h//2, pad_h - pad_h//2),
            (pad_w//2, pad_w - pad_w//2),
            (pad_d//2, pad_d - pad_d//2)
        ), mode="constant")

        return vol

    # ---------------------------------------------------------
    # Длина датасета
    # ---------------------------------------------------------
    def __len__(self):
        return len(self.subjects)

    # ---------------------------------------------------------
    def __getitem__(self, idx):

        if self.cache_in_ram:
            image, mask = self.data[idx]
            fg_cache = self.fg_cache[idx]
        else:
            image, mask, fg_cache = self._load_subject(self.subjects[idx])

        if self.mode == "train" and self.patch_size is not None:
            image, mask = self._get_patch(image, mask, fg_cache)

        if self.mode == "train" and self.augment:
            image, mask = self._augment_3d(image, mask)

        return image.contiguous(), mask.contiguous()

    # ---------------------------------------------------------
    def _get_patch(self, image, mask, fg_cache):
        _, D, H, W = image.shape
        pd, ph, pw = self.patch_size

        # ---------- выбираем центр ----------
        if random.random() < self.p_foreground:
            for label in (3, 1, 2):
                bbox = fg_cache[label]
                if bbox is not None:
                    mins, maxs = bbox
                    cd = random.randint(mins[0], maxs[0])
                    ch = random.randint(mins[1], maxs[1])
                    cw = random.randint(mins[2], maxs[2])
                    break
            else:
                cd = random.randint(0, D - 1)
                ch = random.randint(0, H - 1)
                cw = random.randint(0, W - 1)
        else:
            cd = random.randint(0, D - 1)
            ch = random.randint(0, H - 1)
            cw = random.randint(0, W - 1)

        # ---------- вычисляем окно ----------
        d0 = cd - pd // 2
        h0 = ch - ph // 2
        w0 = cw - pw // 2

        # ---------- сдвигаем окно внутрь объёма ----------
        d0 = max(0, min(d0, D - pd))
        h0 = max(0, min(h0, H - ph))
        w0 = max(0, min(w0, W - pw))


        d1 = d0 + pd
        h1 = h0 + ph
        w1 = w0 + pw

        image_patch = image[:, d0:d1, h0:h1, w0:w1]
        mask_patch  = mask[d0:d1, h0:h1, w0:w1]

        return image_patch, mask_patch

    # ---------------------------------------------------------
    def _augment_3d(self, image, mask):
        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[2])
            mask  = torch.flip(mask, dims=[1])

        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[3])
            mask  = torch.flip(mask, dims=[2])

        if torch.rand(1) < 0.5:
            k = random.randint(0, 3)
            image = torch.rot90(image, k, dims=(2,3))
            mask  = torch.rot90(mask, k, dims=(1,2))

        return image, mask

def collect_subject_paths(data_path):
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset path not found: {data_path}")

    subjects = []

    for group in os.listdir(data_path):
        group_path = os.path.join(data_path, group)
        if not os.path.isdir(group_path):
            continue

        for subj in os.listdir(group_path):
            subj_path = os.path.join(group_path, subj)
            if os.path.isdir(subj_path):
                subjects.append(subj_path)

    print(f"Total subjects found: {len(subjects)}")
    return subjects