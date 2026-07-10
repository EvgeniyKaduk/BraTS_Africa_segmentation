# -*- coding: utf-8 -*-
"""Скачивает веса модели и демо-данные (3 пациента) с Hugging Face."""
import os
from huggingface_hub import hf_hub_download, snapshot_download

os.makedirs("models", exist_ok=True)
os.makedirs("demo_data", exist_ok=True)

# 1. Веса модели
print("Downloading model weights...")
hf_hub_download(
    repo_id="EvgeniyEV/ResidualUNet3d",
    filename="best_model_.pth",
    local_dir="models",
)
print("Model weights -> models/ ✅")

# 2. Демо-данные (скачиваются "как есть", с сохранением структуры папок)
print("Downloading demo data (3 subjects)...")
snapshot_download(
    repo_id="EvgeniyEV/brats-africa-demo",
    repo_type="dataset",
    local_dir="demo_data",
    allow_patterns=["*.nii.gz", "*.nii"],  # можно не тянуть README и прочее
)
print("Demo data -> demo_data/ ✅")

print("\nDone! Now run:")
print("  python demo_inference.py --data_path demo_data/BraTS_Africa_demo")