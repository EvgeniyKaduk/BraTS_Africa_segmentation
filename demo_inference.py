# -*- coding: utf-8 -*-
import os
import torch
import argparse

from src.model import FastResidualUNet3D
from src.dataset import BraTSAfricaDataset, collect_subject_paths
from src.inference import (
    sliding_window_fast,
    logits_to_segmentation,
    visualize_case_overlay_color,
)

torch.set_num_threads(os.cpu_count())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="demo_data",
                        help="Корневая папка демо-данных (внутри: group/subject/файлы)")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth")
    parser.add_argument("--subject_idx", type=int, default=None,
                        help="Индекс пациента (0,1,2). Если не задан — прогоняются все")
    parser.add_argument("--save_dir", type=str, default="outputs",
                        help="Папка для сохранения картинок PNG")
    parser.add_argument("--no_show", action="store_true",
                        help="Не показывать окна (только сохранять в файлы)")
    return parser.parse_args()


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = FastResidualUNet3D(in_channels=4, num_classes=4)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Model loaded from {model_path} (device: {device})")
    return model


def run_inference(model, image, device):
    image = image.unsqueeze(0).to(device)  # [1, 4, D, H, W]
    logits = sliding_window_fast(
        model, image,
        patch_size=(128, 128, 128),
        stride=(128, 128, 128),
        sw_batch_size=1,
    )
    prediction = logits_to_segmentation(logits)
    return prediction[0].cpu()


def main():
    args = parse_args()

    model = load_model(args.model_path, DEVICE)

    # Собираем всех пациентов (все 3)
    subjects = collect_subject_paths(args.data_path)
    if len(subjects) == 0:
        raise RuntimeError(
            f"Не найдено пациентов в {args.data_path}. "
            f"Проверь структуру: data_path/group/subject/файлы"
        )

    dataset = BraTSAfricaDataset(
        data_path=args.data_path,
        subjects=subjects,
        mode="val",          # без патчей, без аугментаций, seg обязателен
        augment=False,
    )

    # Выбираем: один пациент или все
    if args.subject_idx is not None:
        indices = [args.subject_idx]
    else:
        indices = list(range(len(dataset)))

    for i in indices:
        subject_name = os.path.basename(subjects[i])
        print(f"\n=== Subject {i+1}/{len(dataset)}: {subject_name} ===")
        image, mask = dataset[i]

        print("Running inference...")
        prediction = run_inference(model, image, DEVICE)

        print("Visualizing...")
        save_path = os.path.join(args.save_dir, f"{subject_name}.png")
        visualize_case_overlay_color(
            image, mask, prediction,
            save_path=save_path,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()