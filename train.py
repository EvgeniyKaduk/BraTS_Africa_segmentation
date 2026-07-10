# -*- coding: utf-8 -*-
import os
import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataset import BraTSAfricaDataset, collect_subject_paths
from src.model import FastResidualUNet3D
from src.losses import TCFocusedLoss
from src.train_utils import train_one_epoch, validate_one_epoch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/BraTS-Africa",
        help="Path to dataset"
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=42)

    return parser.parse_args()

# MAIN
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Dataset
    all_subjects = collect_subject_paths(args.data_path)

    train_subjects, val_subjects = train_test_split(
        all_subjects,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True
    )

    train_dataset = BraTSAfricaDataset(
        args.data_path,
        subjects=train_subjects,
        mode="train",
        augment=True
    )

    val_dataset = BraTSAfricaDataset(
        args.data_path,
        subjects=val_subjects,
        mode="val",
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Model
    model = FastResidualUNet3D(
        in_channels=4,
        num_classes=4,
        base_c=32
    ).to(device)

    scaler = torch.amp.GradScaler("cuda")

    criterion = TCFocusedLoss(
        max_epochs=args.epochs
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
        fused=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=5e-6
    )

    stride = (96, 96, 96)

    best_val = 0.0
    epochs_no_improve = 0
    patience = 10

    # Training loop
    for epoch in range(args.epochs):

        criterion.set_epoch(epoch)

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        torch.cuda.empty_cache()

        val_loss, val_dice = validate_one_epoch(
            model,
            val_loader,
            criterion,
            stride=stride
        )

        mean_val_dice = (
            val_dice['WT'] +
            val_dice['TC'] +
            1.5 * val_dice['ET']
        ) / 3.5

        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Dice: {val_dice}")
        print(f"Mean Dice: {mean_val_dice:.4f}")

        os.makedirs("models", exist_ok=True)
        # Save best model
        if mean_val_dice > best_val:
            best_val = mean_val_dice
            epochs_no_improve = 0

            torch.save(model.state_dict(), "models/best_model.pth")
            print("✅ Saved new best model")

        else:
            epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

        print("-" * 50)

# ENTRY POINT
if __name__ == "__main__":
    main()

