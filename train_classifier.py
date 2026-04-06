# train_classifier.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CelebADataset
from classifier import CelebAClassifier
from torch.amp import GradScaler, autocast


def train(
    celeba_root="celeba",
    attr="Smiling",
    epochs=5,
    batch_size = 512 if torch.cuda.is_available() else 32,
    lr=1e-4,
    save_path="checkpoints/classifier_smiling.pt",
    img_dir=None,
    size = 128 if torch.cuda.is_available() else 256
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    use_cuda = device.type == "cuda"
    if use_cuda:
        # Enable kernel autotuning for fixed-size image workloads.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    # Datasets
    train_ds = CelebADataset(celeba_root, attr=attr, split="train", size=size, img_dir=img_dir)
    val_ds   = CelebADataset(celeba_root, attr=attr, split="val",   size=size, img_dir=img_dir)

    if use_cuda:
        num_workers = min(8, os.cpu_count() or 4)
    else:
        num_workers = 2
    pin_memory = use_cuda

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        **loader_kwargs,
    )

    # Model
    model = CelebAClassifier(num_classes=2).to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)


    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0

    scaler = GradScaler(enabled=use_cuda)
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_cuda:
                imgs = imgs.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        # --- Validate ---
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_cuda:
                    imgs = imgs.to(memory_format=torch.channels_last)
                with autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                    logits = model(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += imgs.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}  "
              f"loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"  Saved best model  val_acc={val_acc:.4f}")

    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()