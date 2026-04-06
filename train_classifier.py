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
    print(f"Device: {device}")

    # Datasets
    train_ds = CelebADataset(celeba_root, attr=attr, split="train", size=size, img_dir=img_dir)
    val_ds   = CelebADataset(celeba_root, attr=attr, split="val",   size=size, img_dir=img_dir)

    num_workers = 16 if torch.cuda.is_available() else 2
    pin_memory = torch.cuda.is_available()  # also fix the MPS warning

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True,  # keep worker processes alive between epochs
        prefetch_factor=4,        # each worker prefetches 4 batches ahead
)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Model
    model = CelebAClassifier(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        batch_size = batch_size * torch.cuda.device_count()



    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_acc = 0.0

    scaler = GradScaler(enabled=torch.cuda.is_available())
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
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
                imgs, labels = imgs.to(device), labels.to(device)
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
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model  val_acc={val_acc:.4f}")

    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()