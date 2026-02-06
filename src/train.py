import os
import time
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


@dataclass
class Config:
    base_dir: str = "C:/Users/patri/OneDrive/Desktop/JobHunt2025/cv_natural_scenes/data"
    train_dir: str = "seg_train/seg_train"
    test_dir: str = "seg_test/seg_test"

    out_dir: str = "C:/Users/patri/OneDrive/Desktop/JobHunt2025/cv_natural_scenes/outputs"
    model_dir: str = "C:/Users/patri/OneDrive/Desktop/JobHunt2025/cv_natural_scenes/models"

    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    val_split: float = 0.15
    seed: int = 42

    warmup_epochs: int = 2
    finetune_epochs: int = 3
    warmup_lr: float = 1e-3
    finetune_lr: float = 3e-4


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(cm, labels, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    model.train() if train_mode else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return (running_loss / total), (correct / total)


def freeze_backbone(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_last_block_and_fc(model: nn.Module):
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True


def main():
    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    train_path = os.path.join(cfg.base_dir, cfg.train_dir)
    test_path = os.path.join(cfg.base_dir, cfg.test_dir)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train path not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test path not found: {test_path}")

    device = get_device()
    print("Device:", device)

    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_train = datasets.ImageFolder(train_path, transform=train_tfms)
    class_names = full_train.classes
    num_classes = len(class_names)

    val_size = int(len(full_train) * cfg.val_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    val_ds.dataset.transform = eval_tfms

    test_ds = datasets.ImageFolder(test_path, transform=eval_tfms)

    print("Classes:", class_names)
    print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_path = os.path.join(cfg.model_dir, "resnet18_intel_scenes.pt")

    history = {
        "warmup": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
        "finetune": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
    }

    start = time.time()

    print("\nPHASE 1: Warm-up (train classifier head only)")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.warmup_lr)

    for epoch in range(1, cfg.warmup_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)

        history["warmup"]["train_loss"].append(tr_loss)
        history["warmup"]["train_acc"].append(tr_acc)
        history["warmup"]["val_loss"].append(va_loss)
        history["warmup"]["val_acc"].append(va_acc)

        print(f"Warmup Epoch {epoch}/{cfg.warmup_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model_state": model.state_dict(), "class_names": class_names, "config": cfg.__dict__}, best_path)

    print("\nPHASE 2: Fine-tune (unfreeze last block + classifier)")
    unfreeze_last_block_and_fc(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.finetune_lr)

    for epoch in range(1, cfg.finetune_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)

        history["finetune"]["train_loss"].append(tr_loss)
        history["finetune"]["train_acc"].append(tr_acc)
        history["finetune"]["val_loss"].append(va_loss)
        history["finetune"]["val_acc"].append(va_acc)

        print(f"Finetune Epoch {epoch}/{cfg.finetune_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model_state": model.state_dict(), "class_names": class_names, "config": cfg.__dict__}, best_path)

    print("\nTraining done. Best val acc:", round(best_val_acc, 4))
    print("Saved best model:", best_path)
    print("Elapsed seconds:", int(time.time() - start))

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(labels.tolist())

    report = classification_report(all_true, all_preds, target_names=class_names, digits=3)
    print("\nTEST CLASSIFICATION REPORT\n")
    print(report)

    cm = confusion_matrix(all_true, all_preds)
    cm_path = os.path.join(cfg.out_dir, "confusion_matrix_test.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    print("Saved confusion matrix:", cm_path)

    report_path = os.path.join(cfg.out_dir, "test_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved report:", report_path)

    hist_path = os.path.join(cfg.out_dir, "train_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("Saved history:", hist_path)


if __name__ == "__main__":
    main()
