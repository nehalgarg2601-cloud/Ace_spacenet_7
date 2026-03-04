import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_spacenet import SpaceNetDataset
from split_train_test import create_train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.hrnet_w48_sn7 import HRNetW48


# -------------------------------------------------------
# CONFIG & ARGS
# -------------------------------------------------------

parser = argparse.ArgumentParser(description="Train HRNet for SpaceNet7")
parser.add_argument("--data-root", type=str, required=True, help="Root directory for training data")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--epochs", type=int, default=70, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
args = parser.parse_args()

# Directories relative to this script
THIS_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(THIS_DIR, "checkpoints")
SPLIT_DIR = os.path.join(THIS_DIR, "..", "splits")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CHECKPOINT_DIR, "train.log")),
        logging.StreamHandler()
    ]
)

logging.info(f"Args: {args}")


# -------------------------------------------------------
# DEVICE
# -------------------------------------------------------

def get_device():

    if torch.backends.mps.is_available():
        logging.info("🚀 Using Apple MPS")
        return torch.device("mps")

    elif torch.cuda.is_available():
        logging.info("🚀 Using CUDA")
        return torch.device("cuda")

    else:
        logging.info("💻 Using CPU")
        return torch.device("cpu")


device = get_device()


# -------------------------------------------------------
# DATASET + TRAIN/TEST SPLIT
# -------------------------------------------------------

# Create train/test split files if they do not exist yet
train_list_path = os.path.join(SPLIT_DIR, "train_list.txt")
test_list_path = os.path.join(SPLIT_DIR, "test_list.txt")

if not (os.path.exists(train_list_path) and os.path.exists(test_list_path)):
    logging.info("Creating train/test split...")
    create_train_test_split(args.data_root, SPLIT_DIR, train_ratio=0.8, seed=42)

train_dataset = SpaceNetDataset(args.data_root, split="train", split_list_dir=SPLIT_DIR)
test_dataset = SpaceNetDataset(args.data_root, split="test", split_list_dir=SPLIT_DIR)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Test dataset size: {len(test_dataset)}")


# -------------------------------------------------------
# MODEL
# -------------------------------------------------------

model = HRNetW48(in_channels=4, num_classes=2)
model = model.to(device)


# -------------------------------------------------------
# OPTIMIZER
# -------------------------------------------------------

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[40, 60],
    gamma=0.1
)

criterion = nn.CrossEntropyLoss()


# -------------------------------------------------------
# CHECKPOINT FUNCTIONS
# -------------------------------------------------------

def save_checkpoint(epoch, model, optimizer, scheduler, best_test_loss, is_best=False):

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_test_loss": best_test_loss
    }

    # Save checkpoint for this epoch
    epoch_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
    torch.save(checkpoint, epoch_path)

    # Save latest checkpoint
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pth")
    torch.save(checkpoint, latest_path)

    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "best.pth")
        torch.save(checkpoint, best_path)
        logging.info(f" ⭐ Best checkpoint saved: {best_path}")

    logging.info(f" Checkpoint saved: {epoch_path}")


def load_checkpoint():

    path = os.path.join(CHECKPOINT_DIR, "latest.pth")

    if not os.path.exists(path):
        logging.info("No checkpoint found")
        return 0, float('inf')

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    logging.info(f" Resumed from epoch {checkpoint['epoch']}")
    best_loss = checkpoint.get("best_test_loss", float('inf'))

    return checkpoint["epoch"] + 1, best_loss


# -------------------------------------------------------
# RESUME TRAINING
# -------------------------------------------------------

start_epoch, best_test_loss = load_checkpoint()


# -------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------

for epoch in range(start_epoch, args.epochs):

    # ----------------------
    # Training loop
    # ----------------------
    model.train()

    total_loss = 0.0

    pbar = tqdm(train_loader)

    for images, masks in pbar:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        preds = model(images)

        preds = F.interpolate(
            preds,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )

        loss = criterion(preds, masks)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        pbar.set_description(f"Epoch {epoch} Train")
        pbar.set_postfix(loss=loss.item())

    scheduler.step()

    avg_train_loss = total_loss / len(train_loader)

    # ----------------------
    # Evaluation on test split
    # ----------------------
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            preds = F.interpolate(
                preds,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )

            loss = criterion(preds, masks)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)

    logging.info(
        f"Epoch {epoch} "
        f"avg train loss: {avg_train_loss:.4f} | "
        f"avg test loss: {avg_test_loss:.4f}"
    )

    is_best = avg_test_loss < best_test_loss
    if is_best:
        best_test_loss = avg_test_loss

    save_checkpoint(epoch, model, optimizer, scheduler, best_test_loss, is_best=is_best)


logging.info(" Training finished")
