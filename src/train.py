import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from dataset import ThyroidSegmentationDataset
from model import UNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        d_loss = dice_loss(preds, targets)
        return bce_loss + d_loss


def dice_coef(pred, target, smooth=1.0):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def train(model, loader, optimizer, criterion, device):
    model.train()
    loop = tqdm(loader, leave=False)
    running_loss = 0

    for imgs, masks in loop:
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Train Loss: {loss.item():.4f}")

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    running_dice = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)
            running_loss += loss.item()

            running_dice += dice_coef(preds, masks)

    avg_loss = running_loss / len(loader)
    avg_dice = running_dice / len(loader)
    return avg_loss, avg_dice


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images_dir = "data/images"
    masks_dir = "data/masks"

    image_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    full_dataset = ThyroidSegmentationDataset(images_dir, masks_dir, image_transforms, mask_transforms)

    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = UNet().to(device)
    
    print("Architecture\n")
    summary(model, (3, 256, 256))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = BCEDiceLoss()

    best_val_loss = float('inf')
    epochs = 20

    train_losses = []
    val_losses = []
    val_dice_scores = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), val_dice_scores, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Score per Epoch')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()