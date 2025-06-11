import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms

from data import CASME2Dataset
from model import SimpleCNN
from utils import save_model
from train import train, test, plot_training_history

def main():
    # Settings
    csv_path         = './data/expression_labels.csv'
    image_root       = './data/optical_flow'
    model_save_path  = 'SimpleCNN_model.pth'
    epochs           = 10
    batch_size       = 32
    learning_rate    = 0.001
    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Dataset
    dataset = CASME2Dataset(csv_path=csv_path,
                             image_root=image_root,
                             transform=transform)
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.label2idx)}")

    # Split train/val
    indices = list(range(len(dataset)))
    stratify_labels = [dataset[i][1] for i in indices]
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=stratify_labels)

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size)

    # Model
    model = SimpleCNN(num_classes=len(dataset.label2idx))

    # Train & plot
    model, history = train(
        model, train_loader, val_loader,
        epochs, learning_rate, device)

    # Save model
    save_model(model, model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    # Plot training history
    plot_training_history(history)

    # Test
    test(model, val_loader, device)

if __name__ == '__main__':
    main()