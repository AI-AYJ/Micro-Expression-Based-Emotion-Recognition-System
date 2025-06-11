import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt



def plot_training_history(history):
    """
    history: dict with keys 'train_loss','val_loss','train_acc','val_acc'
             each is a list of epoch values
    """
    epochs = list(range(1, len(history['train_loss']) + 1))

    # Plot Loss
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'],   label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def train(model, train_dataset, val_dataset, epochs=10, batch_size=32, learning_rate=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            y = y.squeeze()
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100
        val_loss, val_acc = evaluate(model, val_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")
        

    return model, history





def evaluate(model, val_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            y = y.squeeze()
            preds = model(x)
            loss += criterion(preds, y).item()
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return loss / len(val_loader), correct / total * 100

def test(model, test_dataset, batch_size=32):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    preds_all, labels_all, paths_all = [], [], []
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, paths = batch
            else:
                x, y = batch
                paths = [''] * len(y)  # 빈 문자열 리스트로 채움

            y = y.squeeze()
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()

            _, preds = torch.max(out, 1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            preds_all.extend(preds.tolist())
            labels_all.extend(y.tolist())
            paths_all.extend(paths)

    # 정확도 및 평균 손실 계산
    test_accuracy = correct / total * 100
    test_loss = total_loss / len(test_loader)

    # 결과 저장
    if hasattr(test_dataset.dataset, "idx2label"):
        idx2label = test_dataset.dataset.idx2label
        result_df = pd.DataFrame({
            "ImagePath": paths_all,
            "TrueLabel": [idx2label[i] for i in labels_all],
            "PredictedLabel": [idx2label[i] for i in preds_all]
        })
    else:
        result_df = pd.DataFrame({
            "ImagePath": paths_all,
            "TrueLabel": labels_all,
            "PredictedLabel": preds_all
        })

    result_df.to_csv("test_predictions.csv", index=False)
    

    # 결과 반환
    
    return test_loss, test_accuracy

    
