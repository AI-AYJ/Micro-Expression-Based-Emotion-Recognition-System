import torch
import torch.optim as optim
import torch.nn as nn
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


def train(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()
            total += inputs.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc  = running_corrects / total

        # Validation phase
        model.eval()
        val_running_loss, val_running_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_running_corrects += (preds == labels).sum().item()
                val_total += inputs.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc  = val_running_corrects / val_total

        # Record metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        # Print metrics
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc*100:.2f}%")
        print(f"  Val   Loss: {epoch_val_loss:.4f}, Val   Accuracy: {epoch_val_acc*100:.2f}%")

    history = {
        'train_loss': train_losses,
        'val_loss':   val_losses,
        'train_acc':  train_accs,
        'val_acc':    val_accs
    }
    return model, history


def test(model, test_loader, device):
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total * 100
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
