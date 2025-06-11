import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train(model, train_dataloader, val_dataloader, epochs=10, batch_size=32, learning_rate=0.001):
   
    train_dataloader = DataLoader(train_dataloader,batch_size=batch_size,shuffle=False)
    val_dataloader = DataLoader(val_dataloader,batch_size=batch_size,shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
       
        for x_batch, y_batch in train_dataloader:
            y_batch = y_batch.squeeze()
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
           
            _, predicted_labels = torch.max(predictions,1)
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)
           
        train_accuracy = correct / total * 100
        train_loss = epoch_loss / len(train_dataloader)
           
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
           
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                y_batch = y_batch.squeeze()
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
               
                _, predicted_labels = torch.max(predictions, 1)
                val_correct += (predicted_labels == y_batch).sum().item()
                val_total += y_batch.size(0)
        #print(val_correct, val_total)            
        val_accuracy = val_correct / val_total * 100
        val_loss = val_loss / len(val_dataloader)
               
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
               
    return model
    
def test(model,test_dataset,batch_size=32):
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    criterion=torch.nn.CrossEntropyLoss()
    
    model.eval()
    
    test_loss=0
    correct=0
    total=0
    
    with torch.no_grad():
        for x_batch,y_batch in test_dataloader:
            y_batch=y_batch.squeeze()
            predictions=model(x_batch)
            loss = criterion(predictions,y_batch)
            test_loss += loss.item()
            
            _, predicted_labels = torch.max(predictions, 1)
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)
        
    test_accuracy= correct/ total*100
    test_loss= test_loss/len(test_dataloader)
    

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:2f}%")
