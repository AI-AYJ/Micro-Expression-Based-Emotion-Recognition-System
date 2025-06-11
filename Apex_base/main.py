import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms

from data import CASME2Dataset
from model import SimpleCNN
from train import train, test, plot_training_history
from utils import save_model

def main():
    # 설정값
    csv_path = './data/expression_labels.csv'
    image_root = './data/ApexImages_noOthersFear'  # 압축 푼 기준 경로
    model_save_path = 'SimpleCNN_model.pth'
    epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # 전체 데이터셋 로드
    dataset = CASME2Dataset(csv_path=csv_path, image_root=image_root, transform=transform)
    print(f"Total samples: {len(dataset)}")
    num_classes = len(dataset.label2idx)
    print(f"Number of classes: {num_classes}")

    # 학습/검증 분리
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=[dataset[i][1] for i in indices])

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    


    # 모델 정의
    model = SimpleCNN(num_classes=num_classes)

    # 학습
    model, history = train(model, train_dataset, val_dataset, epochs, batch_size, learning_rate)

    # 저장
    save_model(model, model_save_path)
    print(f"✅ Model saved to {model_save_path}")
    
    # Plot training history
    plot_training_history(history)
    

    # 테스트
    test(model, val_dataset, batch_size)

if __name__ == '__main__':
    main()

    
    
    