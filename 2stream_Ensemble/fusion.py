import torch
from torchvision import transforms
from data_apex import CASME2Dataset as ApexDataset
from data_flow import CASME2Dataset as FlowDataset
from model import SimpleCNN
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 이미지 전처리 (학습에 사용했던 것과 동일하게)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 각각의 데이터셋 로드
apex_dataset = ApexDataset(csv_path='./data/expression_labels.csv', image_root='./data/ApexImages_noOthersFear', transform=transform)
flow_dataset = FlowDataset(csv_path='./data/expression_labels.csv', image_root='./data/optical_flow', transform=transform)

model1 = SimpleCNN(num_classes=5)
model2 = SimpleCNN(num_classes=5)


# 저장된 state_dict 불러오기
model1.load_state_dict(torch.load('SimpleCNN_model.pth'))
model2.load_state_dict(torch.load('optical_model.pth'))
model1.eval()
model2.eval()

# 앙상블 함수 정의
def ensemble_predict(model1, model2, x1, x2, weight1=0.5, weight2=0.5):
    with torch.no_grad():
        softmax1 = F.softmax(model1(x1), dim=1)
        softmax2 = F.softmax(model2(x2), dim=1)
        final_score = weight1 * softmax1 + weight2 * softmax2
        prediction = torch.argmax(final_score, dim=1)
    return prediction, final_score

# 전체 데이터셋에 대해 반복 수행
correct = 0
for idx in range(len(apex_dataset)):
    x1, label1 = apex_dataset[idx]
    x2, label2 = flow_dataset[idx]
    assert label1 == label2, f"Label mismatch at index {idx}"

    x1 = x1.unsqueeze(0)  # (1, C, H, W)
    x2 = x2.unsqueeze(0)

    pred, score = ensemble_predict(model1, model2, x1, x2)
    if pred.item() == label1:
        correct += 1

accuracy = correct / len(apex_dataset) * 100
print(f"앙상블 정확도: {accuracy:.2f}%")

# ✅ 예측 결과 수집
all_preds = []
for idx in range(len(apex_dataset)):
    x1, label1 = apex_dataset[idx]
    x2, label2 = flow_dataset[idx]
    assert label1 == label2, f"❗ Index {idx}: 라벨 불일치"

    x1 = x1.unsqueeze(0)
    x2 = x2.unsqueeze(0)

    pred, _ = ensemble_predict(model1, model2, x1, x2)
    all_preds.append(pred.item())

# ✅ 분포 출력
print("🎯 예측 클래스 분포:")
print(Counter(all_preds))

# Apex 또는 Flow dataset 중 하나에서 꺼내기
print("🔢 인덱스 → 감정 라벨 매핑:")
print(apex_dataset.idx2label)

# 실제 정답 수집
all_labels = [apex_dataset[i][1] for i in range(len(apex_dataset))]

# confusion matrix 계산 및 시각화
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(apex_dataset.label2idx.keys()))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Ensemble Confusion Matrix')
plt.tight_layout()
plt.show()