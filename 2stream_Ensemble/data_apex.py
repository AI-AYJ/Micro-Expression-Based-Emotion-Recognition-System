import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CASME2Dataset(Dataset):
    def __init__(self, csv_path='./data/expression_labels.csv', image_root='./data/ApexImages_noOthersFear', transform=None):

        """
        Args:
            csv_path (str): labels.csv 파일 경로
            image_root (str): 이미지 파일 루트 디렉토리 (csv 기준 경로)
            transform (callable, optional): 이미지 전처리 transform
        """
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

        # 레이블 문자열을 정수 인덱스로 변환
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.data['label_idx'] = self.data['label'].map(self.label2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 이미지 경로 만들기
        rel_path = self.data.iloc[idx]['path']
        img_path = os.path.join(self.image_root, rel_path)
        
        # 이미지 로드
        image = read_image(img_path).float() / 255.0  # [C, H, W] 형태
        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx]['label_idx']
        return image, label


        
        
        