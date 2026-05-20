import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FallDetectionVideoDataset(Dataset):
    def __init__(self, manifest_path, mode='train', sequence_length=10, transform=None):
        # 1. 정답지(CSV) 읽어오기
        self.manifest = pd.read_csv(manifest_path)
        self.mode = mode # 'train', 'val', 'test' 중 하나
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_path = row['video_path'] # 예: clips/C_3_7_5_...mp4
        label = int(row['true_label']) # 전도=1, 정상=0

        # 2. 영상 파일에서 프레임(이미지) 추출하기
        frames = self._load_video_frames(video_path)

        # 3. 모델 전처리 (224x224 리사이즈 및 텐서 변환)
        transformed_frames = []
        for img_array in frames:
            # OpenCV의 numpy 배열을 PIL 이미지로 변환 (transforms 적용을 위해)
            img_pil = Image.fromarray(img_array)
            if self.transform:
                img_pil = self.transform(img_pil)
            transformed_frames.append(img_pil)

        # 4. 10장의 이미지를 하나의 텐서 블록으로 묶기 -> (10, 3, 224, 224)
        frames_tensor = torch.stack(transformed_frames)
        return frames_tensor, label

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV는 BGR로 읽으므로 RGB로 색상 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        total_frames = len(frames)

        # 🚨 예외 처리 1: 영상이 아예 비어있거나 읽기 실패한 경우 (검은 화면 반환)
        if total_frames == 0:
            print(f"⚠️ [경고] 영상을 읽을 수 없습니다: {video_path}")
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.sequence_length

        # 🚨 예외 처리 2: 영상이 10장(3.3초)보다 짧은 경우 (마지막 프레임 복사해서 채워넣기)
        if total_frames < self.sequence_length:
            padding = [frames[-1]] * (self.sequence_length - total_frames)
            frames.extend(padding)
            total_frames = self.sequence_length

        # 🎲 ----------------------------------------------------
        # ⭐️ 핵심 마법: "랜덤하게 10장 뽑기 (Temporal Jittering)" ⭐️
        # -------------------------------------------------------
        if self.mode == 'train':
            # 학습할 때는 주사위를 굴려서 시작점(start_idx)을 무작위로 정합니다!
            start_idx = np.random.randint(0, total_frames - self.sequence_length + 1)
        else:
            # 평가(Test/Val)할 때는 항상 공정하게 영상의 '정중앙(가운데)' 부분을 뽑아옵니다.
            start_idx = (total_frames - self.sequence_length) // 2

        # 무작위(또는 중앙) 시작점부터 딱 10장을 잘라옵니다.
        selected_frames = frames[start_idx : start_idx + self.sequence_length]
        
        return selected_frames


def get_dataloader(manifest_path, mode='train', batch_size=8, sequence_length=10):
    # ResNet50 모델에 맞춘 이미지 전처리 (크기 조정 및 정규화)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 클래스 생성
    dataset = FallDetectionVideoDataset(
        manifest_path=manifest_path,
        mode=mode,
        sequence_length=sequence_length,
        transform=transform
    )

    if len(dataset) == 0:
        raise ValueError(f"🚨 데이터 0개 에러! {manifest_path} 파일을 확인해주세요.")

    # DataLoader 묶기 (학습할 때는 섞어주고(shuffle), 평가할 때는 그대로!)
    is_shuffle = (mode == 'train')
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_shuffle, 
        num_workers=0 # 윈도우 환경 에러 방지
    )
    
    return dataloader