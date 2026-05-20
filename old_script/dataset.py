import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FallDetectionSequenceDataset(Dataset):
    def __init__(self, root_dir, split='train', sequence_length=10, transform=None):
        # 예: C:/Users/qoiop/Downloads/mlops/dataset/train
        self.root_dir = os.path.join(root_dir, split)
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = []
        self.labels = []

        # 라벨 맵핑 (정상: 0, 전도: 1)
        label_map = {'stand': 0, 'fall': 1}

        # 데이터 탐색 로직 (train/fall, train/stand 폴더 안의 seq_ 폴더들 탐색)
        for label_name, label_idx in label_map.items():
            label_dir = os.path.join(self.root_dir, label_name)
            if not os.path.exists(label_dir):
                continue

            for seq_folder in os.listdir(label_dir):
                seq_path = os.path.join(label_dir, seq_folder)
                if os.path.isdir(seq_path):
                    # 해당 시퀀스 폴더 안의 이미지 프레임 10장 수집 및 이름순 정렬
                    frames = sorted(glob.glob(os.path.join(seq_path, '*.[jp][pn]*[g]'))) # jpg, png 모두 허용
                    
                    if len(frames) >= self.sequence_length:
                        # 앞에서부터 10장만 확실하게 자름
                        self.sequences.append(frames[:self.sequence_length])
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_frames = self.sequences[idx]
        label = self.labels[idx]

        images = []
        for frame_path in seq_frames:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # 10장의 이미지를 하나의 텐서 블록으로 합침 -> (10, 3, 224, 224)
        images_tensor = torch.stack(images)
        return images_tensor, label

def get_dataloader(split, batch_size, sequence_length=10):
    # 1. 최상위 데이터셋 경로 (절대 경로로 고정)
    root_dir = "C:/Users/qoiop/Downloads/mlops/dataset"

    # 2. ResNet50 모델에 맞춘 이미지 전처리 (크기 조정 및 정규화)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. 데이터셋 클래스 생성
    dataset = FallDetectionSequenceDataset(
        root_dir=root_dir,
        split=split,
        sequence_length=sequence_length,
        transform=transform
    )

    # 안전장치: 데이터가 0개로 읽혔을 때 즉시 원인을 알려줌
    if len(dataset) == 0:
        raise ValueError(f"🚨 데이터 0개 에러! '{os.path.join(root_dir, split)}' 폴더에 이미지가 제대로 있는지 확인해주세요.")

    print(f"📊 {split.upper()} 데이터셋 로드 완료: 총 {len(dataset)}개 시퀀스")

    # 4. 고성능 PC 맞춤형 DataLoader 반환
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'), # Train은 섞고, Test/Val은 섞지 않음
        num_workers=8,              # CPU의 남는 코어를 적극 활용하여 데이터 로딩 병목 제거
        pin_memory=True,            # RAM에서 GPU VRAM으로 데이터를 쏘아주는 속도 극대화
        drop_last=True              # 마지막 자투리 배치를 버려서 학습 안정성 확보
    )

    return dataloader