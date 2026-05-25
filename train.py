import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import mlflow
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 우리가 만든 새로운 숟가락(DataLoader) 가져오기
from dataset import get_dataloader

# ==========================================
# 1. 모델 아키텍처 정의 (ResNet50 + LSTM)
# ==========================================
class FallDetectionModel(nn.Module):
    def __init__(self, sequence_length=10):
        super(FallDetectionModel, self).__init__()
        self.sequence_length = sequence_length
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity() # 마지막 분류기 제거
        
        self.lstm = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        
        x = x.view(batch_size * seq_length, c, h, w)
        features = self.resnet(x)
        
        features = features.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(features)
        
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

# ==========================================
# 2. 메인 학습 파이프라인
# ==========================================
def train_model():
    print("🚀 MLOps 파이프라인: 모델 학습을 준비합니다...\n")

    # ⚙️ 설정값 세팅 (통합 정답지 경로)
    TRAIN_MANIFEST = "train_manifest.csv" 
    VAL_MANIFEST = "valid_manifest.csv"
    
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 0.0001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 사용 중인 장비: {device}")

    train_loader = get_dataloader(TRAIN_MANIFEST, mode='train', batch_size=BATCH_SIZE)
    val_loader = get_dataloader(VAL_MANIFEST, mode='val', batch_size=BATCH_SIZE)
    
    model = FallDetectionModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # DagsHub 및 MLflow 클라우드 연동
    dagshub.init(repo_owner='Luvid-Lexus', repo_name='Fall_Detection', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Luvid-Lexus/Fall_Detection.mlflow")
    
    # -----------------------------------------------------
    # ✨ 스마트 모델 저장소 (자동 버전업 로직)
    # -----------------------------------------------------
    existing_dirs = glob.glob("model_v*")
    next_version = len(existing_dirs) + 1
    
    save_dir = f"model_v{next_version}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_model_v{next_version}.pth")
    
    print(f"\n📁 이번 학습의 최고 모델은 '{save_dir}' 폴더에 자동으로 안전하게 저장됩니다!")
    # -----------------------------------------------------

    best_recall = 0.0

    print("\n🎬 본격적인 학습을 시작합니다!")
    
    # 장부 이름도 버전에 맞게 자동으로 올라가도록 세팅!
    with mlflow.start_run(run_name=f"Balanced_Data_V{next_version}"):
        mlflow.log_params({"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR})
        
        for epoch in range(EPOCHS):
            # 🥊 학습 모드
            model.train()
            train_loss = 0.0
            
            for frames, labels in train_loader:
                frames = frames.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(frames)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 📝 검증 모드
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for frames, labels in val_loader:
                    frames = frames.to(device)
                    labels = labels.to(device).float().unsqueeze(1)
                    
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # 통계 지표 계산
            acc = accuracy_score(all_labels, all_preds)
            rec = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            
            mlflow.log_metrics({
                "train_loss": train_loss/len(train_loader),
                "val_loss": val_loss/len(val_loader),
                "val_recall": rec,
                "val_f1": f1
            }, step=epoch)
            
            # 모델 저장 로직 완벽 복구
            if rec >= best_recall:
                best_recall = rec
                torch.save(model.state_dict(), best_model_path)
                print(f"   ⭐ 최고 성능 갱신! '{best_model_path}'에 안전하게 저장했습니다.")

        # ---------------------------------------------------------
        # 3. 학습 종료 및 클라우드 업로드
        # ---------------------------------------------------------
        print("\n🎉 모든 학습이 끝났습니다. 최고 모델을 MLflow로 포장합니다...")
        model.load_state_dict(torch.load(best_model_path))
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="best_model",
            registered_model_name="FallDetection_Prod_Model"
        )
        print("✅ DagsHub Registry에 완벽하게 등록되었습니다!")

if __name__ == "__main__":
    train_model()