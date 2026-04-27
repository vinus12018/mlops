import torch
import torch.nn as nn
import torchvision.models as models
import mlflow
import mlflow.pytorch
from sklearn.metrics import recall_score, f1_score
from dataset import get_dataloader 

# ==========================================
# 1. 모델 아키텍처 정의 (ResNet + LSTM)
# ==========================================
class FallDetectionModel(nn.Module):
    def __init__(self, lstm_hidden_size=256, num_classes=1):
        super(FallDetectionModel, self).__init__()
        # 시각 담당: ResNet50 (사전 학습된 가중치 사용)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 마지막 분류층을 제거하고 특징(Feature) 추출기로만 사용 (출력차원: 2048)
        self.resnet.fc = nn.Identity() 
        
        # 기억 담당: LSTM (시간적 맥락 분석)
        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=lstm_hidden_size, 
            num_layers=1, 
            batch_first=True # (Batch, Sequence, Feature) 형태 유지
        )
        
        # 최종 판단: 전도 여부 이진 분류 (Normal: 0, Fall: 1)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Sequence(10), Channels(3), H(224), W(224))
        b, seq_len, c, h, w = x.size()
        
        # ResNet에 넣기 위해 Batch와 Sequence를 일렬로 합침
        x = x.view(b * seq_len, c, h, w) 
        features = self.resnet(x) # shape: (B * 10, 2048)
        
        # LSTM에 넣기 위해 다시 Sequence 형태로 복구
        features = features.view(b, seq_len, -1) # shape: (B, 10, 2048)
        
        # LSTM 통과
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # 시퀀스의 가장 마지막 시점(10번째 프레임)의 출력값만 사용하여 판단
        last_time_step = lstm_out[:, -1, :] 
        out = self.fc(last_time_step) 
        return out # Sigmoid는 거치지 않음 (손실함수에서 처리)

# ==========================================
# 2. 메인 학습 및 MLflow 기록 루프
# ==========================================
def train_model():
    # --- 하이퍼파라미터 세팅 ---
    epochs = 10
    batch_size = 8
    learning_rate = 1e-4
    sequence_length = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallDetectionModel().to(device)
    
    # 이진 분류를 위한 손실 함수 (내부적으로 Sigmoid 포함)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader = get_dataloader('train', batch_size, sequence_length)
    val_loader = get_dataloader('test', batch_size, sequence_length)

    # --- [MLflow 1단계] 실험 세팅 (클라우드 공유용) ---
    import dagshub
    
    # 1. DagsHub 클라우드와 연결 (실행 시 터미널에서 인증 링크가 뜹니다)
    # 본인의 DagsHub 아이디와 방금 만든 프로젝트 이름을 적어주세요.
    dagshub.init(repo_owner='Luvid-Lexus', repo_name='Fall_Detection', mlflow=True)

    # 2. 로컬(내 컴퓨터)이 아닌, 모니터링 담당자가 볼 수 있는 클라우드 주소로 세팅!
    mlflow.set_tracking_uri("https://dagshub.com/Luvid-Lexus/Fall_Detection.mlflow")
    mlflow.set_experiment("Fall_Detection_ResNet_LSTM")

    print("클라우드 연동 완료! 학습 데이터를 DagsHub 서버로 전송합니다...")
    
    # MLflow 기록 시작 (이 블록 안에서 일어나는 모든 일이 대시보드에 기록됨)
    with mlflow.start_run() as run:
        # 파라미터 로깅
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "sequence_length": sequence_length,
            "model_type": "ResNet50+LSTM"
        })
        
        best_f1 = 0.0
        best_recall = 0.0

        for epoch in range(epochs):
            # ---------------- 학습 단계 ----------------
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
            
            # ---------------- 검증 단계 ----------------
            model.eval()
            all_preds = []
            all_labels = []
            val_loss_total = 0.0
            
            # 검증 시에는 기울기(Gradient) 계산을 끄고 메모리를 절약합니다.
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float()
                    
                    outputs = model(inputs)
                    
                    # 차원 맞추기 (squeeze()로 불필요한 차원 제거)
                    outputs = outputs.squeeze()
                    if outputs.ndim == 0: # 배치 사이즈가 1인 경우 에러 방지
                        outputs = outputs.unsqueeze(0)
                        
                    loss = criterion(outputs, labels)
                    val_loss_total += loss.item()
                    
                    # BCE 손실함수를 썼기 때문에, 추론 시에는 Sigmoid를 직접 씌워 확률(0~1)로 만듭니다.
                    probs = torch.sigmoid(outputs)
                    
                    # 0.5 이상이면 전도(1), 미만이면 정상(0)으로 예측
                    preds = torch.round(probs)
                    
                    # 배치별 결과를 전체 리스트에 차곡차곡 모음 (CPU 메모리로 이동)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            
            # ---------------- 지표 계산 (가짜 값 대체) ----------------
            # 데이터로더 전체 길이를 나누어 평균 Loss를 구합니다.
            current_val_loss = val_loss_total / len(val_loader)
            
            # 사이킷런(sklearn)을 이용해 진짜 Recall과 F1 스코어를 계산합니다.
            current_recall = recall_score(all_labels, all_preds, zero_division=0)
            current_f1 = f1_score(all_labels, all_preds, zero_division=0)

            print(f"Epoch {epoch+1}: Loss {current_val_loss:.4f} | Recall {current_recall:.4f} | F1 {current_f1:.4f}")
            
            # 매 에포크마다 지표를 MLflow에 로깅 (그래프 생성용)
            mlflow.log_metrics({
                "val_loss": current_val_loss,
                "val_recall": current_recall,
                "val_f1_score": current_f1
            }, step=epoch)

            # Best 모델 업데이트 (Recall과 F1 기준)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_recall = current_recall
                # MLflow에 Best 모델 파일 저장 (Tracking)
                mlflow.pytorch.log_model(model, "best_model")

        # ==========================================
        # 3. MLflow Model Registry (배포 기준 통과 시 자동 등록)
        # ==========================================
        print("\n학습 종료. 배포 적합성 검사를 시작합니다.")
        
        # 우리 팀의 엄격한 모니터링 기준: 미탐 방지를 위해 Recall 90% 이상 필수
        if best_recall >= 0.90 and best_f1 >= 0.85:
            registered_name = "FallDetection_Prod_Model"
            
            print(f"조건 통과! 로컬에 저장된 최고 모델을 MLflow 규격으로 포장하여 업로드합니다...")
            
            # 1. 로컬에 저장된 최고 모델 가중치 불러오기
            best_model = FallDetectionModel().to(device)
            best_model.load_state_dict(torch.load("best_model.pth"))
            
            # 2. MLflow에 PyTorch 모델 로깅 및 자동 등록 (포장 + 장부 기록 동시 처리)
            mlflow.pytorch.log_model(
                pytorch_model=best_model, 
                artifact_path="best_model", 
                registered_model_name=registered_name
            )
            
            print(f"성공! Recall({best_recall:.2f})이 기준을 통과하여 '{registered_name}'으로 Registry에 완벽하게 등록되었습니다.")
        else:
            print(f"보류: Recall({best_recall:.2f}) 또는 F1({best_f1:.2f})이 기준 미달입니다. 모델을 Registry에 등록하지 않습니다.")

if __name__ == "__main__":
    train_model()