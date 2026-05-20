import mlflow
import dagshub
import torch
from train import FallDetectionModel # 기존 모델 구조 임포트

# 1. 내 DagsHub와 연결
dagshub.init(repo_owner='Luvid-Lexus', repo_name='Fall_Detection', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Luvid-Lexus/Fall_Detection.mlflow")

# 2. 로컬에 있는 최고 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallDetectionModel().to(device)

# 모델 파일이 있는 전체 경로 설정
model_path = r"C:\Users\qoiop\Downloads\mlops\model_v1\best_model.pth"

# 지정된 경로에서 모델 가중치 로드
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ {model_path}에서 모델을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"❌ 해당 경로에 'best_model.pth' 파일이 없습니다. 경로를 다시 확인해주세요: {model_path}")
    exit()

# 3. MLflow 장부에 강제 수동 등록!
print("🚀 수동으로 DagsHub Registry에 모델을 업로드합니다...")
with mlflow.start_run(run_name="Manual_Model_Upload"):
    mlflow.pytorch.log_model(
        pytorch_model=model, 
        name="best_model", 
        registered_model_name="FallDetection_Prod_Model"
        # ⭐️ pt2 포맷과 가짜 데이터 관련 코드를 모두 삭제했습니다. (가장 확실한 기본값 사용)
    )
print("✨ DagsHub 클라우드 업로드 및 등록 완료! 배포 담당자에게 토큰을 넘겨주세요.")