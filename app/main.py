import os
import sys
import time
import json
from datetime import datetime
import tempfile
import cv2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import mlflow
import mlflow.pytorch
from fastapi.responses import JSONResponse
import glob
from fastapi.middleware.cors import CORSMiddleware # CORS 추가
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client

# --- Supabase DB 연결 설정 ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# 만약 허깅페이스 환경변수가 잘 들어왔다면 클라이언트 생성
if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    print("⚠️ Supabase 연결 정보가 없습니다.")

# ==========================================
# 0. MLflow pickle 로드를 위한 모듈 alias
# ==========================================
sys.modules['train'] = sys.modules[__name__]

KST = timezone(timedelta(hours=9))
# ==========================================
# 1. 모델 아키텍처 정의
# ==========================================
class FallDetectionModel(nn.Module):
    def __init__(self, lstm_hidden_size=256, num_classes=1):
        super(FallDetectionModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        b, seq_len, c, h, w = x.size()
        x = x.view(b * seq_len, c, h, w)
        features = self.resnet(x)
        features = features.view(b, seq_len, -1)
        lstm_out, (h_n, c_n) = self.lstm(features)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out


app = FastAPI(title="Fall Detection MLOps API")

# 프론트엔드 연동을 위한 CORS 필수 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# ==========================================
# 2. MLflow & DagsHub 연동
# ==========================================
@app.on_event("startup")
def load_model():
    global model

    DAGSHUB_URI = "https://dagshub.com/Luvid-Lexus/Fall_Detection.mlflow"
    os.environ["MLFLOW_TRACKING_URI"] = DAGSHUB_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Luvid-Lexus"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "b95d928ceee4d1d363d277bf124997ac18d4338a"
    mlflow.set_tracking_uri(DAGSHUB_URI)

    try:
        model_name = "FallDetection_Prod_Model"
        model_uri = f"models:/{model_name}/Production"
        print(f"모델 다운로드 중... URI: {model_uri}")

        model = mlflow.pytorch.load_model(
            model_uri,
            map_location=torch.device("cpu")
        )
        model.to(device)
        model.eval()
        print("✅ MLflow 모델 로드 완료!")

    except Exception as e:
        print(f"❌ MLflow 연동 실패: {e}")
        print("⚠️  빈 FallDetectionModel로 대체합니다.")
        model = FallDetectionModel().to(device)
        model.eval()


# ==========================================
# 3. 전처리 및 로그 유틸리티
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def save_inference_log(log_data: dict):
    # 한국 시간(KST) 설정 (UTC + 9시간)
    KST = timezone(timedelta(hours=9))
    
    log_dir = "/code/app/logs" if os.path.exists("/code") else "app/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # datetime.now() 안에 KST를 넣어줍니다!
    log_file = os.path.join(log_dir, f"log_{datetime.now(KST).strftime('%Y%m%d')}.json")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")


# ==========================================
# 4. 슬라이딩 윈도우 프레임 추출 함수
# ==========================================
def extract_frames_sliding(video_path: str, fps_to_sample: int = 3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30.0

    skip_interval = max(int(video_fps / fps_to_sample), 1)

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(Image.fromarray(frame_rgb)))
        frame_idx += 1

    cap.release()
    return frames


# ==========================================
# 5. 추론 API 엔드포인트
# ==========================================
@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    camera_id: str = "CAM_01",
    video_snippet_id: str = "SNIPPET_001"
):
    if model is None:
        return {"error": "모델이 준비되지 않았습니다."}

    start_time = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        all_frames = extract_frames_sliding(tmp_path, fps_to_sample=3)

        if len(all_frames) < 10:
            return {
                "error": f"영상이 너무 짧거나 유효 프레임이 부족합니다. (추출된 프레임: {len(all_frames)}장)"
            }

        max_confidence = 0.0
        step_size = 2

        with torch.no_grad():
            for start in range(0, len(all_frames) - 10 + 1, step_size):
                window_tensor = torch.stack(
                    all_frames[start: start + 10]
                ).unsqueeze(0).to(device)

                logits = model(window_tensor)
                confidence = torch.sigmoid(logits).item()

                if confidence > max_confidence:
                    max_confidence = confidence

        final_confidence = round(max_confidence, 4)
        predicted_class = 1 if final_confidence >= 0.5 else 0
        display_message = "전도 상황 감지" if predicted_class == 1 else "정상"
        
        actual_snippet_id = video.filename

        # 4. 민준님이 요청한 8가지 데이터 구성
        log_data = {
            "timestamp": datetime.now(KST).isoformat(),
            "camera_id": camera_id,
            "video_snippet_id": actual_snippet_id,
            "predicted_class": predicted_class,
            "confidence": final_confidence,
            "inference_time_ms": round((time.time() - start_time) * 1000, 2),
            "input_frames": 10,
            "model_version": "FallDetection_Prod_Model",
            "threshold_applied": 0.5,
            "admin_confirm_label": None,
            "error_type": None
        }
        background_tasks.add_task(save_inference_log, log_data)

        # --- Supabase DB에 로그 전송 ---
        if supabase:
            try:
                supabase.table('logs').insert({
                    "filename": actual_snippet_id,    # 찬호님 코드 변수
                    "prediction": display_message,    # 찬호님 코드 변수
                    "probability": final_confidence   # 찬호님 코드 변수
                }).execute()
                print("✅ Supabase DB에 로그 저장 완료!")
            except Exception as e:
                print(f"❌ DB 저장 실패: {e}")

        # 💡 핵심 수정 포인트: return 에 "data" 키를 추가하여 log_data를 통째로 넘겨줍니다!
        return {
            "status": "success",
            "message": display_message,
            "confidence": final_confidence,
            "is_fall": bool(predicted_class),
            "processed_video": video.filename,
            "data": log_data  # <--- 이 부분이 추가되었습니다.
        }

    except Exception as e:
        error_log = {
            "timestamp": datetime.now(KST).isoformat(),
            "camera_id": camera_id,
            "video_snippet_id": video_snippet_id,
            "predicted_class": None,
            "confidence": None,
            "inference_time_ms": round((time.time() - start_time) * 1000, 2),
            "input_frames": None,
            "model_version": "FallDetection_Prod_Model",
            "threshold_applied": 0.5,
            "admin_confirm_label": None,
            "error_type": str(e)
        }
        background_tasks.add_task(save_inference_log, error_log)
        return {"status": "error", "message": str(e)}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        import gc
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.get("/logs")
def get_logs():
    log_dir = "/code/app/logs" if os.path.exists("/code") else "app/logs"
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    
    all_logs = []
    for file in log_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_logs.append(json.loads(line))
                    
    return JSONResponse(content=all_logs)