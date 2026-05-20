import os
import json
import torch
import mlflow
import dagshub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms

# 우리가 만든 모듈 불러오기
from train import FallDetectionModel
from dataset import FallDetectionVideoDataset

# ⚙️ 설정값
MANIFEST_PATH = "monitoring_eval_manifest.csv"
MODEL_PATH = r"model_v1\best_model.pth"  # 로컬에 저장된 최고 모델 경로
OUTPUT_CSV = "outputs/eval_predictions.csv"
THRESHOLD = 0.5

def main():
    print("🚀 자동 채점기(Evaluator) 작동을 시작합니다...\n")

    # 0. 준비 작업 (outputs 폴더 생성)
    os.makedirs("outputs", exist_ok=True)

    # 1. MLflow 연결 세팅 (DagsHub)
    dagshub.init(repo_owner='Luvid-Lexus', repo_name='Fall_Detection', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Luvid-Lexus/Fall_Detection.mlflow")
    
    # 2. 장비(GPU/CPU) 및 모델 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallDetectionModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # ⭐️ 평가 모드로 전환 (학습 안 함)
    
    # 3. 데이터셋(숟가락) 준비
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # mode='test'로 설정하면 중심 구간 10장만 공정하게 뽑아옵니다.
    dataset = FallDetectionVideoDataset(manifest_path=MANIFEST_PATH, mode='test', transform=transform)
    df = pd.read_csv(MANIFEST_PATH) # 결과 기록을 위해 엑셀 원본도 읽어둠
    
    results = []
    
    print(f"📊 총 {len(dataset)}개의 클립을 채점합니다. 잠시만 기다려주세요...")

    # 4. 🚀 본격적인 자동 채점 (Inference)
    with torch.no_grad(): # 평가할 때는 기울기 계산을 꺼서 메모리를 절약합니다.
        for idx in range(len(dataset)):
            # 숟가락으로 영상 떠오기
            frames_tensor, true_label = dataset[idx]
            frames_tensor = frames_tensor.unsqueeze(0).to(device) # (1, 10, 3, 224, 224) 꼴로 변환
            
            # 모델에 넣고 결과 뽑기
            outputs = model(frames_tensor)
            probability = torch.sigmoid(outputs).item() # 0 ~ 1 사이의 확률값으로 변환
            predicted_label = 1 if probability >= THRESHOLD else 0
            
            # 기획안 로직: TP, TN, FP, FN 판별
            if predicted_label == 1 and true_label == 1:
                eval_result = "TP"
                error_type = ""
            elif predicted_label == 0 and true_label == 0:
                eval_result = "TN"
                error_type = ""
            elif predicted_label == 1 and true_label == 0:
                eval_result = "FP"
                error_type = f"FP_{df.iloc[idx]['behavior_type']}" # 예: FP_물건집기
            elif predicted_label == 0 and true_label == 1:
                eval_result = "FN"
                error_type = f"FN_{df.iloc[idx]['behavior_type']}" # 예: FN_계단전도
            
            # 결과 기록장에 한 줄 쓰기
            row_data = df.iloc[idx].to_dict()
            row_data.update({
                "predicted_label": predicted_label,
                "probability": round(probability, 4),
                "eval_result": eval_result,
                "error_type": error_type
            })
            results.append(row_data)

    # 5. 채점 완료! 성적표(CSV) 만들기
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    # 6. 전체 통계 지표(Metrics) 계산
    y_true = results_df['true_label']
    y_pred = results_df['predicted_label']
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix (혼동 행렬) 시각화 및 저장
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal(0)", "Fall(1)"], yticklabels=["Normal(0)", "Fall(1)"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    cm_path = "outputs/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # 오답 노트(FN, FP) 따로 빼기
    fn_df = results_df[results_df['eval_result'] == 'FN']
    fp_df = results_df[results_df['eval_result'] == 'FP']
    fn_df.to_csv("outputs/fn_cases.csv", index=False, encoding='utf-8-sig')
    fp_df.to_csv("outputs/fp_cases.csv", index=False, encoding='utf-8-sig')

    # 평가 지표 JSON 저장
    metrics_dict = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # 7. ☁️ MLflow에 기록 남기기
    print("\n☁️ MLflow 클라우드에 성적표를 전송합니다...")
    with mlflow.start_run(run_name="Monitoring_Eval_V1"):
        # 파라미터 기록
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("dataset_size", len(dataset))
        
        # 성적 기록
        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_precision", prec)
        mlflow.log_metric("eval_recall", rec)
        mlflow.log_metric("eval_f1", f1)
        
        # 파일(Artifact) 통째로 업로드!
        mlflow.log_artifact(OUTPUT_CSV)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact("outputs/fn_cases.csv")
        mlflow.log_artifact("outputs/fp_cases.csv")
        mlflow.log_artifact("outputs/metrics.json")

    # 8. 기획안 재학습 판단 기준 피드백 출력
    print("=" * 50)
    print("✨ [평가 결과 요약] ✨")
    print(f"정확도 (Accuracy) : {acc:.4f}")
    print(f"정밀도 (Precision): {prec:.4f}")
    print(f"재현율 (Recall)   : {rec:.4f} (⭐️ 전도 감지 모델의 핵심 지표)")
    print(f"F1 점수 (F1 Score): {f1:.4f}")
    print("=" * 50)
    
    print("\n💡 [모니터링 AI의 다음 스텝 제안]")
    if rec < 0.90:
        print("🚨 Recall(재현율)이 0.90 미만입니다. 전도를 놓치고 있으니 재학습이 필요합니다!")
    elif f1 < 0.85:
        print("🚨 F1 점수가 0.85 미만입니다. 정상/전도 구분이 불안정하니 재학습을 권장합니다.")
    else:
        print("✅ 모델 성능이 훌륭합니다! 현재 버전을 유지하셔도 좋습니다.")
        
    if len(fn_df) > 0:
        print(f"🔍 모델이 놓친 전도(FN): {len(fn_df)}건 (outputs/fn_cases.csv 확인)")
    if len(fp_df) > 0:
        print(f"🔍 정상인데 전도로 착각함(FP): {len(fp_df)}건 (outputs/fp_cases.csv 확인)")

if __name__ == "__main__":
    main()