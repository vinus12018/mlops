import os
import json
import torch
import mlflow
import dagshub
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms

# 프로젝트 내부 모듈 불러오기
from train import FallDetectionModel
from dataset import FallDetectionVideoDataset

def main(args):
    print("자동 채점 및 평가 프로세스를 시작합니다...\n")

    # 결과물 저장 폴더 생성
    os.makedirs("outputs", exist_ok=True)

    # MLflow 연결
    dagshub.init(repo_owner='Luvid-Lexus', repo_name='Fall_Detection', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Luvid-Lexus/Fall_Detection.mlflow")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallDetectionModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 평가용 전처리 파이프라인
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset 로드 (Valid 또는 Test 셋)
    dataset = FallDetectionVideoDataset(manifest_path=args.manifest, mode='test', transform=transform)
    df = pd.read_csv(args.manifest) 
    
    results = []
    
    print(f"총 {len(dataset)}개의 클립에 대해 추론을 진행합니다. 잠시만 기다려주세요...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            frames_tensor, true_label = dataset[idx]
            frames_tensor = frames_tensor.unsqueeze(0).to(device) 
            
            outputs = model(frames_tensor)
            probability = torch.sigmoid(outputs).item() 
            predicted_label = 1 if probability >= args.threshold else 0
            
            # 기획안 기준에 따른 평가 결과(TP/TN/FP/FN) 및 에러 타입 산출
            if predicted_label == 1 and true_label == 1:
                eval_result = "TP"
                error_type = ""
            elif predicted_label == 0 and true_label == 0:
                eval_result = "TN"
                error_type = ""
            elif predicted_label == 1 and true_label == 0:
                eval_result = "FP"
                error_type = f"FP_{df.iloc[idx]['behavior_type']}"
            elif predicted_label == 0 and true_label == 1:
                eval_result = "FN"
                error_type = f"FN_{df.iloc[idx]['behavior_type']}" 
            
            # 행 데이터 갱신
            row_data = df.iloc[idx].to_dict()
            row_data.update({
                "predicted_label": predicted_label,
                "probability": round(probability, 4),
                "eval_result": eval_result,
                "error_type": error_type
            })
            results.append(row_data)

    # 1. 전체 예측 결과 CSV 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    
    # 평가지표 산출
    y_true = results_df['true_label']
    y_pred = results_df['predicted_label']
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # TP/TN/FP/FN 카운트 산출
    tp_count = int((results_df['eval_result'] == 'TP').sum())
    tn_count = int((results_df['eval_result'] == 'TN').sum())
    fp_count = int((results_df['eval_result'] == 'FP').sum())
    fn_count = int((results_df['eval_result'] == 'FN').sum())
    
    # 2. Confusion Matrix 저장
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal(0)", "Fall(1)"], yticklabels=["Normal(0)", "Fall(1)"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    cm_path = "outputs/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # 3. FP, FN 오답 노트 분리 저장
    fn_df = results_df[results_df['eval_result'] == 'FN']
    fp_df = results_df[results_df['eval_result'] == 'FP']
    fn_df.to_csv("outputs/fn_cases.csv", index=False, encoding='utf-8-sig')
    fp_df.to_csv("outputs/fp_cases.csv", index=False, encoding='utf-8-sig')

    # 4. Metrics JSON 저장
    metrics_dict = {
        "accuracy": acc, 
        "precision": prec, 
        "recall": rec, 
        "f1_score": f1,
        "tp_count": tp_count,
        "tn_count": tn_count,
        "fp_count": fp_count,
        "fn_count": fn_count
    }
    with open("outputs/metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)

    # 5. MLflow 로깅
    print("\nMLflow 클라우드에 성적표를 전송합니다...")
    with mlflow.start_run(run_name="Monitoring_Eval_Production"):
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("dataset_size", len(dataset))
        mlflow.log_param("model_path", args.model_path)
        
        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_precision", prec)
        mlflow.log_metric("eval_recall", rec)
        mlflow.log_metric("eval_f1", f1)
        mlflow.log_metric("tp_count", tp_count)
        mlflow.log_metric("tn_count", tn_count)
        mlflow.log_metric("fp_count", fp_count)
        mlflow.log_metric("fn_count", fn_count)
        
        mlflow.log_artifact(args.output_csv)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact("outputs/fn_cases.csv")
        mlflow.log_artifact("outputs/fp_cases.csv")
        mlflow.log_artifact("outputs/metrics.json")

    print("\n[평가 결과 요약]")
    print("-" * 50)
    print(f"정확도 (Accuracy) : {acc:.4f}")
    print(f"정밀도 (Precision): {prec:.4f}")
    print(f"재현율 (Recall)   : {rec:.4f} (낙상 감지 모델 핵심 지표)")
    print(f"F1 점수 (F1 Score): {f1:.4f}")
    print(f"오탐지 (FP)       : {fp_count}건 (정상을 전도로 오인)")
    print(f"미탐지 (FN)       : {fn_count}건 (실제 전도를 놓침)")
    print("-" * 50)
    print("평가 리포트 및 오답 노트가 outputs/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="낙상 감지 모델 자동 평가 스크립트")
    # 반드시 새로 분할한 시험지(valid_manifest.csv)를 입력받도록 설정합니다.
    parser.add_argument("--manifest", type=str, default="valid_manifest.csv", help="평가할 정답지 CSV 경로")
    # 현재 환경에 맞게 학습된 모델 경로를 지정해 줍니다 (예: model_v7/best_model.pt)
    parser.add_argument("--model_path", type=str, required=True, help="평가할 모델 가중치 경로 (필수 입력)")
    parser.add_argument("--output_csv", type=str, default="outputs/eval_predictions.csv", help="예측 결과 저장 경로")
    parser.add_argument("--threshold", type=float, default=0.5, help="전도 판별 임계값")
    
    args = parser.parse_args()
    main(args)