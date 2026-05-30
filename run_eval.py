import os
import json
import torch
import mlflow
import dagshub
import argparse # ⭐️ 1. argparse 추가
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms

from train import FallDetectionModel
from dataset import FallDetectionVideoDataset

def main(args): # ⭐️ 2. args 받기
    print("🚀 자동 채점기(Evaluator) 작동을 시작합니다...\n")

    os.makedirs("outputs", exist_ok=True)

    dagshub.init(repo_owner='Luvid-Lexus', repo_name='Fall_Detection', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Luvid-Lexus/Fall_Detection.mlflow")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallDetectionModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device)) # ⭐️ args 사용
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FallDetectionVideoDataset(manifest_path=args.manifest, mode='test', transform=transform) # ⭐️ args 사용
    df = pd.read_csv(args.manifest) 
    
    results = []
    
    print(f"📊 총 {len(dataset)}개의 클립을 채점합니다. 잠시만 기다려주세요...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            frames_tensor, true_label = dataset[idx]
            frames_tensor = frames_tensor.unsqueeze(0).to(device) 
            
            outputs = model(frames_tensor)
            probability = torch.sigmoid(outputs).item() 
            predicted_label = 1 if probability >= args.threshold else 0 # ⭐️ args 사용
            
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
            
            row_data = df.iloc[idx].to_dict()
            row_data.update({
                "predicted_label": predicted_label,
                "probability": round(probability, 4),
                "eval_result": eval_result,
                "error_type": error_type
            })
            results.append(row_data)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False, encoding='utf-8-sig') # ⭐️ args 사용
    
    y_true = results_df['true_label']
    y_pred = results_df['predicted_label']
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ⭐️ 3. TP, TN, FP, FN 개수 계산 추가
    tp_count = int((results_df['eval_result'] == 'TP').sum())
    tn_count = int((results_df['eval_result'] == 'TN').sum())
    fp_count = int((results_df['eval_result'] == 'FP').sum())
    fn_count = int((results_df['eval_result'] == 'FN').sum())
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal(0)", "Fall(1)"], yticklabels=["Normal(0)", "Fall(1)"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    cm_path = "outputs/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    fn_df = results_df[results_df['eval_result'] == 'FN']
    fp_df = results_df[results_df['eval_result'] == 'FP']
    fn_df.to_csv("outputs/fn_cases.csv", index=False, encoding='utf-8-sig')
    fp_df.to_csv("outputs/fp_cases.csv", index=False, encoding='utf-8-sig')

    # ⭐️ 4. metrics.json에 개수(count) 정보 추가
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
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print("\n☁️ MLflow 클라우드에 성적표를 전송합니다...")
    with mlflow.start_run(run_name="Monitoring_Eval_V1"):
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("dataset_size", len(dataset))
        mlflow.log_param("model_path", args.model_path) # 어떤 모델을 평가했는지 기록
        
        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_precision", prec)
        mlflow.log_metric("eval_recall", rec)
        mlflow.log_metric("eval_f1", f1)
        # ⭐️ 5. MLflow에도 개수 기록 (나중에 그래프로 오탐지 추세 확인 가능)
        mlflow.log_metric("tp_count", tp_count)
        mlflow.log_metric("tn_count", tn_count)
        mlflow.log_metric("fp_count", fp_count)
        mlflow.log_metric("fn_count", fn_count)
        
        mlflow.log_artifact(args.output_csv)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact("outputs/fn_cases.csv")
        mlflow.log_artifact("outputs/fp_cases.csv")
        mlflow.log_artifact("outputs/metrics.json")

    print("=" * 50)
    print("✨ [평가 결과 요약] ✨")
    print(f"정확도 (Accuracy) : {acc:.4f}")
    print(f"정밀도 (Precision): {prec:.4f}")
    print(f"재현율 (Recall)   : {rec:.4f} (⭐️ 전도 감지 모델의 핵심 지표)")
    print(f"F1 점수 (F1 Score): {f1:.4f}")
    print(f"오탐지 (FP)       : {fp_count}건")
    print(f"미탐지 (FN)       : {fn_count}건")
    print("=" * 50)
    
    if rec < 0.90:
        print("🚨 Recall(재현율)이 0.90 미만입니다. 전도를 놓치고 있으니 재학습이 필요합니다!")
    elif f1 < 0.85:
        print("🚨 F1 점수가 0.85 미만입니다. 정상/전도 구분이 불안정하니 재학습을 권장합니다.")
    else:
        print("✅ 모델 성능이 훌륭합니다! 현재 버전을 유지하셔도 좋습니다.")

# ⭐️ 6. argparse 실행부 추가
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="낙상 감지 모델 자동 평가 스크립트")
    parser.add_argument("--manifest", type=str, default="monitoring_eval_manifest.csv", help="평가할 정답지 CSV 경로")
    parser.add_argument("--model_path", type=str, default="model_v1/best_model_v6.pth", help="평가할 모델 가중치 경로 (슬래시 사용)")
    parser.add_argument("--output_csv", type=str, default="outputs/eval_predictions.csv", help="예측 결과 저장 경로")
    parser.add_argument("--threshold", type=float, default=0.5, help="전도 판별 임계값")
    
    args = parser.parse_args()
    main(args)