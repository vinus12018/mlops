import os
import json
import argparse
import pandas as pd
from collections import Counter

def main(args):
    print("재학습 판단(Retrain Trigger) 프로세스를 시작합니다...\n")

    # 1. 파일 존재 여부 확인
    if not os.path.exists(args.metrics_path) or not os.path.exists(args.predictions_path):
        print(f"에러: 평가 결과 파일이 존재하지 않습니다. run_eval.py를 먼저 실행해 주세요.")
        return

    # 2. 평가 지표 로드
    with open(args.metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 3. 예측 결과 로드 (에러 타입 분석용)
    df = pd.read_csv(args.predictions_path)

    retrain_needed = False
    reasons = []

    # 4. 기획안의 5가지 재학습 트리거 조건 검사
    if metrics.get("recall", 1.0) < 0.90:
        reasons.append(f"재현율(Recall) 저하: {metrics.get('recall')} < 0.90")
    
    if metrics.get("f1_score", 1.0) < 0.85:
        reasons.append(f"F1 점수 저하: {metrics.get('f1_score')} < 0.85")
        
    if metrics.get("fn_count", 0) >= 3:
        reasons.append(f"미탐지(FN) 과다: {metrics.get('fn_count')}건 발생")
        
    if metrics.get("fp_count", 0) >= 3:
        reasons.append(f"오탐지(FP) 과다: {metrics.get('fp_count')}건 발생")

    # 특정 에러 패턴 반복 검사 (빈 값이 아닌 에러 타입만 추출)
    errors = df[df["error_type"].notna() & (df["error_type"] != "")]
    error_counts = Counter(errors["error_type"].tolist())

    for error_type, count in error_counts.items():
        if count >= 3:
            reasons.append(f"특정 에러 패턴 반복: {error_type} ({count}건)")

    # 5. 최종 판단
    if len(reasons) > 0:
        retrain_needed = True

    # 6. 결과 저장
    result = {
        "retrain_needed": retrain_needed,
        "retrain_reason": reasons
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    # 7. 진단 결과 출력
    print("=" * 50)
    print("[재학습 필요 여부 진단 결과]")
    print(f"재학습 대상 여부 : {retrain_needed}")
    
    if retrain_needed:
        print("\n[상세 사유]")
        for r in reasons:
            print(f"- {r}")
    else:
        print("\n현재 모델 상태가 안정적이므로 재학습이 필요하지 않습니다.")
    print("=" * 50)
    print(f"진단 결과가 '{args.output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="재학습 트리거 판단 스크립트")
    parser.add_argument("--metrics_path", type=str, default="outputs/metrics.json", help="평가 지표 JSON 경로")
    parser.add_argument("--predictions_path", type=str, default="outputs/eval_predictions.csv", help="예측 결과 CSV 경로")
    parser.add_argument("--output_path", type=str, default="outputs/retrain_trigger_result.json", help="판단 결과 저장 경로")
    
    args = parser.parse_args()
    main(args)