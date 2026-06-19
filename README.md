# MLOps 기반 CCTV 영상 낙상 감지 모델 및 자동화 파이프라인 구축

## 1. 프로젝트 요약

본 프로젝트는 **MLOps 기반 CCTV 영상 낙상 감지 모델 및 자동화 파이프라인 구축**을 목표로 한다.

무인 매장, 편의점, 실내 CCTV 환경에서 사람이 넘어지는 전도/낙상 상황을 감지하고, 모델 배포 이후에도 예측 로그를 저장하여 모델 성능을 지속적으로 모니터링하는 구조를 설계하였다.

초기에는 단순히 영상을 업로드하면 모델이 전도 여부를 예측하고, 사람이 예측 결과를 확인하는 방식으로 진행하였다. 이후 교수님 피드백을 반영하여, AIHub 라벨링 XML 데이터를 활용해 정답 라벨인 `true_label`을 만들고, 모델 예측값과 자동으로 비교하는 방향으로 개선하였다.

즉, 본 프로젝트는 단순 모델 학습 프로젝트가 아니라 다음 흐름을 포함하는 MLOps 프로젝트이다.

<pre><code>영상 데이터 수집
→ 라벨링 XML 기반 true_label 생성
→ 전도/비전도 클립 구성
→ 모델 학습 및 추론
→ 예측 로그 DB 저장
→ 모니터링 대시보드 확인
→ prediction과 true_label 자동 비교
→ TP/TN/FP/FN 자동 판정
→ Precision, Recall, F1-score 계산
→ 오류 유형 분석
→ 재학습 필요 여부 판단</code></pre>

---

## 2. 프로젝트 배경

무인 매장이나 CCTV 기반 환경에서는 고객이 넘어지는 사고가 발생해도 즉시 사람이 확인하기 어렵다. 따라서 CCTV 영상에서 전도 상황을 자동으로 감지하는 AI 모델이 필요하다.

하지만 낙상 감지 모델은 단순히 “전도 영상만 잘 맞히는 것”으로 충분하지 않다. 실제 환경에서는 사람이 정상적으로 걷거나, 물건을 줍거나, 쭈그려 앉거나, 상품을 확인하는 행동도 자주 발생한다. 이런 정상 행동을 전도로 잘못 감지하면 오탐이 발생한다.

따라서 본 프로젝트에서는 전도 데이터뿐 아니라 정상/비전도 데이터를 함께 고려하여, 모델이 실제 운영 환경에서 전도와 정상 행동을 구분할 수 있도록 하는 MLOps 기반 모니터링 구조를 설계하였다.

---

## 3. 프로젝트 목표

### 3.1 핵심 목표

- CCTV 영상에서 전도/낙상 상황을 감지한다.
- 모델 예측 결과를 DB에 자동 저장한다.
- 저장된 추론 로그를 대시보드에서 확인한다.
- 라벨링 데이터 기반으로 모델 예측이 맞았는지 자동 판별한다.
- TP, TN, FP, FN을 자동 계산한다.
- Accuracy, Precision, Recall, F1-score를 자동 계산한다.
- 반복되는 오류 유형을 분석하여 재학습 후보를 도출한다.
- 재학습 전후 모델 성능을 비교할 수 있는 구조를 만든다.

### 3.2 프로젝트에서 중요하게 본 지표

전도 감지 모델에서는 단순 Accuracy보다 **Recall**이 중요하다.

Recall은 실제 전도 상황 중 모델이 전도로 감지한 비율이다.

<pre><code>Recall = TP / (TP + FN)</code></pre>

전도 감지 모델에서 FN은 실제로 사람이 넘어졌는데 모델이 정상으로 판단한 경우이다. 이는 안전 관점에서 위험하므로, 본 프로젝트에서는 FN을 줄이는 것을 중요한 목표로 두었다.

---

## 4. 팀 구성

총 투입 인원: 5명

| 이름 | 역할 | 주요 담당 |
|---|---|---|
| 박민준 | 모니터링 / 데이터 검증 / 오류 분석 | 추론 로그 확인, 라벨링 데이터 기반 자동 평가 구조 설계, 오탐/미탐 분석, 재학습 판단 기준 정리 |
| 이찬호 | 프론트엔드 / 백엔드 / 배포 | Streamlit 대시보드 구현, Supabase DB 연동, Render 배포, GitHub 기반 자동 배포 구조 구축 |
| 김호준 | 모델 학습 / 재학습 | 전도 감지 모델 학습, 라벨링 XML 기반 manifest 생성, 재학습 및 평가 코드 구현 |
| 원성민 | MLOps 흐름 정리 / 파이프라인 보조 | 전체 MLOps 구조 정리, 재학습 흐름 문서화, 발표 및 산출물 정리 |
| 김민재 | 데이터 / 발표 / 문서 보조 | 데이터 정리, 발표 자료 보조, 결과 정리 |

> 실제 이름과 역할은 최종 팀 상황에 맞게 수정 예정

---

## 5. 전체 시스템 구조

<pre><code>[AIHub 데이터 수집]
        ↓
[원천데이터 + 라벨링 XML 확보]
        ↓
[XML 파싱]
        ↓
[true_label 생성]
        ↓
[manifest.csv 생성]
        ↓
[전도/비전도 영상 클립 구성]
        ↓
[모델 학습 및 추론]
        ↓
[Supabase DB에 예측 로그 저장]
        ↓
[Streamlit 모니터링 대시보드]
        ↓
[prediction과 true_label 자동 비교]
        ↓
[TP/TN/FP/FN 자동 판정]
        ↓
[Precision / Recall / F1-score 계산]
        ↓
[오류 유형 분석]
        ↓
[재학습 필요 여부 판단]</code></pre>

---

## 6. 사용 기술 스택

### 6.1 모델 및 데이터 처리

- Python
- OpenCV
- PyTorch
- ResNet + LSTM 기반 영상 분류 모델
- AIHub 데이터셋
- XML 라벨링 데이터
- CSV manifest 기반 데이터 관리

### 6.2 백엔드 / DB / 배포

- Supabase
- Streamlit
- Render
- GitHub

### 6.3 MLOps / 실험 관리

- MLflow
- 모델 버전 관리
- 실험 결과 기록
- 평가 지표 기록
- 재학습 전후 성능 비교

---

## 7. 데이터 구성

### 7.1 전도 데이터

전도 데이터는 AIHub CCTV 이상행동/전도 관련 데이터를 사용하였다.

초기에는 원천 데이터만 활용하여 사람이 직접 영상을 확인하고 전도 여부를 판단했으나, 이후 라벨링 XML 데이터를 활용하는 방향으로 변경하였다.

전도 데이터의 라벨링 XML에는 다음과 같은 정보가 포함되어 있었다.

<pre><code>fall_start
fall_end</code></pre>

예시:

<pre><code>fall_start = 137
fall_end = 179</code></pre>

이를 통해 해당 영상 또는 클립이 전도 상황을 포함하고 있음을 자동으로 판단할 수 있다.

### 7.2 정상 / 비전도 데이터

전도 데이터만으로는 모델이 정상 행동과 전도 상황을 구분하기 어렵다.

따라서 다음과 같은 정상/비전도 행동 데이터를 추가로 확보하는 방향으로 개선하였다.

- 정상 보행
- 상품 확인
- 구매 행동
- 물건 집기
- 쭈그려 앉기
- 서 있기
- 화면 진입/이탈
- 진열대 근처 이동

정상 데이터 후보로는 AIHub의 **실내(편의점, 매장) 구매행동 데이터**를 검토하였다.  
해당 데이터는 무인 매장 환경과 유사하기 때문에, 모델이 정상 매장 행동을 전도로 오탐하는지 확인하는 데 적합하다.

---

## 8. 기존 방식의 문제점

초기에는 모델이 예측한 결과를 사람이 직접 보고 다음 값을 입력했다.

<pre><code>admin_confirm_label
error_type</code></pre>

예를 들어 모델이 “정상”이라고 예측했는데 실제로는 전도라면 사람이 직접 확인 후 다음과 같이 입력했다.

<pre><code>admin_confirm_label = 1
error_type = FN_계단전도</code></pre>

하지만 이 방식은 사람이 예측 결과를 보고 직접 맞았는지 판단해야 하므로, 자동화된 모니터링이라고 보기 어렵다.

교수님 피드백 이후, 라벨링 데이터의 정답 라벨과 모델 예측값을 자동으로 비교하는 구조가 필요하다는 점을 확인하였다.

---

## 9. 개선 방향: 라벨링 데이터 기반 자동 평가

### 9.1 핵심 변경점

기존 방식:

<pre><code>영상 업로드
→ 모델 추론
→ prediction 저장
→ 사람이 결과를 직접 확인
→ admin_confirm_label 수동 입력
→ error_type 수동 입력
→ 수동 집계</code></pre>

개선 방식:

<pre><code>라벨링 XML 확보
→ true_label 생성
→ manifest.csv 생성
→ 모델 추론
→ prediction 생성
→ true_label과 prediction 자동 비교
→ TP/TN/FP/FN 자동 판정
→ Precision / Recall / F1 자동 계산
→ 재학습 필요 여부 자동 판단</code></pre>

### 9.2 true_label 기준

<pre><code>true_label = 1 → 전도
true_label = 0 → 정상 / 비전도</code></pre>

### 9.3 predicted_label 기준

<pre><code>prediction = 전도 상황 감지 → predicted_label = 1
prediction = 정상 → predicted_label = 0</code></pre>

### 9.4 자동 판정 기준

<pre><code>predicted_label=1, true_label=1 → TP
predicted_label=1, true_label=0 → FP
predicted_label=0, true_label=0 → TN
predicted_label=0, true_label=1 → FN</code></pre>

---

## 10. Manifest 기반 데이터 관리

라벨링 XML을 직접 모델에 넣는 것이 아니라, XML에서 필요한 정보를 추출하여 `manifest.csv`를 생성한다.

예시:

| clip_id | video_path | label_file | true_label | behavior_type | start_frame | end_frame | start_sec | end_sec | source_dataset | split | memo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fall_001 | clips/fall_001.mp4 | labels/fall_001.xml | 1 | 계단전도 | 137 | 179 | 4.57 | 5.97 | AIHub_CCTV_Fall | test | fall_start/fall_end 존재 |
| normal_001 | clips/normal_001.mp4 | labels/normal_001.xml | 0 | 정상보행 |  |  |  |  | AIHub_Store_Action | test | 전도 아님 |
| normal_002 | clips/normal_002.mp4 | labels/normal_002.xml | 0 | 상품확인 |  |  |  |  | AIHub_Store_Action | test | 전도 아님 |

### 주요 컬럼 설명

| 컬럼 | 설명 |
|---|---|
| clip_id | 클립 고유 ID |
| video_path | 모델 입력 영상 경로 |
| label_file | 라벨링 XML 파일 경로 |
| true_label | 실제 정답 라벨 |
| behavior_type | 세부 행동 유형 |
| start_frame | 전도 시작 프레임 |
| end_frame | 전도 종료 프레임 |
| start_sec | 전도 시작 시간 |
| end_sec | 전도 종료 시간 |
| source_dataset | 데이터 출처 |
| split | train / valid / test 구분 |
| memo | 기타 설명 |

---

## 11. 영상 클립 생성 방식

전도 데이터의 경우 XML에서 `fall_start`, `fall_end`를 가져와 전도 구간을 기준으로 클립을 생성한다.

예시:

<pre><code>fps = 30
fall_start = 137
fall_end = 179

start_sec = 137 / 30 = 약 4.57초
end_sec = 179 / 30 = 약 5.97초</code></pre>

전도 순간 앞뒤 맥락을 포함하기 위해 다음과 같이 클립을 생성한다.

<pre><code>clip_start_sec = max(0, start_sec - 5)
clip_end_sec = end_sec + 5</code></pre>

즉, 전도 발생 전후를 포함한 10초 내외 클립을 생성한다.

정상 데이터는 정상보행, 상품확인, 구매행동 등의 구간을 5~10초 단위로 잘라 `true_label=0`으로 관리한다.

---

## 12. 모델 평가 지표

### 12.1 Accuracy

<pre><code>Accuracy = (TP + TN) / 전체</code></pre>

전체 예측 중 맞힌 비율이다.

### 12.2 Precision

<pre><code>Precision = TP / (TP + FP)</code></pre>

모델이 전도라고 예측한 것 중 실제 전도였던 비율이다.

### 12.3 Recall

<pre><code>Recall = TP / (TP + FN)</code></pre>

실제 전도 중 모델이 전도로 감지한 비율이다.  
전도 감지 시스템에서는 실제 전도를 놓치지 않는 것이 중요하므로 Recall이 특히 중요하다.

### 12.4 F1-score

<pre><code>F1 = 2 * Precision * Recall / (Precision + Recall)</code></pre>

Precision과 Recall의 균형을 보는 지표이다.

---

## 13. 오류 유형 관리

모델 예측과 실제 라벨을 비교하여 오류 유형을 자동으로 생성한다.

<pre><code>eval_result = FN → error_type = "FN_" + behavior_type
eval_result = FP → error_type = "FP_" + behavior_type
TP / TN → error_type 없음</code></pre>

예시:

<pre><code>true_label=1, predicted_label=0, behavior_type=계단전도
→ eval_result=FN
→ error_type=FN_계단전도</code></pre>

<pre><code>true_label=0, predicted_label=1, behavior_type=물건집기
→ eval_result=FP
→ error_type=FP_물건집기</code></pre>

---

## 14. 재학습 판단 기준

다음 조건 중 하나라도 만족하면 재학습 검토 대상으로 본다.

<pre><code>Recall < 0.90
F1-score < 0.85
같은 error_type이 3건 이상 반복
FN_계단전도 3건 이상
FP_물건집기 3건 이상
FP_쭈그려앉기 3건 이상</code></pre>

예시:

<pre><code>FN_계단전도 3건 이상 발생
→ 계단 전도 hard example 추가 필요
→ 재학습 후보로 전달</code></pre>

<pre><code>FP_물건집기 3건 이상 발생
→ 물건 집기 행동을 전도로 오탐
→ 정상/비전도 데이터 보강 필요</code></pre>

---

## 15. 모니터링 대시보드

현재 Streamlit 기반 대시보드에서는 다음 기능을 제공한다.

### 15.1 현재 구현된 기능

- 영상 업로드
- 모델 추론 결과 확인
- prediction 확인
- probability 확인
- Supabase DB에 로그 저장
- admin_confirm_label 수정
- error_type 수정

### 15.2 개선 예정 기능

라벨링 데이터 기반 자동 평가 기능을 추가한다.

추가 예정 필드:

<pre><code>true_label
predicted_label
eval_result
behavior_type
label_file
source_dataset
start_frame
end_frame
start_sec
end_sec
model_version
threshold_applied
run_id</code></pre>

추가 예정 화면:

- 총 평가 수
- 실제 전도 수
- 실제 정상 수
- TP / TN / FP / FN
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- FN 목록
- FP 목록
- error_type별 발생 횟수
- 재학습 필요 여부

---

## 16. Supabase DB 구조 개선안

### 16.1 eval_manifest 테이블

정답 라벨 정보 저장용 테이블이다.

| 컬럼 | 설명 |
|---|---|
| clip_id | 클립 ID |
| filename | 파일명 |
| video_path | 영상 경로 |
| label_file | XML 라벨 파일 |
| true_label | 실제 정답 라벨 |
| behavior_type | 행동 유형 |
| start_frame | 시작 프레임 |
| end_frame | 종료 프레임 |
| start_sec | 시작 시간 |
| end_sec | 종료 시간 |
| source_dataset | 데이터 출처 |
| memo | 기타 설명 |

### 16.2 inference_logs 테이블

모델 추론 결과 저장용 테이블이다.

| 컬럼 | 설명 |
|---|---|
| id | 로그 ID |
| created_at | 생성 시간 |
| clip_id | 클립 ID |
| filename | 파일명 |
| prediction | 모델 예측 문자열 |
| predicted_label | 모델 예측 숫자값 |
| probability | 전도 확률 |
| true_label | 실제 정답 |
| eval_result | TP/TN/FP/FN |
| error_type | 오류 유형 |
| model_version | 모델 버전 |
| threshold_applied | 적용 threshold |
| inference_time_ms | 추론 시간 |

---

## 17. MLflow 활용 계획

MLflow는 학습 실험 관리와 모델 버전 비교에 사용한다.

### 17.1 기록할 Metrics

<pre><code>eval_accuracy
eval_precision
eval_recall
eval_f1
tp_count
tn_count
fp_count
fn_count</code></pre>

### 17.2 기록할 Parameters

<pre><code>model_type
threshold
sequence_length
input_size
train_manifest_path
eval_manifest_path
dataset_version</code></pre>

### 17.3 기록할 Artifacts

<pre><code>monitoring_eval_manifest.csv
eval_predictions.csv
metrics.json
confusion_matrix.png
fn_cases.csv
fp_cases.csv</code></pre>

이를 통해 기존 모델과 재학습 모델의 성능을 비교하고, 재학습 후 실제로 성능이 개선되었는지 확인할 수 있다.

---

## 18. 내가 담당한 작업

본 프로젝트에서 나는 **모니터링 담당**으로 참여하였다.

### 18.1 주요 담당 업무

- 모델 추론 로그 확인
- Supabase DB에 저장된 예측 결과 검토
- Streamlit 라벨링 대시보드 검증
- admin_confirm_label / error_type 입력 기준 정리
- FP / FN 오류 유형 분류
- 계단 전도 미탐 사례 분석
- 정상/비전도 데이터 부족 문제 발견
- AIHub 라벨링 XML 활용 방향 제안
- true_label 기반 자동 평가 구조 설계
- manifest.csv 구조 설계
- TP/TN/FP/FN 자동 판정 기준 정리
- Precision / Recall / F1-score 계산 기준 정리
- 재학습 필요 조건 정리
- 팀원별 수정 방향 정리

### 18.2 내가 발견한 주요 문제

초기 모니터링 구조는 사람이 모델 예측 결과를 직접 보고 맞았는지 틀렸는지 판단하는 방식이었다.

이 방식은 운영 로그 검수에는 사용할 수 있지만, 자동화된 MLOps 모니터링 구조라고 보기 어렵다는 문제가 있었다.

이를 해결하기 위해 라벨링 XML에서 `true_label`을 추출하고, 모델 예측값과 자동 비교하는 구조를 제안하였다.

### 18.3 내가 제안한 개선 방향

<pre><code>기존:
prediction 확인
→ 사람이 직접 맞음/틀림 판단
→ admin_confirm_label 입력
→ 수동 집계

개선:
라벨링 XML 파싱
→ true_label 생성
→ prediction과 자동 비교
→ TP/TN/FP/FN 자동 생성
→ 지표 자동 계산
→ 재학습 후보 자동 도출</code></pre>

---

## 19. 현재까지의 실험 결과

초기 테스트에서는 계단 전도 데이터 중심으로 모델을 확인하였다.

대표적으로 다음 오류가 반복적으로 발생하였다.

<pre><code>FN_계단전도</code></pre>

즉, 실제로 계단에서 전도 상황이 발생했지만 모델이 정상으로 예측한 사례가 있었다.

초기 대표 로그 기준으로는 실제 전도 데이터에서 일부 미탐이 발생했으며, 정상 데이터 수가 부족하여 오탐률을 일반화하기는 어려웠다.

따라서 다음 결론을 도출하였다.

<pre><code>현재 모델은 계단 전도 일부 케이스에서 미탐이 발생한다.
정상/비전도 데이터가 부족하므로 오탐 여부를 충분히 판단하기 어렵다.
정상 보행, 상품 확인, 물건 집기 등 비전도 데이터를 추가해야 한다.
라벨링 데이터 기반 자동 평가 구조로 변경해야 한다.</code></pre>

---

## 20. 변경 전후 비교

### 20.1 변경 전

<pre><code>영상 업로드
→ 모델 추론
→ prediction 저장
→ 사람이 결과를 직접 확인
→ admin_confirm_label 수동 입력
→ error_type 수동 입력
→ 수동으로 지표 계산</code></pre>

### 20.2 변경 후

<pre><code>라벨링 XML 확보
→ true_label 생성
→ manifest.csv 생성
→ 모델 추론
→ prediction 생성
→ true_label과 prediction 자동 비교
→ TP/TN/FP/FN 자동 판정
→ Precision / Recall / F1 자동 계산
→ 재학습 필요 여부 자동 판단</code></pre>

---

## 21. 향후 작업 계획

### 21.1 라벨링 XML 파싱

- 전도 XML에서 fall_start / fall_end 추출
- true_label=1 생성
- start_frame / end_frame 저장

### 21.2 정상 데이터 확보

- AIHub 실내 편의점/매장 구매행동 데이터 다운로드
- 정상보행, 상품확인, 물건집기 등 true_label=0 데이터 확보

### 21.3 manifest.csv 생성

- 전도 데이터와 정상 데이터를 하나의 manifest로 통합
- train / valid / test split 구성

### 21.4 자동 평가 코드 구현

- manifest 기반 일괄 추론
- prediction과 true_label 비교
- TP/TN/FP/FN 자동 생성
- Precision / Recall / F1-score 자동 계산

### 21.5 대시보드 개선

- true_label 기반 자동 평가 결과 표시
- Confusion Matrix 표시
- FN/FP 목록 표시
- 재학습 필요 여부 표시

### 21.6 재학습 및 성능 비교

- 반복 오류 유형 기반 데이터 보강
- 재학습 수행
- MLflow에 기존 모델과 새 모델 성능 비교 기록

---

## 22. 프로젝트를 통해 배운 점

이번 프로젝트를 통해 단순히 모델을 학습시키는 것보다, 운영 중인 모델을 어떻게 관리하고 개선할 것인지가 더 중요하다는 것을 배웠다.

특히 모니터링 단계에서는 사람이 예측 결과를 직접 보고 판단하는 것보다, 라벨링 데이터 기반으로 모델 예측값과 정답을 자동 비교하는 구조가 필요하다는 점을 알게 되었다.

또한 전도 감지 모델은 전도 데이터만 많이 넣는다고 좋아지는 것이 아니라, 정상 보행, 상품 확인, 물건 집기, 쭈그려 앉기 같은 비전도 행동을 함께 학습해야 실제 환경에서 오탐을 줄일 수 있다는 점을 확인하였다.

MLOps 관점에서는 다음 요소들이 중요하다는 것을 경험했다.

- 데이터 라벨 관리
- 실험 기록
- 모델 버전 관리
- 운영 로그 저장
- 성능 모니터링
- 오류 유형 분석
- 재학습 판단 기준
- 자동화된 평가 파이프라인

---

## 23. 폴더 구조 예시

<pre><code>MLOps_Monitoring/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── fall_videos/
│   │   ├── normal_videos/
│   │   └── labels_xml/
│   │
│   ├── clips/
│   │   ├── fall/
│   │   └── normal/
│   │
│   └── manifests/
│       ├── train_manifest.csv
│       ├── valid_manifest.csv
│       ├── test_manifest.csv
│       └── monitoring_eval_manifest.csv
│
├── scripts/
│   ├── parse_aihub_xml.py
│   ├── make_clips_from_manifest.py
│   ├── run_eval.py
│   └── calculate_metrics.py
│
├── outputs/
│   ├── eval_predictions.csv
│   ├── metrics.json
│   ├── fn_cases.csv
│   ├── fp_cases.csv
│   └── confusion_matrix.png
│
└── docs/
    ├── monitoring_plan.md
    ├── retraining_policy.md
    └── error_type_definition.md</code></pre>

---

## 24. 포트폴리오 관점에서의 핵심 어필 포인트

이 프로젝트에서 내가 어필할 수 있는 부분은 단순히 모델 학습에 참여했다는 것이 아니라, **모델 운영 이후의 모니터링 구조를 설계했다는 점**이다.

특히 다음 내용을 강조할 수 있다.

<pre><code>1. 모델 예측 로그를 운영 DB에 저장하는 구조 이해
2. 라벨링 데이터 기반 true_label 설계
3. prediction과 true_label 자동 비교 구조 설계
4. TP/TN/FP/FN 자동 판정 기준 정의
5. Recall, Precision, F1-score 기반 재학습 판단 기준 설계
6. 오류 유형별 재학습 후보 도출
7. Streamlit + Supabase 기반 모니터링 대시보드 개선 방향 제안
8. MLflow를 활용한 재학습 전후 성능 비교 계획 수립</code></pre>

---

## 25. 최종 요약

본 프로젝트는 무인 매장 내 전도/낙상 감지 모델을 개발하고, 배포 이후 모델의 예측 결과를 지속적으로 모니터링하는 MLOps 프로젝트이다.

초기에는 사람이 모델 예측 결과를 수동으로 확인하는 방식이었지만, 교수님 피드백 이후 AIHub 라벨링 XML을 활용하여 true_label을 생성하고, 모델 prediction과 자동 비교하는 구조로 개선 방향을 전환하였다.

이를 통해 단순 수동 검수에서 벗어나, 라벨 기반 자동 평가, 지표 계산, 오류 유형 분석, 재학습 판단까지 이어지는 MLOps 모니터링 파이프라인을 설계하였다.
