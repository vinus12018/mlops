import os
import shutil
import random
from collections import defaultdict

def create_sliding_window_dataset():
    # 1. 경로 설정
    source_dir = "C:/Users/qoiop/Downloads/mlops/final_images_merge"
    target_base = "C:/Users/qoiop/Downloads/mlops/dataset"

    print("🔍 19,995장의 원본 이미지를 비디오(클립) 단위로 분석합니다...")

    # 2. 비디오 이름으로 그룹화 및 프레임 번호 추출
    video_groups = defaultdict(list)
    for file in os.listdir(source_dir):
        if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
            try:
                # 예: ..._M2_00.jpg -> 비디오 이름과 프레임 번호(00) 분리
                parts = file.split('_')
                frame_num_str = parts[-1].replace('.jpg', '').replace('.JPG', '')
                frame_num = int(frame_num_str)
                video_name = "_".join(parts[:-1])
                
                # (파일명, 프레임번호) 형태로 저장
                video_groups[video_name].append((file, frame_num))
            except ValueError:
                continue

    # 3. 비디오별로 프레임 번호 순으로 정렬 (시간의 흐름 보장)
    for v_name in video_groups:
        video_groups[v_name].sort(key=lambda x: x[1])

    # 4. Train(8) : Test(2) 분리 (비디오 단위로 분리하여 데이터 누수 방지)
    video_names = list(video_groups.keys())
    random.seed(42) # 재현성 고정
    random.shuffle(video_names)
    
    split_idx = int(len(video_names) * 0.8)
    train_vids = video_names[:split_idx]
    test_vids = video_names[split_idx:]

    # 5. 슬라이딩 윈도우 기반 시퀀스 생성 함수
    seq_counter = 0
    def process_videos_with_sliding_window(v_list, split_name):
        nonlocal seq_counter
        
        for v_name in v_list:
            frames = video_groups[v_name]
            
            # 한 영상에서 10장짜리 슬라이딩 윈도우 추출 (1칸씩 이동)
            # 31장 영상 기준: 31 - 10 + 1 = 총 22개의 시퀀스 생성
            for i in range(len(frames) - 10 + 1):
                window = frames[i : i + 10]
                
                # 라벨링 로직: 윈도우 내에 15번 프레임 이상(쓰러지는 시점)이 하나라도 있다면 Fall
                is_fall = any(frame_num >= 15 for _, frame_num in window)
                label = "fall" if is_fall else "stand"
                
                seq_counter += 1
                seq_folder_name = f"seq_{seq_counter:05d}"
                target_dir = os.path.join(target_base, split_name, label, seq_folder_name)
                os.makedirs(target_dir, exist_ok=True)
                
                # 10장의 이미지 복사
                for file_name, _ in window:
                    src = os.path.join(source_dir, file_name)
                    dst = os.path.join(target_dir, file_name)
                    shutil.copy2(src, dst)
                    
                # 진행 상황 출력 (500개 생성 시점마다)
                if seq_counter % 500 == 0:
                    print(f"   ⏳ [진행 중] {seq_counter}번째 시퀀스(10장) 묶음 생성 완료...")

    # 6. 실행
    print("🚀 슬라이딩 윈도우 전처리 및 복사 작업을 시작합니다. (약 14,000개 시퀀스 예상)")
    
    process_videos_with_sliding_window(train_vids, "train")
    process_videos_with_sliding_window(test_vids, "test")

    print("\n✨ 완벽합니다! 슬라이딩 윈도우 기반 데이터 파이프라인 구축이 완료되었습니다.")
    print(f"총 {seq_counter}개의 시퀀스 데이터가 생성되었습니다.")
    print("이제 터미널에서 'python train.py'를 실행하여 가장 똑똑한 LSTM을 만들어 보세요!")

if __name__ == '__main__':
    create_sliding_window_dataset()