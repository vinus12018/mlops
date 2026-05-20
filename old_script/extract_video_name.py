import os
import pandas as pd
import re  # 정규표현식(글자/숫자 분리) 모듈 추가

# 💡 사람(윈도우 탐색기) 방식의 정렬을 위한 마법의 함수
def natural_sort_key(s):
    # 파일명에서 숫자 부분만 진짜 '숫자(int)'로 취급하여 크기 비교를 하도록 만듭니다.
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def extract_video_names_only(video_folders, output_filename):
    video_files = []
    
    for folder in video_folders:
        if not os.path.exists(folder):
            print(f"⚠️ [건너뜀] 폴더를 찾을 수 없습니다: {folder}")
            continue
            
        # 1. mp4 파일만 골라내기
        files_in_folder = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]
        
        # 2. [핵심 수정] 파이썬식 정렬 대신, 윈도우 파일 탐색기식(자연 정렬) 키를 적용!
        files_in_folder.sort(key=natural_sort_key) 
        
        # 3. 전체 목록에 추가
        video_files.extend(files_in_folder)
    
    data = {
        'Video_Name': video_files,
        'Cut_Time': [''] * len(video_files)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"✨ 완벽 정렬! 총 {len(video_files)}개의 영상 제목이 윈도우 탐색기와 100% 똑같은 순서로 저장되었습니다.")

# --- 실행 부분 ---
video_dirs = [
    "C:/Users/qoiop/Downloads/mlops/data_part_13",
    "C:/Users/qoiop/Downloads/mlops/data_part_14",
    "C:/Users/qoiop/Downloads/mlops/data_part_15",
    "C:/Users/qoiop/Downloads/mlops/data_part_16"
]

output_path = "C:/Users/qoiop/Downloads/mlops/cut_data_minjae.csv"

extract_video_names_only(video_dirs, output_path)