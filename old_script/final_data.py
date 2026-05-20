import cv2
import os
import pandas as pd

def extract_images_directly(video_folders, excel_file, output_folder, extract_fps=3):
    # 1. 저장할 폴더가 없으면 만들기
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. 파일 읽기 (엑셀 함수가 아닌 CSV 함수로 수정 완료!)
    try:
        df = pd.read_csv(excel_file, encoding='utf-8-sig')
    except Exception as e:
        df = pd.read_excel(excel_file) # 혹시나 엑셀 파일일 경우를 대비한 백업

    for index, row in df.iterrows():
        video_name = row['Video_Name']
        fall_time = row['Cut_Time'] # CSV의 쓰러진 시간(초)

        if pd.isna(fall_time) or fall_time == '':
            continue

        # 3. 여러 폴더(part_17 ~ 21)를 돌면서 해당 영상이 어디 있는지 찾기
        video_path = None
        for folder in video_folders:
            temp_path = os.path.join(folder, video_name)
            if os.path.exists(temp_path):
                video_path = temp_path
                break 

        # 모든 폴더를 다 뒤졌는데도 없으면 건너뛰기
        if not video_path:
            print(f"⚠️ [{video_name}] 파일 없음 (지정된 폴더들에 존재하지 않음)")
            continue

        # 4. OpenCV로 원본 영상 열기
        cap = cv2.VideoCapture(video_path)
        original_fps = round(cap.get(cv2.CAP_PROP_FPS))
        
        if original_fps == 0:
            continue

        # 5. 자를 구간(초)을 '프레임 번호'로 변환
        start_time = max(0, fall_time - 5)
        end_time = fall_time + 5
        
        start_frame = int(start_time * original_fps)
        end_frame = int(end_time * original_fps)
        
        # 초당 추출할 간격
        frame_interval = max(1, original_fps // extract_fps)

        # 6. 영상의 처음부터 보지 않고, 시작 프레임으로 단숨에 건너뛰기
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        saved_count = 0

        print(f"⏳ [{video_name}] 이미지 추출 중... ({start_time}초 ~ {end_time}초 구간)")

        # 7. 시작점부터 끝점까지만 읽으면서 이미지 저장
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break # 영상 끝

            # 지정한 간격일 때만 이미지(jpg)로 저장
            if current_frame % frame_interval == 0:
                save_name = f"{video_name.split('.')[0]}_{saved_count:02d}.jpg"
                save_path = os.path.join(output_folder, save_name)
                cv2.imwrite(save_path, frame)
                saved_count += 1

            current_frame += 1

        cap.release()
        print(f"✨ [{video_name}] 완료! 총 {saved_count}장 추출됨.\n")

# --- 실행 ---
# 요청하신 대로 17번부터 21번 폴더까지 리스트로 묶었습니다.
video_dirs = [
    "C:/Users/qoiop/Downloads/mlops/data_part_13",
    "C:/Users/qoiop/Downloads/mlops/data_part_14",
    "C:/Users/qoiop/Downloads/mlops/data_part_15",
    "C:/Users/qoiop/Downloads/mlops/data_part_16"
]
excel_path = "C:/Users/qoiop/Downloads/mlops/cut_data_minjae.csv"                     
output_dir = "C:/Users/qoiop/Downloads/mlops/final_images"         

# 1초에 3장씩 추출하도록 실행!
extract_images_directly(video_dirs, excel_path, output_dir, extract_fps=3)