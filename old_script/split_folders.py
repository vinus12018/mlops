import os
import shutil

def split_videos_into_folders(source_folder, chunk_size=32):
    # 1. 원본 폴더가 존재하는지 확인
    if not os.path.exists(source_folder):
        print(f"⚠️ [오류] '{source_folder}' 폴더를 찾을 수 없습니다.")
        return

    # 2. 폴더 내의 영상 파일 목록만 가져오기 (알파벳/숫자 순 정렬)
    # 다른 확장자가 있다면 튜플 안에 추가하세요.
    videos = [f for f in os.listdir(source_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    videos.sort() 
    
    total_videos = len(videos)
    print(f"총 {total_videos}개의 영상을 발견했습니다. 분할 작업을 시작합니다...")

    if total_videos == 0:
        print("이동할 영상이 없습니다.")
        return

    # 3. 129개씩 나누어 파일 이동하기
    for i, video in enumerate(videos):
        # 들어갈 폴더 번호 계산 (0~128은 1번 폴더, 129~257은 2번 폴더...)
        folder_index = (i // chunk_size) + 1
        
        # 새 폴더 이름 지정 (예: C:/Users/qoiop/test/data_part_1)
        # 원본 폴더와 같은 위치에 만들어집니다.
        target_folder = f"{source_folder}_part_{folder_index}"
        
        # 폴더가 없다면 새로 생성
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print(f"📁 새 폴더 생성됨: {target_folder}")
        
        # 원본 파일 경로와 이동할 목적지 경로
        source_path = os.path.join(source_folder, video)
        target_path = os.path.join(target_folder, video)
        
        # 파일 '이동' (원본 폴더에서 사라지고 새 폴더로 넘어감)
        # 만약 원본을 남기고 싶다면 shutil.move 대신 shutil.copy2 를 사용하세요.
        shutil.move(source_path, target_path)
        
    print(f"\n✨ 분할 완료! 총 {folder_index}개의 폴더로 깔끔하게 나뉘었습니다.")

# --- 실행 ---
# 실제 데이터가 들어있는 폴더 경로를 입력하세요. (끝에 슬래시 / 없이 작성)
source_dir = "C:/Users/qoiop/Downloads/mlops/data" 

# 함수 실행 (129개씩 묶기)
split_videos_into_folders(source_dir, chunk_size=32)