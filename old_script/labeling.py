import os
import shutil

def split_images_by_frame_number():
    # 1. 작업할 최상위 폴더 경로 설정
    base_dir = "C:/Users/qoiop/Downloads/mlops/final_images"
    
    # 2. 이동할 대상 폴더 경로 설정 및 생성
    stand_dir = os.path.join(base_dir, "stand")
    fall_dir = os.path.join(base_dir, "fall")
    
    os.makedirs(stand_dir, exist_ok=True)
    os.makedirs(fall_dir, exist_ok=True)

    # 3. 폴더 내의 모든 파일 목록 가져오기
    files = [f for f in os.listdir(base_dir) if f.lower().endswith('.jpg')]
    
    print(f"총 {len(files)}장의 이미지 분류를 시작합니다...")
    
    success_stand = 0
    success_fall = 0

    for filename in files:
        # 4. 파일 이름에서 끝부분 번호만 추출 (예: '..._00.jpg' -> '00' -> 0)
        try:
            # 뒷부분 '_XX.jpg' 에서 숫자만 분리
            num_str = filename.split('_')[-1].replace('.jpg', '').replace('.JPG', '')
            frame_num = int(num_str)
            
            # 원본 파일 경로
            src_path = os.path.join(base_dir, filename)
            
            # 5. 번호에 따라 폴더 이동
            if 0 <= frame_num <= 14:
                dst_path = os.path.join(stand_dir, filename)
                shutil.move(src_path, dst_path)
                success_stand += 1
                
            elif 15 <= frame_num <= 30:
                dst_path = os.path.join(fall_dir, filename)
                shutil.move(src_path, dst_path)
                success_fall += 1
                
        except ValueError:
            print(f"⚠️ 건너뜀: 파일 이름 형식을 인식할 수 없습니다. ({filename})")

    # 6. 결과 출력
    print("\n✨ 분류 작업이 완료되었습니다!")
    print(f"👉 stand 폴더로 이동됨 (00~14): {success_stand}장")
    print(f"👉 fall 폴더로 이동됨 (15~30): {success_fall}장")

if __name__ == '__main__':
    split_images_by_frame_number()