import os
import shutil

def merge_folders(target_folder, num_parts=5):
    # 1. 합쳐질 최종 폴더(data)가 없으면 생성
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"📁 대상 폴더 생성됨: {target_folder}")

    total_moved = 0

    # 2. part_1 부터 part_5 까지 순회하며 파일 꺼내오기
    for i in range(1, num_parts + 1):
        # 분할 폴더 이름 조합 (예: C:/Users/qoiop/test/data_part_1)
        part_folder = f"{target_folder}_part_{i}"

        # 폴더가 존재하는지 확인
        if not os.path.exists(part_folder):
            print(f"⚠️ [건너뜐] 폴더를 찾을 수 없습니다: {part_folder}")
            continue

        # 폴더 안의 파일 목록 가져오기
        files = os.listdir(part_folder)
        
        if not files:
            print(f"ℹ️ [빈 폴더] 파일이 없습니다: {part_folder}")
            os.rmdir(part_folder) # 비어있으면 폴더만 삭제
            continue

        print(f"📂 [{part_folder}]에서 {len(files)}개의 영상 복귀 중...")
        
        # 3. 파일들을 원래 data 폴더로 이동
        for file in files:
            src_path = os.path.join(part_folder, file)
            dst_path = os.path.join(target_folder, file)
            
            shutil.move(src_path, dst_path)
            total_moved += 1

        # 4. 파일 이동이 끝나서 텅 빈 분할 폴더 깔끔하게 삭제
        try:
            os.rmdir(part_folder)
            print(f"🗑️ 빈 폴더 삭제 완료: {part_folder}")
        except OSError:
            print(f"⚠️ 폴더 삭제 실패 (숨김 파일 등이 남아있을 수 있음): {part_folder}")

    print(f"\n✨ 병합 완료! 총 {total_moved}개의 영상이 무사히 '{target_folder}'로 돌아왔습니다.")

# --- 실행 ---
# 영상들이 모일 최종 원본 폴더 경로를 입력하세요.
# (이 경로를 바탕으로 _part_1, _part_2 폴더를 자동으로 찾아냅니다)
target_dir = "C:/Users/qoiop/Downloads/mlops/data" 

# 함수 실행 (part 1부터 5까지 병합)
merge_folders(target_dir, num_parts=5)