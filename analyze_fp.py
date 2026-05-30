import os
import glob
import pandas as pd

# ⚙️ 설정값
FP_CSV_PATH = "outputs/fp_cases.csv"
NORMAL_VIDEOS_DIR = "normal_videos"

def main():
    if not os.path.exists(FP_CSV_PATH):
        print(f"🚨 '{FP_CSV_PATH}' 파일이 없습니다. 평가를 먼저 진행해 주세요!")
        return

    print("🔍 1. 원본 폴더를 스캔하여 영상들의 진짜 고향(하위 폴더)을 지도에 그립니다...")
    
    # { "영상 이름": "하위 폴더 이름" } 형태의 지도(Dictionary) 생성
    clip_to_subfolder = {}
    
    # normal_videos 안의 모든 하위 폴더(*_videos)에 있는 mp4 파일을 찾습니다.
    search_pattern = os.path.join(NORMAL_VIDEOS_DIR, "*", "*.mp4")
    for mp4_path in glob.glob(search_pattern):
        # 폴더 이름 추출 (예: 'purchase_videos')
        subfolder_name = os.path.basename(os.path.dirname(mp4_path))
        # 파일 이름 추출 (예: 'C_1_1_81_BU...')
        clip_name = os.path.splitext(os.path.basename(mp4_path))[0]
        
        clip_to_subfolder[clip_name] = subfolder_name

    print("📝 2. fp_cases.csv 파일에 진짜 행동(폴더명)을 매칭합니다...")
    df = pd.read_csv(FP_CSV_PATH)
    
    # 'clip_id'를 열쇠로 삼아, 아까 만든 지도에서 폴더명을 찾아 'specific_behavior' 컬럼에 적어줍니다.
    df['specific_behavior'] = df['clip_id'].map(clip_to_subfolder)
    
    # 결과를 한눈에 보기 좋게 터미널에 출력
    print("\n" + "=" * 50)
    print("[오탐지(FP) 원인 행동 분석 결과]")
    print("=" * 50)
    
    # 폴더별로 몇 건의 FP가 발생했는지 개수를 세어줍니다.
    behavior_counts = df['specific_behavior'].value_counts()
    for behavior, count in behavior_counts.items():
        print(f" {behavior:<20} : {count}건의 오탐지 발생")
    print("=" * 50)

    # 매칭된 결과를 포함하여 새로운 엑셀 파일로 저장
    output_path = "outputs/fp_cases_analyzed.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 상세 행동이 추가된 새 엑셀 파일이 '{output_path}'에 저장되었습니다!")

if __name__ == "__main__":
    main()