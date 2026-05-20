import os
import pandas as pd
from moviepy.editor import VideoFileClip

# ⚙️ 설정값
MANIFEST_PATH = "monitoring_eval_manifest.csv"
SOURCE_VIDEO_DIR = "source_videos"  # ⭐️ AIHub에서 다운받은 '원본 영상(.mp4)'들이 들어있는 폴더
OUTPUT_CLIP_DIR = "clips"           # 잘린 영상들이 저장될 폴더 (CSV의 video_path와 일치)

def main():
    print(f"🔍 '{MANIFEST_PATH}' 정답지를 읽어옵니다...")
    if not os.path.exists(MANIFEST_PATH):
        print("❌ 정답지 파일이 없습니다. parse_aihub_xml.py를 먼저 실행해주세요.")
        return

    # 출력 폴더 생성
    os.makedirs(OUTPUT_CLIP_DIR, exist_ok=True)
    
    # CSV 읽기
    df = pd.read_csv(MANIFEST_PATH)
    total_videos = len(df)
    success_count = 0

    print(f"🎬 총 {total_videos}개의 클립 생성을 시작합니다!\n")

    for idx, row in df.iterrows():
        clip_id = row['clip_id']
        true_label = row['true_label']
        
        # 원본 영상 경로 조립 (확장자가 mp4라고 가정)
        original_video_path = os.path.join(SOURCE_VIDEO_DIR, f"{clip_id}.mp4")
        output_path = os.path.join(OUTPUT_CLIP_DIR, f"{clip_id}.mp4")

        # 이미 잘린 영상이 있다면 패스 (중단 후 재시작 시 유용)
        if os.path.exists(output_path):
            print(f"⏩ [{idx+1}/{total_videos}] 이미 존재함: {output_path}")
            success_count += 1
            continue

        if not os.path.exists(original_video_path):
            print(f"⚠️ [{idx+1}/{total_videos}] 원본 영상 없음 (건너뜀): {original_video_path}")
            continue

        try:
            # 영상 불러오기
            video = VideoFileClip(original_video_path)
            duration = video.duration
            
            if true_label == 1:
                # 🎯 [전도 데이터] 기획안 로직: 시작 5초 전 ~ 종료 5초 후
                start_sec = float(row['start_sec'])
                end_sec = float(row['end_sec'])
                
                clip_start = max(0, start_sec - 5)
                clip_end = min(duration, end_sec + 5)
                print(f"✂️ [{idx+1}/{total_videos}] 전도 클립 생성 중: {clip_id} ({clip_start:.1f}초 ~ {clip_end:.1f}초)")
            else:
                # 🎯 [정상 데이터] 기획안 로직: 첫 10초만 자르기
                clip_start = 0
                clip_end = min(duration, 10.0)
                print(f"✂️ [{idx+1}/{total_videos}] 정상 클립 생성 중: {clip_id} (0초 ~ {clip_end:.1f}초)")
                
            # 영상 자르기
            subclip = video.subclip(clip_start, clip_end)
            
            # 잘라낸 영상 저장 (오디오 제거, 로그 끄기, x264 코덱 사용)
            subclip.write_videofile(
                output_path, 
                codec="libx264", 
                audio=False, 
                logger=None # 터미널이 지저분해지는 것을 방지
            )
            
            # 메모리 해제
            video.close()
            subclip.close()
            success_count += 1
            
        except Exception as e:
            print(f"❌ [{idx+1}/{total_videos}] 에러 발생 ({clip_id}): {e}")

    print(f"\n✨ 작업 완료! 총 {success_count}/{total_videos}개의 영상이 '{OUTPUT_CLIP_DIR}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()