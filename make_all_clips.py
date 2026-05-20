import os
import pandas as pd
from moviepy.editor import VideoFileClip

# ⚙️ 설정값
MANIFESTS = ["train_manifest.csv", "valid_manifest.csv"]
SOURCE_VIDEO_DIR = "source_videos"
OUTPUT_CLIP_DIR = "clips"

def main():
    os.makedirs(OUTPUT_CLIP_DIR, exist_ok=True)
    
    for manifest in MANIFESTS:
        print(f"\n📄 '{manifest}'에 등록된 영상 자르기를 시작합니다...")
        if not os.path.exists(manifest):
            continue

        df = pd.read_csv(manifest)
        
        for idx, row in df.iterrows():
            clip_id = row['clip_id']
            original_path = os.path.join(SOURCE_VIDEO_DIR, f"{clip_id}.mp4")
            output_path = os.path.join(OUTPUT_CLIP_DIR, f"{clip_id}.mp4")

            # 이미 잘린 영상은 초고속 건너뛰기!
            if os.path.exists(output_path):
                print(f"⏩ 통과 (이미 존재함): {clip_id}")
                continue
                
            if not os.path.exists(original_path):
                print(f"⚠️ 원본 없음 (건너뜀): {clip_id}")
                continue

            try:
                video = VideoFileClip(original_path)
                duration = video.duration
                
                if row['true_label'] == 1:
                    start_sec = float(row['start_sec'])
                    end_sec = float(row['end_sec'])
                    clip_start, clip_end = max(0, start_sec - 5), min(duration, end_sec + 5)
                else:
                    clip_start, clip_end = 0, min(duration, 10.0)
                    
                subclip = video.subclip(clip_start, clip_end)
                subclip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
                video.close()
                subclip.close()
                print(f"✂️ 생성 완료: {clip_id}")
            except Exception as e:
                pass
                
    print("\n✨ 모든 영상 클립 준비가 완료되었습니다!")

if __name__ == "__main__":
    main()