import os
import pandas as pd
from moviepy.editor import VideoFileClip

MANIFESTS = ["train_manifest.csv", "valid_manifest.csv"]
OUTPUT_CLIP_DIR = "clips"

def main():
    os.makedirs(OUTPUT_CLIP_DIR, exist_ok=True)
    
    for manifest in MANIFESTS:
        print(f"\n📄 '{manifest}' 기반 영상 추출을 시작합니다...")
        if not os.path.exists(manifest): continue
        df = pd.read_csv(manifest)
        
        for idx, row in df.iterrows():
            clip_id = row['clip_id']
            source_dir = row['source_dir'] # ⭐️ 전도는 source_videos, 정상은 normal_videos에서 똑똑하게 꺼내옵니다.
            
            original_path = os.path.join(source_dir, f"{clip_id}.mp4")
            output_path = os.path.join(OUTPUT_CLIP_DIR, f"{clip_id}.mp4")

            if os.path.exists(output_path):
                continue
            if not os.path.exists(original_path):
                print(f"⚠️ 원본 파일 없음: {original_path}")
                continue

            try:
                video = VideoFileClip(original_path)
                duration = video.duration
                
                if row['true_label'] == 1:
                    start_sec = float(row['start_sec'])
                    end_sec = float(row['end_sec'])
                    clip_start, clip_end = max(0, start_sec - 5), min(duration, end_sec + 5)
                else:
                    clip_start, clip_end = 0, min(duration, 10.0) # 5초짜리 영상이면 알아서 5초까지만 자릅니다!
                    
                subclip = video.subclip(clip_start, clip_end)
                
                # ⭐️ 마법의 호환성 주문 추가!
                subclip.write_videofile(
                    output_path, 
                    fps=3, 
                    codec="libx264", 
                    audio=False, 
                    logger=None,
                    ffmpeg_params=["-pix_fmt", "yuv420p"]
                )
                video.close()
                subclip.close()
                print(f"✂️ 추출 완료: {clip_id}")
            except Exception as e:
                pass
                
    print("\n✨ 모든 학습용 클립(전도+운동) 준비가 완료되었습니다!")

if __name__ == "__main__":
    main()