import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# ⚙️ 폴더 설정값
TRAIN_XML_DIR = "labels"              # 기존 645개 정답지
TRAIN_VIDEO_DIR = "source_videos"     # 기존 645개 영상

VALID_XML_DIR = "valid_labels"        # ⭐️ 새로운 81개 정답지
VALID_VIDEO_DIR = "valid_videos"      # ⭐️ 새로운 81개 영상

FPS = 3.0

def parse_xml_to_list(xml_dir, video_dir):
    data_list = []
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            name_node = root.find('.//meta/task/name')
            video_name = name_node.text if name_node is not None else os.path.splitext(os.path.basename(xml_path))[0]
            
            start_frame, end_frame = None, None
            for track in root.findall('.//track'):
                label = track.get('label')
                if label == 'fall_start':
                    start_frame = int(track.find('box').get('frame'))
                elif label == 'fall_end':
                    end_frame = int(track.find('box').get('frame'))
                    
            if start_frame is not None and end_frame is not None:
                data_list.append({
                    "clip_id": video_name,
                    "source_dir": video_dir,  # ⭐️ 영상을 어디서 가져올지 명시!
                    "video_path": f"clips/{video_name}.mp4",
                    "true_label": 1,
                    "behavior_type": "전도",
                    "start_sec": round(start_frame / FPS, 2),
                    "end_sec": round(end_frame / FPS, 2),
                })
        except Exception as e:
            pass
            
    return data_list

def main():
    print("🔍 1. Train 데이터 (기존 645개) 스캔 중...")
    train_data = parse_xml_to_list(TRAIN_XML_DIR, TRAIN_VIDEO_DIR)
    pd.DataFrame(train_data).to_csv("train_manifest.csv", index=False, encoding='utf-8-sig')

    print("🔍 2. Validation 데이터 (새로운 81개) 스캔 중...")
    valid_data = parse_xml_to_list(VALID_XML_DIR, VALID_VIDEO_DIR)
    pd.DataFrame(valid_data).to_csv("valid_manifest.csv", index=False, encoding='utf-8-sig')

    print(f"\n✅ 완벽하게 분리된 정답지 생성 완료!")
    print(f"   -> 📚 학습용(Train): {len(train_data)}개 저장됨")
    print(f"   -> 📝 검증용(Valid): {len(valid_data)}개 저장됨")

if __name__ == "__main__":
    main()