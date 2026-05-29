import os
import glob
import random
import pandas as pd
import xml.etree.ElementTree as ET

# ⚙️ 폴더 설정값
TRAIN_XML_DIR = "labels"              # 전도 Train (645개)
TRAIN_VIDEO_DIR = "source_videos"

VALID_XML_DIR = "valid_labels"        # 전도 Valid (81개)
VALID_VIDEO_DIR = "valid_videos"

NORMAL_VIDEO_DIR = "normal_videos"    # 운동/정상 (650개)

FPS = 3.0

def parse_fall_xml(xml_dir, video_dir):
    """전도(1) 데이터를 파싱합니다."""
    data_list = []
    for xml_path in glob.glob(os.path.join(xml_dir, "*.xml")):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            name_node = root.find('.//meta/task/name')
            video_name = name_node.text if name_node is not None else os.path.splitext(os.path.basename(xml_path))[0]
            
            start_frame, end_frame = None, None
            for track in root.findall('.//track'):
                label = track.get('label')
                if label == 'fall_start': start_frame = int(track.find('box').get('frame'))
                elif label == 'fall_end': end_frame = int(track.find('box').get('frame'))
                    
            if start_frame is not None and end_frame is not None:
                data_list.append({
                    "clip_id": video_name,
                    "source_dir": video_dir,
                    "video_path": f"clips/{video_name}.mp4",
                    "true_label": 1,
                    "behavior_type": "전도",
                    "start_sec": round(start_frame / FPS, 2),
                    "end_sec": round(end_frame / FPS, 2),
                })
        except Exception: pass
    return data_list

def parse_normal_videos(video_dir):
    """운동/정상(0) 데이터를 파싱합니다."""
    data_list = []
    for mp4_path in glob.glob(os.path.join(video_dir, "*.mp4")):
        clip_id = os.path.splitext(os.path.basename(mp4_path))[0]
        data_list.append({
            "clip_id": clip_id,
            "source_dir": video_dir,
            "video_path": f"clips/{clip_id}.mp4",
            "true_label": 0,
            "behavior_type": "운동(정상)",
            "start_sec": 0,
            "end_sec": 10.0,
        })
    return data_list

def main():
    print("🔍 1. 전도(Fall) 데이터 스캔 중...")
    train_fall = parse_fall_xml(TRAIN_XML_DIR, TRAIN_VIDEO_DIR)
    valid_fall = parse_fall_xml(VALID_XML_DIR, VALID_VIDEO_DIR)
    
    print("🔍 2. 운동(Normal) 데이터 스캔 및 분할 중...")
    normal_data = parse_normal_videos(NORMAL_VIDEO_DIR)
    
    # 🎲 정상 데이터 무작위 섞기 후 분할 (Train 500개, 나머지 Valid)
    random.seed(42)
    random.shuffle(normal_data)
    split_idx = min(500, len(normal_data)) 
    
    train_normal = normal_data[:split_idx]
    valid_normal = normal_data[split_idx:]
    
    # 💥 최종 통합 및 섞기 (1번과 0번이 뭉쳐있지 않게 쉐킷쉐킷)
    final_train = train_fall + train_normal
    final_valid = valid_fall + valid_normal
    
    random.shuffle(final_train)
    random.shuffle(final_valid)

    pd.DataFrame(final_train).to_csv("train_manifest.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame(final_valid).to_csv("valid_manifest.csv", index=False, encoding='utf-8-sig')

    print(f"\n✨ 마스터 정답지(CSV) 생성 완료!")
    print(f"   -> 📚 학습용(Train): 총 {len(final_train)}개 (전도 {len(train_fall)} + 정상 {len(train_normal)})")
    print(f"   -> 📝 검증용(Valid): 총 {len(final_valid)}개 (전도 {len(valid_fall)} + 정상 {len(valid_normal)})")

if __name__ == "__main__":
    main()