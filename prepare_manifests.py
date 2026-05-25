import os
import glob
import random
import pandas as pd
import xml.etree.ElementTree as ET

# ⚙️ 설정값
XML_DIR = "labels"                  # 기존 전도 데이터 정답지 폴더
FALL_VIDEO_DIR = "source_videos"    # 기존 전도 원본 영상 폴더
NORMAL_VIDEO_DIR = "normal_videos"  # ⭐️ 새로 만든 정상 원본 영상 폴더
TRAIN_CSV = "train_manifest.csv"
VALID_CSV = "valid_manifest.csv"
SPLIT_RATIO = 0.8
FPS = 3.0

def parse_fall_xml(xml_path):
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
        return {
            "clip_id": video_name,
            "source_dir": FALL_VIDEO_DIR, # 전도 영상은 여기서 찾아라!
            "video_path": f"clips/{video_name}.mp4",
            "true_label": 1,
            "behavior_type": "전도",
            "start_sec": round(start_frame / FPS, 2),
            "end_sec": round(end_frame / FPS, 2),
        }
    return None

def parse_normal_video(mp4_path):
    # XML 없이 파일명만 보고 '정상(0)' 라벨을 쾅 찍어줍니다!
    clip_id = os.path.splitext(os.path.basename(mp4_path))[0]
    return {
        "clip_id": clip_id,
        "source_dir": NORMAL_VIDEO_DIR, # 정상 영상은 여기서 찾아라!
        "video_path": f"clips/{clip_id}.mp4",
        "true_label": 0,
        "behavior_type": "운동(정상)",
        "start_sec": 0,    # 0초부터
        "end_sec": 10.0,   # 최대 10초까지 알아서 자르기
    }

def main():
    print("🔍 1. 기존 전도(Fall) 데이터를 스캔합니다...")
    fall_data = []
    for xml_file in glob.glob(os.path.join(XML_DIR, "*.xml")):
        parsed = parse_fall_xml(xml_file)
        if parsed: fall_data.append(parsed)
            
    print("🔍 2. 새로운 운동(Normal) 데이터를 스캔합니다...")
    normal_data = []
    for mp4_file in glob.glob(os.path.join(NORMAL_VIDEO_DIR, "*.mp4")):
        normal_data.append(parse_normal_video(mp4_file))
        
    # 두 데이터 합치기! (약 645 + 650 = 1295개)
    total_data = fall_data + normal_data
    
    # 🎲 골고루 섞기 (전도랑 정상이 뭉쳐있지 않게 쉐킷쉐킷!)
    random.seed(42)
    random.shuffle(total_data)
    
    # 80:20 분할
    split_idx = int(len(total_data) * SPLIT_RATIO)
    train_data = total_data[:split_idx]
    valid_data = total_data[split_idx:]

    pd.DataFrame(train_data).to_csv(TRAIN_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(valid_data).to_csv(VALID_CSV, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 완벽한 황금비율 정답지 생성 완료!")
    print(f"   -> 📚 학습용(Train): {len(train_data)}개")
    print(f"   -> 📝 검증용(Valid): {len(valid_data)}개")

if __name__ == "__main__":
    main()