import os
import glob
import random
import pandas as pd
import xml.etree.ElementTree as ET

# ⚙️ 폴더 설정값
XML_DIR = "labels"                  # 전도 645개 XML
VIDEO_DIR = "source_videos"         # 전도 645개 영상

NORMAL_XML_DIR = "normal_labels"    # ⭐️ 정상 매장 XML 최상위 폴더
NORMAL_VIDEO_DIR = "normal_videos"  # ⭐️ 정상 매장 영상 최상위 폴더

SPLIT_RATIO = 0.8
TARGET_NORMAL_COUNT = 645           # 1:1 비율을 위해 645개만 뽑기!
FPS = 3.0

def parse_fall_xml(xml_path):
    """전도 데이터의 시작/종료 시점을 찾습니다."""
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
            return {
                "clip_id": video_name,
                "source_dir": VIDEO_DIR,
                "video_path": f"clips/{video_name}.mp4",
                "true_label": 1,
                "behavior_type": "전도",
                "start_sec": round(start_frame / FPS, 2),
                "end_sec": round(end_frame / FPS, 2),
            }
    except Exception: pass
    return None

def parse_normal_xml(xml_path, video_dir_map):
    """정상 데이터(매장)의 모든 행동 프레임 중 최소/최대 구간을 찾습니다."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        name_node = root.find('.//meta/task/name')
        video_name = name_node.text if name_node is not None else os.path.splitext(os.path.basename(xml_path))[0]
        
        # 영상 파일이 실제로 존재하는지 지도(map)에서 확인
        if video_name not in video_dir_map:
            return None
            
        frames = []
        for box in root.findall('.//box'):
            frames.append(int(box.get('frame')))
            
        if frames:
            start_frame = min(frames)
            end_frame = max(frames)
            
            return {
                "clip_id": video_name,
                "source_dir": video_dir_map[video_name], # 영상이 숨어있는 진짜 폴더 위치
                "video_path": f"clips/{video_name}.mp4",
                "true_label": 0,
                "behavior_type": "정상(매장)",
                "start_sec": round(start_frame / FPS, 2),
                "end_sec": round(end_frame / FPS, 2),
            }
    except Exception: pass
    return None

def main():
    # ---------------------------------------------------------
    # 1. 전도 데이터 파싱 (645개)
    # ---------------------------------------------------------
    print("🔍 1. 원본 전도 데이터(1번) 스캔 중...")
    fall_data = []
    for xml_file in glob.glob(os.path.join(XML_DIR, "*.xml")):
        parsed = parse_fall_xml(xml_file)
        if parsed: fall_data.append(parsed)
            
    random.seed(42)
    random.shuffle(fall_data)
    
    split_fall_idx = int(len(fall_data) * SPLIT_RATIO)
    train_fall = fall_data[:split_fall_idx]
    valid_fall = fall_data[split_fall_idx:]

    # ---------------------------------------------------------
    # 2. 정상 데이터 파싱 (XML 기반 정확한 구간 추출)
    # ---------------------------------------------------------
    print("🔍 2. 하위 폴더들을 뒤져 정상 매장 데이터(0번)를 매칭합니다...")
    
    # 영상들이 어디있는지 미리 지도를 그려둡니다 (초고속 탐색)
    normal_video_map = {}
    for mp4_path in glob.glob(os.path.join(NORMAL_VIDEO_DIR, "**", "*.mp4"), recursive=True):
        clip_id = os.path.splitext(os.path.basename(mp4_path))[0]
        normal_video_map[clip_id] = os.path.dirname(mp4_path)

    normal_data = []
    for xml_file in glob.glob(os.path.join(NORMAL_XML_DIR, "**", "*.xml"), recursive=True):
        parsed = parse_normal_xml(xml_file, normal_video_map)
        if parsed: normal_data.append(parsed)
        
        
    split_normal_idx = int(len(normal_data) * SPLIT_RATIO)
    train_normal = normal_data[:split_normal_idx]
    valid_normal = normal_data[split_normal_idx:]

    # ---------------------------------------------------------
    # 3. 데이터 통합 및 정답지 발행
    # ---------------------------------------------------------
    train_total = train_fall + train_normal
    valid_total = valid_fall + valid_normal
    
    random.shuffle(train_total)
    random.shuffle(valid_total)

    pd.DataFrame(train_total).to_csv("train_manifest.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame(valid_total).to_csv("valid_manifest.csv", index=False, encoding='utf-8-sig')

    print(f"\n✅ 고품질 XML 추출! 완벽한 1:1 황금 비율 정답지 생성 완료!")
    print(f"   -> 📚 학습용(Train): 총 {len(train_total)}개 (전도 {len(train_fall)} + 정상 {len(train_normal)})")
    print(f"   -> 📝 검증용(Valid): 총 {len(valid_total)}개 (전도 {len(valid_fall)} + 정상 {len(valid_normal)})")

if __name__ == "__main__":
    main()