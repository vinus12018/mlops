import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import GroupShuffleSplit

# 폴더 설정값
XML_DIR = "labels"                  # 전도 645개 XML
VIDEO_DIR = "source_videos"         # 전도 645개 영상

NORMAL_XML_DIR = "normal_labels"    # 정상 매장 XML 최상위 폴더
NORMAL_VIDEO_DIR = "normal_videos"  # 정상 매장 영상 최상위 폴더

SPLIT_RATIO = 0.8
FPS = 3.0

def extract_group_id(filename):
    """파일명에서 카메라 코드를 제외한 사건(Event) 단위 고유 ID를 추출합니다."""
    parts = filename.split('_')
    if len(parts) >= 7:
        return "_".join(parts[:7])
    return filename

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
        
        if video_name not in video_dir_map:
            return None
            
        frames = []
        for box in root.findall('.//box'):
            frames.append(int(box.get('frame')))
            
        if frames:
            start_frame = min(frames)
            end_frame = max(frames)
            
            # 폴더명(행동 타입)을 명확하게 추출하기 위해 수정
            behavior_type = os.path.basename(video_dir_map[video_name])

            return {
                "clip_id": video_name,
                "source_dir": video_dir_map[video_name],
                "video_path": f"clips/{video_name}.mp4",
                "true_label": 0,
                "behavior_type": behavior_type, # 기존 "정상(매장)" 대신 실제 폴더명 사용
                "start_sec": round(start_frame / FPS, 2),
                "end_sec": round(end_frame / FPS, 2),
            }
    except Exception: pass
    return None

def main():
    # 1. 전도 데이터 파싱
    print("[1/3] 원본 전도 데이터 스캔 중...")
    fall_data = []
    for xml_file in glob.glob(os.path.join(XML_DIR, "*.xml")):
        parsed = parse_fall_xml(xml_file)
        if parsed: fall_data.append(parsed)

    # 2. 정상 데이터 파싱
    print("[2/3] 하위 폴더 탐색 및 정상 데이터 매칭 중...")
    normal_video_map = {}
    for mp4_path in glob.glob(os.path.join(NORMAL_VIDEO_DIR, "**", "*.mp4"), recursive=True):
        clip_id = os.path.splitext(os.path.basename(mp4_path))[0]
        normal_video_map[clip_id] = os.path.dirname(mp4_path)

    normal_data = []
    for xml_file in glob.glob(os.path.join(NORMAL_XML_DIR, "**", "*.xml"), recursive=True):
        parsed = parse_normal_xml(xml_file, normal_video_map)
        if parsed: normal_data.append(parsed)

    # 3. 데이터 통합 및 데이터 누수 방지 Group Split 적용
    print("[3/3] 데이터 통합 및 사건(Event) 기반 Group Split 진행 중...")
    
    # 두 리스트를 하나의 데이터프레임으로 통합
    all_data = fall_data + normal_data
    df = pd.DataFrame(all_data)

    # 파일명에서 사건 단위의 Group ID 추출하여 컬럼 추가
    df['group_id'] = df['clip_id'].apply(extract_group_id)

    # GroupShuffleSplit 객체 생성
    gss = GroupShuffleSplit(n_splits=1, train_size=SPLIT_RATIO, random_state=42)
    
    # 생성된 group_id를 기준으로 분할 실행
    train_idx, val_idx = next(gss.split(df, groups=df['group_id']))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # 결과물 CSV 저장
    train_df.to_csv("train_manifest.csv", index=False, encoding='utf-8-sig')
    val_df.to_csv("valid_manifest.csv", index=False, encoding='utf-8-sig')

    # 통계 출력
    train_fall_count = len(train_df[train_df['true_label'] == 1])
    train_normal_count = len(train_df[train_df['true_label'] == 0])
    val_fall_count = len(val_df[val_df['true_label'] == 1])
    val_normal_count = len(val_df[val_df['true_label'] == 0])

    print("\n[완료] 데이터 누수(Data Leakage)를 방지한 데이터셋 생성이 완료되었습니다.")
    print("-" * 50)
    print(f"학습용(Train) : 총 {len(train_df)}개 (전도 {train_fall_count}개 / 정상 {train_normal_count}개)")
    print(f"검증용(Valid) : 총 {len(val_df)}개 (전도 {val_fall_count}개 / 정상 {val_normal_count}개)")
    print("-" * 50)
    print(f"도출된 고유 사건(Group) 수: Train({train_df['group_id'].nunique()}개) / Valid({val_df['group_id'].nunique()}개)")

if __name__ == "__main__":
    main()