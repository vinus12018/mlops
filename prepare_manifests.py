import os
import glob
import random
import pandas as pd
import xml.etree.ElementTree as ET

# ⚙️ 설정값
XML_DIR = "labels" 
TRAIN_CSV = "train_manifest.csv"
VALID_CSV = "valid_manifest.csv"
SPLIT_RATIO = 0.8  # 학습용 80%, 검증용 20%
FPS = 3.0          # ⭐️ 초당 3프레임(3fps) 반영

def parse_single_xml(xml_path):
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
        true_label, behavior_type = 1, "전도"
        start_sec = round(start_frame / FPS, 2)
        end_sec = round(end_frame / FPS, 2)
    else:
        true_label, behavior_type = 0, "정상보행"
        start_sec, end_sec = "", ""
        
    return {
        "clip_id": video_name,
        "video_path": f"clips/{video_name}.mp4",
        "true_label": true_label,
        "behavior_type": behavior_type,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }

def main():
    print(f"🔍 '{XML_DIR}' 폴더에서 전체 XML 데이터를 스캔합니다...")
    xml_files = glob.glob(os.path.join(XML_DIR, "*.xml"))
    xml_files.sort()
    
    if not xml_files:
        print("❌ XML 파일을 찾을 수 없습니다.")
        return

    data_list = []
    for xml_file in xml_files:
        try:
            data_list.append(parse_single_xml(xml_file))
        except Exception as e:
            pass

    # 🎲 80:20 무작위 분할 (seed 고정으로 항상 같은 비율 유지)
    random.seed(42)
    random.shuffle(data_list)
    
    split_index = int(len(data_list) * SPLIT_RATIO)
    train_data = data_list[:split_index]
    valid_data = data_list[split_index:]

    # CSV 저장
    pd.DataFrame(train_data).to_csv(TRAIN_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(valid_data).to_csv(VALID_CSV, index=False, encoding='utf-8-sig')
    
    print(f"✅ 데이터 분할 완료! 총 {len(data_list)}개 중")
    print(f"   -> 📚 학습용(Train): {len(train_data)}개 저장 ({TRAIN_CSV})")
    print(f"   -> 📝 검증용(Valid): {len(valid_data)}개 저장 ({VALID_CSV})")

if __name__ == "__main__":
    main()