import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# ⚙️ [수정] 실제 데이터 규격인 3fps로 변경합니다!
XML_DIR = "labels" 
OUTPUT_CSV = "monitoring_eval_manifest.csv"
FPS = 3.0 # ⭐️ 초당 3프레임(3fps) 환경 반영

def parse_single_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    name_node = root.find('.//meta/task/name')
    video_name = name_node.text if name_node is not None else os.path.splitext(os.path.basename(xml_path))[0]
    
    start_frame, end_frame = None, None
    
    for track in root.findall('.//track'):
        label = track.get('label')
        if label == 'fall_start':
            box = track.find('box')
            if box is not None:
                start_frame = int(box.get('frame'))
        elif label == 'fall_end':
            box = track.find('box')
            if box is not None:
                end_frame = int(box.get('frame'))
                
    if start_frame is not None and end_frame is not None:
        true_label = 1
        behavior_type = "전도"
        # ⭐️ 3fps 기준으로 정확한 초(Second) 계산
        start_sec = round(start_frame / FPS, 2)
        end_sec = round(end_frame / FPS, 2)
        memo = "fall_start/fall_end 존재 (3fps)"
    else:
        true_label = 0
        behavior_type = "정상보행"
        start_frame, end_frame = "", ""
        start_sec, end_sec = "", ""
        memo = "전도 아님 (정상 데이터)"
        
    return {
        "clip_id": video_name,
        "video_path": f"clips/{video_name}.mp4",
        "label_file": xml_path.replace("\\", "/"),
        "true_label": true_label,
        "behavior_type": behavior_type,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "source_dataset": "AIHub_CCTV_Fall",
        "split": "test",
        "memo": memo
    }

def main():
    print(f"🔍 '{XML_DIR}' 폴더에서 XML 파일들을 스캔합니다...")
    xml_files = glob.glob(os.path.join(XML_DIR, "*.xml"))
    xml_files.sort() # 파일명 순 정렬 추가
    
    if not xml_files:
        print("❌ XML 파일을 찾을 수 없습니다.")
        return

    data_list = []
    for xml_file in xml_files:
        try:
            parsed_data = parse_single_xml(xml_file)
            data_list.append(parsed_data)
        except Exception as e:
            print(f"⚠️ {xml_file} 파싱 중 에러 발생: {e}")

    df = pd.DataFrame(data_list)
    columns_order = [
        "clip_id", "video_path", "label_file", "true_label", 
        "behavior_type", "start_frame", "end_frame", 
        "start_sec", "end_sec", "source_dataset", "split", "memo"
    ]
    df = df[columns_order]
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 총 {len(df)}개의 데이터 파싱 완료! 3fps 반영된 '{OUTPUT_CSV}'가 생성되었습니다.")

if __name__ == "__main__":
    main()