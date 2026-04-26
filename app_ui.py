import streamlit as st
import requests
import pandas as pd
import time

# 1. 페이지 기본 설정
st.set_page_config(page_title="AI 전도 감지 시스템", page_icon="🚨", layout="wide")

# 2. 사이드바 (메뉴 및 설정)
st.sidebar.title("⚙️ 시스템 설정")
st.sidebar.markdown("백엔드 서버 주소를 확인하세요.")
api_url = "https://mlops-backend-gl93.onrender.com"

st.sidebar.markdown("---")
menu = st.sidebar.radio("메뉴 이동", ["📹 실시간 영상 분석", "📊 통합 모니터링 대시보드"])

# ==========================================
# 메뉴 1: 실시간 영상 분석 (추론 테스트)
# ==========================================
if menu == "📹 실시간 영상 분석":
    st.title("🏃‍♂️ 실시간 전도 감지 테스트")
    st.markdown("CCTV 영상을 업로드하면 AI가 전도 여부를 즉시 분석합니다.")
    
    uploaded_file = st.file_uploader("동영상 파일 업로드", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("🚨 AI 분석 시작", type="primary"):
            with st.spinner("AI 엔진 가동 중..."):
                try:
                    files = {"video": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{api_url}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("status") == "success":
                            st.subheader("💡 분석 결과")
                            
                            # 백엔드에서 보낸 data 묶음 활용
                            data = result["data"]
                            is_fall = data["predicted_class"] == 1
                            
                            if is_fall:
                                st.error(f"⚠️ {result['message']} (신뢰도: {data['confidence']*100:.2f}%)")
                            else:
                                st.success(f"✅ {result['message']} (신뢰도: {data['confidence']*100:.2f}%)")
                            
                            # 민준이가 궁금해할 핵심 지표 3개 상단 표시
                            col1, col2, col3 = st.columns(3)
                            col1.metric(label="최종 판정", value="전도 감지" if is_fall else "정상")
                            col2.metric(label="신뢰도", value=f"{data['confidence']:.4f}")
                            col3.metric(label="추론 속도", value=f"{data['inference_time_ms']} ms")
                        else:
                            st.error(f"서버 오류: {result.get('message')}")
                    else:
                        st.error("백엔드 서버 응답 실패")
                except Exception as e:
                    st.error(f"연결 에러: {e}")

# ==========================================
# 메뉴 2: 통합 모니터링 대시보드 (민준님 전용)
# ==========================================
elif menu == "📊 통합 모니터링 대시보드":
    st.title("📊 AI 추론 로그 모니터링")
    st.markdown("민준님이 요청하신 8가지 핵심 데이터를 실시간으로 모니터링합니다.")
    
    if st.button("🔄 최신 데이터 불러오기"):
        with st.spinner("서버에서 로그 데이터를 가져오는 중..."):
            try:
                # 💡 중요: 이제 CSV가 아니라 JSON API(/logs)에서 데이터를 가져옵니다.
                response = requests.get(f"{api_url}/logs")
                
                if response.status_code == 200:
                    log_data = response.json()
                    
                    if not log_data:
                        st.warning("아직 저장된 로그가 없습니다. 영상을 먼저 분석해보세요!")
                    else:
                        # JSON을 표(DataFrame)로 변환
                        df = pd.DataFrame(log_data)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values(by="timestamp", ascending=False)
                        
                        # 1. 상단 요약 지표
                        st.subheader("📌 시스템 가동 요약")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("총 분석 횟수", f"{len(df)} 건")
                        c2.metric("전도 감지 횟수", f"{len(df[df['predicted_class'] == 1])} 건")
                        c3.metric("평균 추론 시간", f"{df['inference_time_ms'].mean():.1f} ms")
                        
                        st.markdown("---")
                        
                        # 2. 신뢰도 변화 그래프
                        st.subheader("📈 탐지 신뢰도(Confidence) 추이")
                        st.line_chart(df.set_index('timestamp')['confidence'])
                        
                        # 3. 민준이가 요청한 8가지 데이터 전체 표
                        st.subheader("상세 추론 로그")
                        
                        # 민준님이 보기 편하게 컬럼 순서 정렬
                        cols = [
                            "timestamp", "predicted_class", "confidence", "inference_time_ms",
                            "input_frames", "model_version", "threshold_applied", 
                            "camera_id", "video_snippet_id", "admin_confirm_label", "error_type"
                        ]
                        # 실제 존재하는 컬럼만 필터링하여 출력
                        existing_cols = [c for c in cols if c in df.columns]
                        st.dataframe(df[existing_cols], use_container_width=True)
                        
                else:
                    st.error("로그 데이터를 가져오지 못했습니다.")
            except Exception as e:
                st.error(f"데이터 로드 중 에러 발생: {e}")