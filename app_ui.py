import os
import streamlit as st
import requests
import pandas as pd
from supabase import create_client, Client

# 1. 페이지 기본 설정
st.set_page_config(page_title="AI 전도 감지 시스템", page_icon="🚨", layout="wide")

# 2. 사이드바 (메뉴 및 설정)
st.sidebar.title("⚙️ 시스템 설정")
st.sidebar.markdown("백엔드 서버 주소를 확인하세요.")
api_url = "https://ichanho-fall-detection-api.hf.space"

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
# 메뉴 2: 통합 모니터링 대시보드 (민준님 전용 DB 연동)
# ==========================================
elif menu == "📊 통합 모니터링 대시보드":
    st.title("📊 모니터링 & 라벨링 대시보드")
    st.markdown("저장된 AI 분석 로그를 확인하고, **오탐/미탐 여부(admin_confirm_label)**를 직접 DB에 업데이트할 수 있습니다.")
    
    # Supabase 연결 설정 (Render 환경변수에서 가져옴)
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("⚠️ Supabase 연결 정보가 없습니다. Render 환경변수에 SUPABASE_URL과 SUPABASE_KEY를 등록해주세요.")
    else:
        # DB 연결
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # 1. DB에서 데이터 불러오기 (id 역순 = 최신순)
        response = supabase.table('logs').select("*").order('id', desc=True).execute()
        df = pd.DataFrame(response.data)

        if df.empty:
            st.warning("아직 저장된 로그가 없습니다. 영상을 먼저 분석해보세요!")
        else:
            # 2. 상단 요약 지표 (DB 구조에 맞게 변경)
            st.subheader("📌 시스템 가동 요약")
            c1, c2, c3 = st.columns(3)
            c1.metric("총 분석 횟수", f"{len(df)} 건")
            fall_count = len(df[df['prediction'] == '전도 상황 감지'])
            c2.metric("전도 감지 횟수", f"{fall_count} 건")
            c3.metric("평균 판정 확률(Probability)", f"{df['probability'].mean():.4f}")
            
            st.markdown("---")
            
            # 3. 데이터 에디터 (엑셀 형태의 수정 가능한 표)
            st.subheader("📝 라벨링 대시보드 (수정 가능)")
            st.info("💡 표의 `admin_confirm_label`과 `error_type` 빈칸을 **더블클릭**해서 내용을 입력한 뒤, 아래 **저장 버튼**을 누르세요.")
            
            # 컬럼 순서를 보기 좋게 정렬
            cols = ["id", "created_at", "filename", "prediction", "probability", "admin_confirm_label", "error_type"]
            df = df[cols]
            
            # st.data_editor로 수정 가능한 표 생성
            edited_df = st.data_editor(
                df,
                # AI가 기록한 원본 데이터는 수정 못하게 잠금 처리!
                disabled=["id", "created_at", "filename", "prediction", "probability"],
                use_container_width=True,
                key="log_editor" # 변경 사항 추적을 위한 고유 키
            )
            
            # 4. 수정한 데이터를 실제 DB에 덮어쓰는 버튼
            if st.button("💾 변경사항 DB에 업데이트", type="primary"):
                # 사용자가 수정한 내역만 가져오기
                changes = st.session_state["log_editor"]["edited_rows"]
                
                if changes:
                    with st.spinner("DB에 저장하는 중..."):
                        for row_idx, updates in changes.items():
                            # 수정한 줄의 고유 id 찾기
                            row_id = df.iloc[row_idx]["id"]
                            # 해당 id를 가진 DB 행에 업데이트 날리기
                            supabase.table('logs').update(updates).eq('id', row_id).execute()
                            
                    st.success("✅ DB 업데이트 완료! 호준이가 재학습할 데이터가 안전하게 저장되었습니다.")
                    time.sleep(1) # 1초 대기 후
                    st.rerun() # 화면 새로고침해서 적용된 데이터 보여주기
                else:
                    st.warning("수정된 항목이 없습니다. 표의 빈칸을 더블클릭해서 먼저 입력해주세요.")