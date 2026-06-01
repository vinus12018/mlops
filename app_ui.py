import streamlit as st
import requests

# ── 페이지 기본 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="SafeWatch AI | 전도 감지 시스템",
    page_icon="🛡️",
    layout="centered",
)

API_URL = "https://ichanho-fall-detection-api.hf.space"

# ── 커스텀 CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Noto+Sans+KR:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #c8d6e5;
}

/* 사이드바 + 상단 메뉴 제거 */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* 헤더 */
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #e0eaff;
    letter-spacing: 4px;
    text-align: center;
    text-transform: uppercase;
    margin-bottom: 0;
    padding-top: 1rem;
}
.hero-sub {
    font-size: 0.72rem;
    color: #4a6a8a;
    text-align: center;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
    margin-bottom: 0;
}
.badge-online {
    display: inline-block;
    background: #0a2010;
    color: #2ecc71;
    border: 1px solid #1a5c30;
    border-radius: 3px;
    font-size: 0.65rem;
    letter-spacing: 1px;
    padding: 2px 9px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
}

/* 구분선 */
.divider {
    border: none;
    border-top: 1px solid #1e3050;
    margin: 1.4rem 0;
}

/* 3단 안내 카드 */
.guide-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin: 1rem 0;
}
.guide-card {
    background: #0d1526;
    border: 1px solid #142038;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    font-size: 0.78rem;
    color: #5a7a9a;
    line-height: 1.6;
}
.guide-card strong {
    display: block;
    color: #8aadcc;
    font-size: 0.8rem;
    margin-bottom: 4px;
}

/* 파일 업로더 */
[data-testid="stFileUploader"] {
    background: #0d1526;
    border: 1px dashed #1e3a5f;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2a6eb5;
}

/* 버튼 */
.stButton > button {
    background: linear-gradient(135deg, #1a3a6b 0%, #0d2244 100%);
    color: #7ec8f5;
    border: 1px solid #2a5a9f;
    border-radius: 4px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.55rem 2rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e4a8a 0%, #112d5e 100%);
    border-color: #4a8acf;
    color: #b5dff7;
}
.stButton > button:active {
    transform: scale(0.98);
}

/* 결과 박스 */
.result-fall {
    background: #1a0d0d;
    border: 1px solid #7b1c1c;
    border-left: 4px solid #e24b4b;
    border-radius: 6px;
    padding: 1rem 1.4rem;
    color: #f5a0a0;
    font-size: 0.95rem;
    margin: 1rem 0 0.5rem;
}
.result-safe {
    background: #0a1a0f;
    border: 1px solid #1a5c30;
    border-left: 4px solid #2ecc71;
    border-radius: 6px;
    padding: 1rem 1.4rem;
    color: #7debb0;
    font-size: 0.95rem;
    margin: 1rem 0 0.5rem;
}

/* 지표 카드 */
[data-testid="metric-container"] {
    background: #0d1526;
    border: 1px solid #1a2e4a;
    border-radius: 6px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] {
    color: #4a7a9a !important;
    font-size: 0.7rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #7ec8f5 !important;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem !important;
}

/* 섹션 레이블 */
.section-label {
    font-size: 0.7rem;
    color: #4a7a9a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Streamlit 기본 alert */
.stAlert {
    background: #0d1526 !important;
    border-radius: 6px !important;
}

/* 푸터 */
.footer-text {
    text-align: center;
    color: #1e3a5a;
    font-size: 0.65rem;
    letter-spacing: 1px;
    margin-top: 0.5rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── 헤더 ──────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🛡 SafeWatch AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">'
    'CCTV 전도 사고 자동 감지 시스템 &nbsp;·&nbsp; '
    '<span class="badge-online">● SYSTEM ONLINE</span>'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── 3단 안내 ──────────────────────────────────────────────────
st.markdown("""
<div class="guide-grid">
  <div class="guide-card">
    <strong>📁 영상 업로드</strong>
    분석할 CCTV 영상 파일을 업로드하세요.
  </div>
  <div class="guide-card">
    <strong>🤖 AI 자동 분석</strong>
    딥러닝 모델이 전도 여부를 판별합니다.
  </div>
  <div class="guide-card">
    <strong>⚡ 즉시 결과 확인</strong>
    신뢰도와 추론 속도를 즉시 제공합니다.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── 파일 업로더 ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "동영상 파일을 선택하거나 드래그하세요",
    type=["mp4", "avi", "mov"],
    help="지원 형식: MP4, AVI, MOV  |  최소 10프레임 이상의 영상을 권장합니다.",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    st.video(uploaded_file)
    st.markdown("")

    if st.button("▶  AI 분석 시작", type="primary"):
        with st.spinner("AI 엔진 분석 중..."):
            try:
                files = {
                    "video": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()

                    if result.get("status") == "success":
                        # main.py 응답 구조: result["data"] 안에 핵심 지표 포함
                        data       = result["data"]
                        is_fall    = data["predicted_class"] == 1
                        confidence = data["confidence"]
                        infer_ms   = data["inference_time_ms"]

                        st.markdown('<hr class="divider">', unsafe_allow_html=True)
                        st.markdown(
                            '<p class="section-label">분석 결과</p>',
                            unsafe_allow_html=True,
                        )

                        if is_fall:
                            st.markdown(
                                f'<div class="result-fall">'
                                f"⚠️ 전도 상황이 감지되었습니다 &nbsp;—&nbsp; 신뢰도 {confidence * 100:.1f}%"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="result-safe">'
                                f"✅ 정상 상태입니다 &nbsp;—&nbsp; 신뢰도 {confidence * 100:.1f}%"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        st.markdown("")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("최종 판정", "전도 감지" if is_fall else "정상")
                        c2.metric("신뢰도",    f"{confidence:.4f}")
                        c3.metric("추론 속도", f"{infer_ms} ms")

                    else:
                        # main.py 가 status="error" 로 응답한 경우
                        st.error(f"서버 오류: {result.get('message', '알 수 없는 오류')}")

                elif response.status_code == 422:
                    st.error("파일 형식 오류: 서버가 요청을 처리하지 못했습니다. (HTTP 422)")
                else:
                    st.error(f"백엔드 응답 실패 (HTTP {response.status_code})")

            except requests.exceptions.Timeout:
                st.error("⏱ 요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
            except Exception as e:
                st.error(f"예상치 못한 오류: {e}")

else:
    st.markdown(
        '<p style="text-align:center; color:#2a4a6a; font-size:0.82rem; padding: 1.5rem 0;">'
        "📌 영상을 업로드하면 미리보기가 표시되고 분석 버튼이 나타납니다."
        "</p>",
        unsafe_allow_html=True,
    )

# ── 푸터 ──────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p class="footer-text">'
    "SafeWatch AI &nbsp;·&nbsp; CCTV Fall Detection System &nbsp;·&nbsp; Powered by Deep Learning"
    "</p>",
    unsafe_allow_html=True,
)