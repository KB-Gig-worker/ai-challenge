# -*- coding: utf-8 -*-
"""
RealPay MVP 데모 — 기획안 07. 화면 3개.

실행:
    cd realpay
    streamlit run app/streamlit_app.py

사전 준비(최초 1회 또는 데이터/모델 갱신 시):
    python scripts/run_pipeline.py

앱 흐름: 계좌(worker_id) 조회 -> 이력 3개월 미만이면 온보딩 설문 자동 진입 ->
        완료/기존회원이면 대시보드로 자동 진입. 사이드바 "데모/발표용 컨트롤"에서
        발표 중 다른 워커 프로필로 즉시 전환 가능(로그인 절차 우회).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.industry_codes import list_all_platforms  # noqa: E402
from data.survey import load_questions, options_for, survey_to_profile  # noqa: E402
from engine.tax_engine import (  # noqa: E402
    compute_tax,
    compute_deposit_reserve,
    get_industry_code_candidates,
    SIMPLE_EXPENSE_RATE_CAP_INCOME,
)
from engine.stability import assess_stability
from model.features import build_feature_table  # noqa: E402
from model.predict import IncomePredictor  # noqa: E402
from report.llm_report import build_insight_context, generate_llm_report  # noqa: E402

st.set_page_config(page_title="RealPay · KB AI Challenge", page_icon="\U0001F4B0", layout="wide")

st.markdown("""
<style>
:root {
    --kb-yellow: #FFBC00;
    --kb-dark: #17171A;
}

.main .block-container {
    max-width: 480px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

[data-testid="stAppViewContainer"] {
    background-color: #FAFAFA;
}

[data-testid="stSidebar"] {
    background-color: var(--kb-dark);
}
[data-testid="stSidebar"] * {
    color: #F5F5F7 !important;
}

h1 {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: var(--kb-dark) !important;
}
h2, h3 {
    color: var(--kb-dark) !important;
}

div[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 14px 16px;
    border: 1px solid #ECECEE;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="stMetricValue"] {
    color: var(--kb-dark);
    font-weight: 700;
}
div[data-testid="stMetricLabel"] {
    color: #8A8A93;
}

.stButton>button, [data-testid="stFormSubmitButton"] button {
    background-color: var(--kb-yellow) !important;
    color: var(--kb-dark) !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 0.6rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = ROOT / "data"

PATTERN_GUESS_MAP = {
    "매달 비슷해요(규칙형)": "regular",
    "성수기/비수기가 뚜렷해요(계절형)": "seasonal",
    "달마다 들쭉날쭉해요(불규칙형)": "irregular",
}


def estimate_from_similar_workers(profile: dict, workers_df: pd.DataFrame, deposits_df: pd.DataFrame) -> dict:
    """온보딩 설문 답변(플랫폼, 소득패턴)으로 비슷한 워커들을 찾아 평균 소득을 추정치로 사용.
    본인이 직접 적은 예상소득과 코호트 평균을 절반씩 섞어 최종 추정치를 만든다."""
    pattern = profile.get("pattern") or PATTERN_GUESS_MAP.get(profile.get("pattern_guess"))
    platform = profile.get("primary_income")

    matched = workers_df[(workers_df["primary_platform"] == platform) & (workers_df["pattern"] == pattern)]
    match_desc = f"플랫폼({platform}) + 소득패턴 모두 일치"
    if matched.empty:
        matched = workers_df[workers_df["primary_platform"] == platform]
        match_desc = f"플랫폼({platform})만 일치"
    if matched.empty:
        matched = workers_df[workers_df["pattern"] == pattern]
        match_desc = "소득패턴만 일치"
    if matched.empty:
        matched = workers_df
        match_desc = "일치하는 조건 없음 (전체 워커 평균)"

    matched_deposits = deposits_df[deposits_df["worker_id"].isin(matched["worker_id"])]
    self_report = float(profile["expected_monthly"])
    cohort_avg = float(matched_deposits["monthly_income"].mean()) if len(matched_deposits) else self_report

    return {
        "blended_estimate": 0.5 * self_report + 0.5 * cohort_avg,
        "cohort_avg": cohort_avg,
        "self_report": self_report,
        "n_matched_workers": int(len(matched)),
        "match_desc": match_desc,
    }


# ---------------------------------------------------------------- data/model

@st.cache_data
def load_data():
    workers_path = DATA_DIR / "mock_workers.csv"
    deposits_path = DATA_DIR / "mock_deposits.csv"
    if not workers_path.exists() or not deposits_path.exists():
        return None, None
    workers = pd.read_csv(workers_path)
    deposits = pd.read_csv(deposits_path)
    return workers, deposits


@st.cache_resource
def load_predictor():
    try:
        return IncomePredictor()
    except FileNotFoundError:
        return None


workers_df, deposits_df = load_data()
predictor = load_predictor()

if workers_df is None or predictor is None:
    st.error(
        "목데이터/모델이 아직 없습니다. 터미널에서 다음을 먼저 실행하세요:\n\n"
        "`cd realpay && python scripts/run_pipeline.py`"
    )
    st.stop()


# ---------------------------------------------------------------- 세션 상태 초기화

for key, default in [
    ("logged_in_worker_id", None),
    ("is_new_signup", False),
    ("onboarding_done", False),
    ("onboarding_profile", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------- 사이드바: 기본 정보 + 데모 컨트롤

st.sidebar.markdown("### R E A L P A Y")
st.sidebar.caption("가상 소득 데이터로 세금 대비 금액을 살펴보는 데모")
st.sidebar.warning("가상 데이터 · 실제 계좌 연결/조회/이체 없음")
st.sidebar.divider()

with st.sidebar.expander("🛠 데모/발표용 컨트롤", expanded=False):
    st.caption("발표 중 로그인 절차 없이 바로 다른 워커 프로필로 전환할 때 사용하세요.")
    demo_override = st.checkbox("다른 워커로 강제 전환")
    demo_worker_id = None
    demo_history_months = None
    if demo_override:
        demo_worker_id = st.selectbox(
            "worker_id",
            workers_df["worker_id"].tolist(),
            format_func=lambda x: f"#{x} · {workers_df.loc[workers_df.worker_id == x, 'primary_platform'].values[0]}",
        )
        demo_history_months = st.slider("이력 개월수 강제 조정 (콜드스타트 시뮬레이션)", 1, 24, 24)

if st.session_state["logged_in_worker_id"] is not None:
    st.sidebar.divider()
    if st.sidebar.button("로그아웃"):
        st.session_state["logged_in_worker_id"] = None
        st.session_state["is_new_signup"] = False
        st.session_state["onboarding_done"] = False
        st.session_state["onboarding_profile"] = None
        st.rerun()


# ---------------------------------------------------------------- 로그인 화면 (진입점)

if not demo_override and st.session_state["logged_in_worker_id"] is None:
    st.title("RealPay")
    st.caption("가상 소득 내역으로 다음 달 소득과 세금 대비 금액을 추정하는 개념검증 데모")
    st.info("실제 계좌를 조회하거나 돈을 이체·보관하지 않으며, 세무 신고나 세무 자문을 제공하지 않습니다.")
    st.divider()

    st.subheader("데모 프로필 선택")
    with st.form("login"):
        input_id = st.number_input(
            "데모 프로필 번호",
            min_value=1,
            step=1,
            value=1,
            help="가상 데이터 프로필은 1~450 사이 숫자입니다. 실제 계좌번호가 아닙니다.",
        )
        login_submitted = st.form_submit_button("조회하기")

    if login_submitted:
        if int(input_id) in workers_df["worker_id"].values:
            st.session_state["logged_in_worker_id"] = int(input_id)
            st.session_state["is_new_signup"] = False
            st.rerun()
        else:
            st.error("존재하지 않는 데모 프로필입니다. 새 가상 프로필은 아래 '처음 시작하기'를 눌러주세요.")

    st.divider()
    st.caption("가상 소득 이력이 없는 새 프로필을 체험하시겠어요?")
    if st.button("처음 시작하기"):
        st.session_state["logged_in_worker_id"] = int(workers_df["worker_id"].max()) + 1
        st.session_state["is_new_signup"] = True
        st.session_state["onboarding_done"] = False
        st.session_state["onboarding_profile"] = None
        st.rerun()

    st.stop()


# ---------------------------------------------------------------- 로그인 이후 공통 데이터 계산

worker_id = demo_worker_id if demo_override else st.session_state["logged_in_worker_id"]
is_known_worker = worker_id in workers_df["worker_id"].values

if is_known_worker:
    worker_row = workers_df[workers_df.worker_id == worker_id].iloc[0]
    worker_deposits_full = deposits_df[deposits_df.worker_id == worker_id].sort_values(["year", "month"])
else:
    # 신규 가입자: mock 데이터에 없는 새 계좌 — 실거래 이력 없음, 설문 프로파일로 표시
    _profile = st.session_state.get("onboarding_profile") or {}
    worker_row = pd.Series({
        "worker_id": worker_id,
        "pattern": _profile.get("pattern", "미정"),
        "primary_platform": _profile.get("primary_platform", "미정"),
    })
    worker_deposits_full = pd.DataFrame(columns=deposits_df.columns)

if demo_override:
    history_months = demo_history_months
elif st.session_state["is_new_signup"]:
    # 신규 가입자는 온보딩을 완료해도 실거래 이력이 없으므로 계속 콜드스타트로 취급한다.
    # (onboarding_done은 "온보딩 화면으로 다시 안 돌아가게" 라우팅에만 쓰고, 여기 조건에는 넣지 않는다)
    history_months = 1
else:
    history_months = len(worker_deposits_full)  # 기존 회원은 실제 이력 길이 그대로

worker_deposits = worker_deposits_full.iloc[:history_months].copy()
is_cold_start = history_months < 3


# ---------------------------------------------------------------- 화면 자동 라우팅

force_onboarding = is_cold_start and not st.session_state["onboarding_done"] and not demo_override

if force_onboarding:
    active_screen = "① 온보딩 설문"
else:
    active_screen = st.sidebar.radio(
        "화면", ["② 대시보드", "③ 리포트"], label_visibility="collapsed",
    )


# ---------------------------------------------------------------- screen 1: 온보딩 설문

if active_screen == "① 온보딩 설문":
    st.title("① 온보딩 설문")
    st.caption("가입 직후 3개월치 이력이 없을 때, 초기 프로파일을 만들기 위한 7문항 (06. 콜드스타트 처리)")

    survey = load_questions()
    q = {item["id"]: item for item in survey["questions"]}

    def _opts(qid):
        return [(o["value"], o["label"]) for o in q[qid]["options"]]

    def _label_of(qid, val):
        return next((o["label"] for o in q[qid]["options"] if o["value"] == val), val)

    # q1(직종)은 폼 밖에 두어, 선택에 따라 q4 플랫폼 후보가 즉시 갱신되게 한다.
    job_opts = _opts("q1_job")
    job = st.selectbox(
        f"1. {q['q1_job']['text']}",
        [v for v, _ in job_opts],
        format_func=lambda v: dict(job_opts)[v],
    )
    platform_choices = options_for("q4_platforms", job)

    with st.form("onboarding"):
        col1, col2 = st.columns(2)
        with col1:
            wt_opts = _opts("q2_worktype")
            work_type = st.radio(
                f"2. {q['q2_worktype']['text']}",
                [v for v, _ in wt_opts], format_func=lambda v: dict(wt_opts)[v],
            )
            pc_opts = _opts("q3_platform_count")
            platform_count = st.radio(
                f"3. {q['q3_platform_count']['text']}",
                [v for v, _ in pc_opts], format_func=lambda v: dict(pc_opts)[v], horizontal=True,
            )
            platforms = st.multiselect(
                f"4. {q['q4_platforms']['text']}", platform_choices,
                default=platform_choices[:1],
            )
        with col2:
            avg_workdays = st.slider(
                f"5. {q['q5_workdays']['text']}",
                q["q5_workdays"]["min"], q["q5_workdays"]["max"], q["q5_workdays"]["default"],
            )
            band_opts = _opts("q6_income_band")
            income_band = st.selectbox(
                f"6. {q['q6_income_band']['text']}",
                [v for v, _ in band_opts], index=2, format_func=lambda v: dict(band_opts)[v],
            )
            reg_opts = _opts("q7_regularity")
            regularity = st.radio(
                f"7. {q['q7_regularity']['text']}",
                [v for v, _ in reg_opts], format_func=lambda v: dict(reg_opts)[v],
            )
        submitted = st.form_submit_button("초기 프로파일 생성하고 대시보드로 이동")

    if submitted:
        answers = {
            "q1_job": job, "q2_worktype": work_type, "q3_platform_count": platform_count,
            "q4_platforms": platforms, "q5_workdays": avg_workdays,
            "q6_income_band": income_band, "q7_regularity": regularity,
        }
        st.session_state["onboarding_profile"] = survey_to_profile(answers)
        st.session_state["onboarding_done"] = True
        st.success("초기 프로파일이 생성되었습니다. 대시보드로 이동합니다...")
        st.rerun()


# ---------------------------------------------------------------- screen 2: 대시보드

elif active_screen == "② 대시보드":
    st.title("② 대시보드")
    st.caption(f"worker #{worker_id} · {worker_row['primary_platform']} · 소득패턴: {worker_row['pattern']}")

    if is_cold_start:
        profile = st.session_state.get("onboarding_profile")
        st.warning(
            "이력이 3개월 미만입니다 → **콜드스타트 모드**: 온보딩 설문 + 유사 워커 데이터 기반 추정치를 사용합니다."
            + ("" if profile else " (① 온보딩 설문을 먼저 작성하면 더 정확한 추정이 반영됩니다)")
        )
        this_month_income = int(worker_deposits.iloc[-1]["monthly_income"]) if len(worker_deposits) else None
        if profile:
            match = estimate_from_similar_workers(profile, workers_df, deposits_df)
            predicted_next_month = match["blended_estimate"]
            st.caption(
                f"유사 조건 워커 {match['n_matched_workers']}명({match['match_desc']})의 평균 소득 "
                f"{match['cohort_avg']:,.0f}원과 본인 예상치 {match['self_report']:,.0f}원을 절반씩 반영했습니다."
            )
            if profile.get("is_target_segment_candidate"):
                st.caption("✓ 이 프로파일은 RealPay 핵심 타깃(연 2,400~4,800만·다중 플랫폼·주업형)에 해당합니다.")
        else:
            predicted_next_month = this_month_income or 0
        predicted_annual = predicted_next_month * 12
        reserve_rate_source = "설문+유사워커 기반 추정"
    else:
        this_month_income = int(worker_deposits.iloc[-1]["monthly_income"])
        feat_table = build_feature_table(worker_deposits)
        latest_row = feat_table.dropna(subset=["lag3_income"]).iloc[[-1]]
        predicted_next_month = predictor.predict_next_month(latest_row)
        predicted_annual = predictor.predict_annual(latest_row)
        reserve_rate_source = "LightGBM 모델 예측"

    rec = get_industry_code_candidates(int(predicted_annual), worker_row["primary_platform"])
    industry_code = rec["representative"].industry_code

    tax_result = compute_tax(int(predicted_annual), industry_code)
    reserve_info = compute_deposit_reserve(this_month_income or 0, int(predicted_annual), industry_code)

    vault_balance = 0
    for _, row in worker_deposits.iterrows():
        r = compute_deposit_reserve(int(row["monthly_income"]), max(int(predicted_annual), 1), industry_code)
        vault_balance += r["reserve_amount"]

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #17171A 0%, #2A2A30 100%);
                    border-radius: 20px; padding: 24px 20px; margin-bottom: 16px; color: white;">
            <div style="font-size: 0.85rem; color: #B8B8C0; margin-bottom: 6px;">가상 추가납부 대비 누적액</div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #FFBC00;">{vault_balance:,}원</div>
            <div style="font-size: 0.8rem; color: #8A8A93; margin-top: 6px;">3.3% 원천징수 완료 가정 · 실제 보관/이체 없음</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("이번 달 소득", f"{this_month_income:,}원" if this_month_income is not None else "이력 없음")
    c2.metric(
        "이번 달 추가 대비 참고액",
        f"{reserve_info['reserve_amount']:,}원" if this_month_income is not None else "첫 입금 후 시작",
        f"{reserve_info['reserve_rate']*100:.1f}%" if this_month_income is not None else None,
    )
    c3.metric("다음 달 소득 추정치", f"{predicted_next_month:,.0f}원", reserve_rate_source)
    st.caption("소득·세액은 목데이터와 단순화된 가정에 따른 점추정치입니다. 실제 결과는 크게 달라질 수 있습니다.")

    st.divider()
    st.subheader("예측 그래프 — 실제 vs 예측")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(worker_deposits) + 1)),
        y=worker_deposits["monthly_income"],
        mode="lines+markers", name="실제 소득",
        line=dict(color="#17171A"),
    ))
    fig.add_trace(go.Scatter(
        x=[len(worker_deposits) + 1],
        y=[predicted_next_month],
        mode="markers", name="다음 달 예측",
        marker=dict(color="#FFBC00", size=14, symbol="star"),
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="개월차", yaxis_title="소득(원)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- screen 3: 리포트

else:
    st.title("③ 리포트")
    st.caption(f"worker #{worker_id} · {worker_row['primary_platform']}")

    if worker_deposits.empty:
        st.warning("아직 실거래 이력이 없습니다. 리포트는 입금 이력이 쌓이면 제공됩니다. "
                   "그전까지는 ② 대시보드에서 설문 기반 추정치를 확인하세요.")
        st.stop()

    feat_table = build_feature_table(worker_deposits)
    ready = feat_table.dropna(subset=["lag3_income"])

    if ready.empty:
        st.warning("이력이 부족해 리포트를 만들 수 없습니다. 사이드바 데모 컨트롤에서 이력 개월수를 3개월 이상으로 올려주세요.")
        st.stop()

    latest_row = ready.iloc[[-1]]
    this_month_income = int(worker_deposits.iloc[-1]["monthly_income"])
    roll3_mean = float(latest_row.iloc[0]["roll3_mean_income"])
    predicted_annual = predictor.predict_annual(latest_row)
    explanation = predictor.explain(latest_row)

    rec = get_industry_code_candidates(int(predicted_annual), worker_row["primary_platform"])
    tax_result = compute_tax(int(predicted_annual), rec["representative"].industry_code)

    st.subheader("연환산 세액 참고 추정")
    c1, c2, c3 = st.columns(3)
    c1.metric("다음 달 추정치 × 12", f"{predicted_annual:,.0f}원")
    c2.metric("참고 세액 추정치", f"{tax_result.total_tax:,}원")
    c3.metric(
        "추가 납부 참고액" if tax_result.additional_payment >= 0 else "환급 가능 참고액",
        f"{abs(tax_result.additional_payment):,}원",
    )
    method_note = ""
    if tax_result.expense_method == "기준경비율":
        method_note = " (기준수입금액 초과로 기준경비율 적용 — 주요경비 증빙 미반영 보수적 추정)"
    st.caption(
        f"※ 목데이터 기반 참고 추정치입니다. 다음 달 추정치를 12배했으며, {tax_result.expense_method}"
        f"({tax_result.expense_rate*100:.1f}%) 및 기본공제만 반영했습니다.{method_note} "
        f"데모 기준수입금액: {SIMPLE_EXPENSE_RATE_CAP_INCOME:,}원. 실제 판정·세액은 업종, 귀속연도, "
        "직전연도 수입, 다른 소득과 공제에 따라 달라지므로 홈택스 자료 또는 세무전문가에게 확인하세요."
    )

    st.divider()
    st.subheader("업종코드 검토 후보")
    st.caption("플랫폼명으로 좁힌 참고 후보입니다. 실제 신고 코드는 세액이 아니라 수행한 업무의 사실관계로 결정해야 합니다.")

    rows = []
    for r in rec["candidates"]:
        rows.append({
            "업종코드": r.industry_code,
            "업종명": r.industry_name,
            "적용 경비율": f"{r.expense_rate*100:.1f}%",
            "예상 총세액": f"{r.total_tax:,}원",
            "구분": "플랫폼 대표 후보" if r.industry_code == rec["representative"].industry_code else "추가 확인 후보",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.warning("이 표는 신고 코드 추천이나 절세 자문이 아닙니다. 지급명세서와 실제 용역 내용을 확인하세요.")

    st.divider()
    st.subheader("모델이 참고한 주요 입력 요인")
    st.caption("모델 내부 계산에 대한 설명이며, 실제 소득 변화의 원인이나 예측 정확도를 증명하지 않습니다.")

    factors = explanation["top_factors"]
    fig2 = go.Figure(go.Bar(
        x=[f["shap"] for f in factors],
        y=[f["label"] for f in factors],
        orientation="h",
        marker_color=["#A8382A" if f["shap"] < 0 else "#0F6B60" for f in factors],
    ))
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="다음 달 예측치에 대한 영향(원)")
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("소득 안정성 자가진단")
    st.caption("내 소득 상태를 스스로 확인하는 정량 지표입니다 (신용평가가 아닙니다)")

    try:
        stab = assess_stability(worker_deposits)
        grade_color = {"안정": "#0F6B60", "보통": "#B8860B", "불안정": "#A8382A"}[stab.grade]
        st.markdown(
            f"""
            <div style="display:inline-block; background:{grade_color}; color:white;
                        border-radius:12px; padding:6px 18px; font-weight:800; font-size:1.1rem;
                        margin-bottom:10px;">
                안정성 수준: {stab.grade}
            </div>
            """,
            unsafe_allow_html=True,
        )
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("변동계수(6개월)", f"{stab.cv6:.2f}")
        s2.metric("소득 추세", f"{stab.trend_pct:+.1f}%")
        s3.metric("계절성 진폭", f"{stab.seasonal_amplitude:.0f}%")
        s4.metric("활성 플랫폼", f"{stab.n_platforms}개")
        for reason in stab.grade_reasons:
            st.caption(f"· {reason}")
    except ValueError:
        stab = None
        st.info("이력이 6개월 미만이라 안정성 자가진단을 건너뜁니다.")

    st.divider()
    st.subheader("요약")
    st.caption("기본값은 로컬 템플릿입니다. 외부 LLM 사용에 동의하면 아래 재무 추정 정보가 API 제공업체로 전송될 수 있습니다.")
    use_external_llm = st.checkbox(
        "외부 LLM 전송에 동의하고 AI 요약 사용",
        value=False,
        help="worker ID는 보내지 않지만 소득·세액·안정성 추정치가 전송됩니다.",
    )

    ctx = build_insight_context(
        worker_name=f"worker #{worker_id}",
        this_month_income=this_month_income,
        roll3_mean_income=roll3_mean,
        predicted_annual_income=predicted_annual,
        top_shap_factors=factors,
        tax_result=tax_result,
        recommendation=rec,
        stability=stab,
    )
    report_text, source = generate_llm_report(ctx, prefer_api=use_external_llm)
    st.markdown(f"> {report_text}")
    if source == "claude-api":
        st.caption("생성 방식: `claude-api` (사용자 동의 후 전송)")
    elif use_external_llm:
        st.caption("생성 방식: `template` (API를 사용할 수 없어 로컬 템플릿으로 대체)")
    else:
        st.caption("생성 방식: `template` (외부 전송 없음)")

    st.divider()
    st.subheader("모델 비교 — 왜 LightGBM인가")
    st.caption("5-fold 교차검증(worker 단위) 결과 트리 기반 3개 모델은 오차범위 내 동급 — "
               "성능이 같다면 SHAP 설명가능성과 속도가 앞서는 LightGBM을 채택했습니다.")

    comparison_path = ROOT / "model" / "artifacts" / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path, encoding="utf-8") as f:
            comparison = json.load(f)
        comp_rows = []
        for name, r in comparison.items():
            comp_rows.append({
                "모델": name,
                "평균 MAE": f"{r['cv_mae_mean']:,.0f}원",
                "표준편차": f"± {r['cv_mae_std']:,.0f}원",
                "채택 여부": "⭐ 최종 채택" if name == "LightGBM" else "",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("비교 결과가 없습니다. `python model/compare_baselines.py` 를 먼저 실행하세요.")
