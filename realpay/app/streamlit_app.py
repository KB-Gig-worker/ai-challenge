# -*- coding: utf-8 -*-
"""
RealPay MVP 데모 — 기획안 07. 화면 3개.

실행:
    cd realpay
    streamlit run app/streamlit_app.py

사전 준비(최초 1회 또는 데이터/모델 갱신 시):
    python scripts/run_pipeline.py
"""

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
    recommend_industry_code,
    SIMPLE_EXPENSE_RATE_CAP_INCOME,
)
from model.features import build_feature_table  # noqa: E402
from model.predict import IncomePredictor  # noqa: E402
from report.llm_report import build_insight_context, generate_llm_report  # noqa: E402

st.set_page_config(page_title="RealPay · KB AI Challenge", page_icon="\U0001F4B0", layout="wide")

DATA_DIR = ROOT / "data"


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


# ---------------------------------------------------------------- sidebar

st.sidebar.markdown("### R E A L P A Y")
st.sidebar.caption("긱워커 통장 입금 -> 세금 예측 -> 자동 적립 AI 에이전트")

screen = st.sidebar.radio(
    "화면",
    ["① 온보딩 설문", "② 대시보드", "③ 리포트"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("**시뮬레이션 대상 긱워커**")
worker_id = st.sidebar.selectbox(
    "worker_id",
    workers_df["worker_id"].tolist(),
    format_func=lambda x: f"#{x} · {workers_df.loc[workers_df.worker_id == x, 'primary_platform'].values[0]}",
)

worker_row = workers_df[workers_df.worker_id == worker_id].iloc[0]
history_months = st.sidebar.slider(
    "지금까지 쌓인 이력(개월) — 콜드스타트 시뮬레이션", 1, 24, 24,
    help="06. 데이터 계획: 3개월 미만이면 온보딩 설문 기반 추정치를 쓰고, "
         "3개월 이상 쌓이면 실제 이력 기반 모델 예측으로 전환합니다.",
)

worker_deposits_full = deposits_df[deposits_df.worker_id == worker_id].sort_values(["year", "month"])
worker_deposits = worker_deposits_full.iloc[:history_months].copy()

is_cold_start = history_months < 3


# ---------------------------------------------------------------- screen 1

if screen == "① 온보딩 설문":
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
        submitted = st.form_submit_button("초기 프로파일 생성")

    if submitted:
        answers = {
            "q1_job": job, "q2_worktype": work_type, "q3_platform_count": platform_count,
            "q4_platforms": platforms, "q5_workdays": avg_workdays,
            "q6_income_band": income_band, "q7_regularity": regularity,
        }
        st.session_state["onboarding_profile"] = survey_to_profile(answers)
        prof = st.session_state["onboarding_profile"]
        st.success(
            "초기 프로파일이 생성되었습니다. ② 대시보드에서 '콜드스타트 이력 개월 수'를 3개월 미만으로 두면 "
            "이 설문 기반 추정치가 사용됩니다."
        )
        if prof["is_target_segment_candidate"]:
            st.info("이 응답은 RealPay 핵심 타깃(연 2,400~4,800만·다중 플랫폼·주업형)에 해당합니다.")

    if "onboarding_profile" in st.session_state:
        st.json(st.session_state["onboarding_profile"])


# ---------------------------------------------------------------- screen 2

elif screen == "② 대시보드":
    st.title("② 대시보드")
    st.caption(f"worker #{worker_id} · {worker_row['primary_platform']} · 소득패턴: {worker_row['pattern']}")

    if is_cold_start:
        profile = st.session_state.get("onboarding_profile")
        st.warning(
            "이력이 3개월 미만입니다 → **콜드스타트 모드**: 온보딩 설문 기반 추정치를 사용합니다."
            + ("" if profile else " (① 온보딩 설문을 먼저 작성하면 더 정확한 추정이 반영됩니다)")
        )
        this_month_income = int(worker_deposits.iloc[-1]["monthly_income"]) if len(worker_deposits) else 0
        predicted_next_month = float(profile["expected_monthly"]) if profile else this_month_income
        predicted_annual = predicted_next_month * 12
        reserve_rate_source = "설문 기반 추정"
    else:
        this_month_income = int(worker_deposits.iloc[-1]["monthly_income"])
        feat_table = build_feature_table(worker_deposits)
        latest_row = feat_table.dropna(subset=["lag3_income"]).iloc[[-1]]
        predicted_next_month = predictor.predict_next_month(latest_row)
        predicted_annual = predictor.predict_annual(latest_row)
        reserve_rate_source = "LightGBM 모델 예측"

    tax_result = compute_tax(int(predicted_annual), "940909")
    reserve_info = compute_deposit_reserve(this_month_income, int(predicted_annual), "940909")

    # 세금 금고 잔고: 이력 전체에 대해 매달 적립했다고 가정하고 누적
    vault_balance = 0
    for _, row in worker_deposits.iterrows():
        r = compute_deposit_reserve(int(row["monthly_income"]), max(int(predicted_annual), 1), "940909")
        vault_balance += r["reserve_amount"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("이번 달 소득", f"{this_month_income:,}원")
    c2.metric("이번 입금 적립 필요액", f"{reserve_info['reserve_amount']:,}원", f"{reserve_info['reserve_rate']*100:.1f}%")
    c3.metric("세금 금고 잔고(누적)", f"{vault_balance:,}원")
    c4.metric("다음 달 예상 소득", f"{predicted_next_month:,.0f}원", reserve_rate_source)

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
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------- screen 3

else:
    st.title("③ 리포트")
    st.caption(f"worker #{worker_id} · {worker_row['primary_platform']}")

    feat_table = build_feature_table(worker_deposits)
    ready = feat_table.dropna(subset=["lag3_income"])

    if ready.empty:
        st.warning("이력이 부족해 리포트를 만들 수 없습니다. 콜드스타트 이력 개월 수를 3개월 이상으로 올려주세요.")
        st.stop()

    latest_row = ready.iloc[[-1]]
    this_month_income = int(worker_deposits.iloc[-1]["monthly_income"])
    roll3_mean = float(latest_row.iloc[0]["roll3_mean_income"])
    predicted_annual = predictor.predict_annual(latest_row)
    explanation = predictor.explain(latest_row)

    rec = recommend_industry_code(int(predicted_annual), worker_row["primary_platform"])
    tax_result = compute_tax(int(predicted_annual), rec["recommended"].industry_code)

    st.subheader("연간 예상 세액")
    c1, c2, c3 = st.columns(3)
    c1.metric("연간 예상 소득", f"{predicted_annual:,.0f}원")
    c2.metric("예상 총 결정세액(소득세+지방세)", f"{tax_result.total_tax:,}원")
    c3.metric(
        "5월 추가 납부 예상" if tax_result.additional_payment >= 0 else "5월 환급 예상",
        f"{abs(tax_result.additional_payment):,}원",
    )
    st.caption(
        f"※ 예상치이며 단순경비율({tax_result.expense_rate*100:.1f}%) 및 기본공제만 반영한 근사치입니다. "
        f"단순경비율 유지 기준선: {SIMPLE_EXPENSE_RATE_CAP_INCOME:,}원"
    )

    st.divider()
    st.subheader("업종코드 추천")
    st.caption("같은 소득이라도 업종코드에 따라 세액이 달라집니다 (01. 문제 ① 국정감사 실사례와 동일한 로직).")

    rows = []
    for r in rec["candidates"]:
        rows.append({
            "업종코드": r.industry_code,
            "업종명": r.industry_name,
            "단순경비율": f"{r.expense_rate*100:.1f}%",
            "예상 총세액": f"{r.total_tax:,}원",
            "추천": "⭐ 최적" if r.industry_code == rec["recommended"].industry_code else "",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    if rec["max_savings_vs_worst"] > 0:
        st.info(f"최적 코드로 신고하면 최대 **{rec['max_savings_vs_worst']:,}원** 절세 효과가 있습니다.")

    st.divider()
    st.subheader("SHAP 근거 — 왜 이 예측이 나왔는가")

    factors = explanation["top_factors"]
    fig2 = go.Figure(go.Bar(
        x=[f["shap"] for f in factors],
        y=[f["label"] for f in factors],
        orientation="h",
        marker_color=["#A8382A" if f["shap"] < 0 else "#0F6B60" for f in factors],
    ))
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="다음 달 예측치에 대한 영향(원)")
    st.plotly_chart(fig2, width='stretch')

    st.divider()
    st.subheader("LLM 요약")

    ctx = build_insight_context(
        worker_name=f"worker #{worker_id}",
        this_month_income=this_month_income,
        roll3_mean_income=roll3_mean,
        predicted_annual_income=predicted_annual,
        top_shap_factors=factors,
        tax_result=tax_result,
        recommendation=rec,
    )
    report_text, source = generate_llm_report(ctx)
    st.markdown(f"> {report_text}")
    st.caption(f"생성 방식: `{source}`" + ("" if source == "claude-api" else " (ANTHROPIC_API_KEY 미설정 — 템플릿 폴백)"))
