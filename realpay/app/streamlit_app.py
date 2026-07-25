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

    with st.form("onboarding"):
        col1, col2 = st.columns(2)
        with col1:
            job = st.text_input("1. 직업(플랫폼 안에서의 역할)", value="배달 라이더")
            primary_income = st.selectbox("2. 주수입원 플랫폼", list_all_platforms())
            n_platforms = st.slider("3. 현재 등록한 플랫폼 개수", 1, 5, 1)
            avg_workdays = st.slider("4. 한 달 평균 근무일수", 0, 31, 20)
        with col2:
            pattern_guess = st.radio(
                "5. 소득이 어떤 편인가요?",
                ["매달 비슷해요(규칙형)", "성수기/비수기가 뚜렷해요(계절형)", "달마다 들쭉날쭉해요(불규칙형)"],
            )
            expected_monthly = st.number_input("6. 예상 월평균 소득(원)", min_value=0, value=2_000_000, step=100_000)
            tax_knowledge = st.select_slider(
                "7. 세금 지식 수준", options=["전혀 모름", "3.3% 정도만 앎", "종합소득세 신고 경험 있음"]
            )
        submitted = st.form_submit_button("초기 프로파일 생성")

    if submitted:
        st.session_state["onboarding_profile"] = {
            "job": job,
            "primary_income": primary_income,
            "n_platforms": n_platforms,
            "avg_workdays": avg_workdays,
            "pattern_guess": pattern_guess,
            "expected_monthly": expected_monthly,
            "tax_knowledge": tax_knowledge,
        }
        st.success("초기 프로파일이 생성되었습니다. ② 대시보드에서 '콜드스타트 이력 개월 수'를 3개월 미만으로 두면 "
                   "이 설문 기반 추정치가 사용되는 것을 확인할 수 있습니다.")

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
    st.plotly_chart(fig, use_container_width=True)


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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
    st.plotly_chart(fig2, use_container_width=True)

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
