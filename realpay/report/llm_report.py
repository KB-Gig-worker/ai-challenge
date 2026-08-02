# -*- coding: utf-8 -*-
"""
LLM 리포트 생성 — 기획안 05. AI는 어디에 들어가는가 ② (보조 축)
04. MVP 범위: "수치 -> 자연어 인사이트".

두 가지 경로:
  1) ANTHROPIC_API_KEY 환경변수가 있으면 실제 Claude API로 자연어 리포트 생성.
  2) 없으면(데모/오프라인 환경) 같은 입력을 받는 결정론적 템플릿 렌더러로 대체.
     -> 데모가 API 키 유무에 흔들리지 않도록 하기 위함. 발표 리허설/네트워크 단절
     상황에서도 화면이 죽지 않는다.
"""

import os
from engine.tax_engine import SIMPLE_EXPENSE_RATE_CAP_INCOME


def build_insight_context(
    worker_name: str,
    this_month_income: int,
    roll3_mean_income: float,
    predicted_annual_income: float,
    top_shap_factors: list,
    tax_result,
    recommendation: dict,
    stability=None,
) -> dict:
    """모델/엔진 출력들을 리포트 생성에 필요한 하나의 컨텍스트로 모은다."""
    pct_vs_3m = 0.0
    if roll3_mean_income:
        pct_vs_3m = (this_month_income - roll3_mean_income) / roll3_mean_income * 100

    under_cap = predicted_annual_income < SIMPLE_EXPENSE_RATE_CAP_INCOME

    return {
        "worker_name": worker_name,
        "this_month_income": int(this_month_income),
        "roll3_mean_income": int(roll3_mean_income),
        "pct_vs_3m": round(pct_vs_3m, 1),
        "predicted_annual_income": int(predicted_annual_income),
        "under_simple_expense_cap": under_cap,
        "top_shap_factors": top_shap_factors,
        "representative_code": recommendation["representative"].industry_code,
        "representative_name": recommendation["representative"].industry_name,
        "estimated_total_tax": int(tax_result.total_tax),
        "additional_payment": int(tax_result.additional_payment),
        "stability_grade": stability.grade if stability else None,
        "stability_cv6": stability.cv6 if stability else None,
        "stability_trend_pct": stability.trend_pct if stability else None,
        "stability_n_platforms": stability.n_platforms if stability else None,
    }


def render_template_report(ctx: dict) -> str:
    """오프라인 폴백: 결정론적 템플릿. 기획안 05.의 '출력 예시' 문장 스타일을 그대로 따른다."""
    lines = []

    cmp_word = "낮습니다" if ctx["pct_vs_3m"] < 0 else "높습니다"
    lines.append(
        f"이번 달은 지난 3개월 평균보다 {abs(ctx['pct_vs_3m'])}% {cmp_word}."
    )

    if ctx["under_simple_expense_cap"]:
        lines.append(
            f"다음 달 추정치의 단순 연환산 값이 {ctx['predicted_annual_income']:,}원으로, "
            f"기준선 {SIMPLE_EXPENSE_RATE_CAP_INCOME:,}원 아래라 단순경비율 대상이 유지됩니다."
        )
    else:
        lines.append(
            f"다음 달 추정치의 단순 연환산 값이 {ctx['predicted_annual_income']:,}원으로, "
            f"기준선 {SIMPLE_EXPENSE_RATE_CAP_INCOME:,}원을 넘어설 것으로 보여 "
            f"단순경비율 유지 여부를 확인해야 합니다."
        )

    if ctx["top_shap_factors"]:
        f0 = ctx["top_shap_factors"][0]
        direction = "낮추는" if f0["shap"] < 0 else "높이는"
        lines.append(
            f"가장 큰 영향을 준 요인은 '{f0['label']}'로, 이번 예측을 {direction} "
            f"방향으로 약 {abs(f0['pct_of_prediction'])}% 작용했습니다."
        )

    if ctx["additional_payment"] > 0:
        lines.append(
            f"지금까지의 3.3% 원천징수만으로는 부족해, 5월에 약 {ctx['additional_payment']:,}원을 "
            f"추가로 낼 가능성이 있습니다. 원천징수 내역과 공제자료를 확인한 뒤 대비 금액을 조정하세요."
        )
    else:
        lines.append(
            f"현재 추세라면 원천징수액이 예상 세액보다 많아, 5월에 약 {abs(ctx['additional_payment']):,}원 "
            f"환급이 예상됩니다."
        )

    if ctx.get("stability_grade"):
        lines.append(
            f"소득 안정성 수준은 '{ctx['stability_grade']}'입니다 "
            f"(최근 6개월 변동계수 {ctx['stability_cv6']}, "
            f"3개월 추세 {ctx['stability_trend_pct']:+.1f}%, "
            f"활성 플랫폼 {ctx['stability_n_platforms']}개 기준)."
        )

    return " ".join(lines)


def generate_llm_report(ctx: dict, prefer_api: bool = False) -> tuple:
    """(리포트 텍스트, 생성 방식) 튜플을 반환한다. 방식은 'claude-api' 또는 'template'."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if prefer_api and api_key:
        try:
            return _generate_via_claude(ctx, api_key), "claude-api"
        except Exception:
            pass
    return render_template_report(ctx), "template"


def _generate_via_claude(ctx: dict, api_key: str) -> str:
    import json
    import anthropic  # 지연 임포트: 패키지 없거나 키 없으면 폴백 경로로 빠짐

    client = anthropic.Anthropic(api_key=api_key)
    system = (
        "너는 긱워커를 위한 세금 대비 참고정보 도구 RealPay의 리포트 작성기다. "
        "아래 JSON 데이터만 근거로, 과장 없이 담백한 한국어 리포트를 다음 4단계 구조로 써라: "
        "1) 한 줄 진단(안정성 요약과 가장 큰 리스크), "
        "2) 이번 달 소득(3개월 평균 대비 %와 추세), "
        "3) 세금 대비(단순 연환산 값, 참고 세액 추정치, 추가 납부 대비 참고액, 데모 기준선 여부), "
        "4) 리스크와 행동 제안(비수기 대비, 플랫폼 분산 등). "
        "숫자는 반드시 입력값 그대로 사용해라. '신용점수', '신용등급', '등급'이라는 단어는 쓰지 말고 "
        "'안정성 수준'이라고 표현해라. 세액은 항상 '참고 추정치'라고 표현하고, "
        "업종코드를 신고용으로 추천하거나 절세 효과를 주장하지 마라. "
        "이 리포트는 본인 확인용 자가진단임을 벗어나는 해석(대출 가능성, 타인과의 비교 백분위 등)은 하지 마라."
    )
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}],
    )
    return message.content[0].text.strip()
