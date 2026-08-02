# -*- coding: utf-8 -*-
"""
세법 룰 엔진 — 기획안 03. 핵심 구조 / "결정" 단계, 08. 법적 검토 반영.

단순화 가정 (데모용, README에도 명시):
  - 소득공제는 '기본공제(본인 1인, 150만원)'만 반영. 인적공제(부양가족 등)는 생략.
  - 세액공제는 '표준세액공제(7만원, 소득세법 59조의4)'만 반영. 그 외 특별세액공제는 생략.
  - 국민연금 등 사회보험료 공제는 개인별 실제 납부액 데이터가 없어 생략
    (기획안 10. 한계 ④와 일치).
  - 단순/기준경비율 판정은 데모용 2,400만원 단일 기준을 사용한다. 실제 판정은 업종,
    귀속연도, 직전연도 수입 및 신규사업자 여부 등에 따라 달라질 수 있다.
  - 지방소득세(소득세의 10%)를 합산해 '실효 결정세액'을 계산한다.
  - 3.3% 원천징수는 '기납부세액'으로 차감한다.

2025년 종합소득세 누진세율표(단위: 원) — 실제 세율표를 그대로 사용.
"""

from dataclasses import dataclass

from data.industry_codes import INDUSTRY_CODES, get_candidate_codes

BASIC_DEDUCTION = 1_500_000  # 본인 기본공제
WITHHOLDING_RATE = 0.033  # 3.3% 원천징수 (사업소득)
LOCAL_TAX_RATE = 0.10  # 지방소득세 = 소득세의 10%
STANDARD_TAX_CREDIT = 70_000  # 표준세액공제(소득세법 59조의4) — 근로소득 없는 사업소득자, 성실신고 대상 아닌 경우

# (하한, 상한, 세율, 누진공제)
TAX_BRACKETS_2025 = [
    (0, 14_000_000, 0.06, 0),
    (14_000_000, 50_000_000, 0.15, 1_260_000),
    (50_000_000, 88_000_000, 0.24, 5_760_000),
    (88_000_000, 150_000_000, 0.35, 15_440_000),
    (150_000_000, 300_000_000, 0.38, 19_940_000),
    (300_000_000, 500_000_000, 0.40, 25_940_000),
    (500_000_000, 1_000_000_000, 0.42, 35_940_000),
    (1_000_000_000, float("inf"), 0.45, 65_940_000),
]

SIMPLE_EXPENSE_RATE_CAP_INCOME = 24_000_000  # 단순경비율 적용 유지 기준(데모 근사치)


@dataclass
class TaxResult:
    industry_code: str
    industry_name: str
    annual_income: int
    expense_rate: float
    necessary_expense: int
    taxable_income: int  # 소득금액 (수입 - 필요경비)
    tax_base: int  # 과세표준 (소득금액 - 기본공제)
    income_tax: int  # 종합소득세 산출세액
    local_tax: int  # 지방소득세
    total_tax: int  # 총 결정세액(소득세+지방세)
    withheld_tax: int  # 기납부(원천징수 3.3%) 추정액
    additional_payment: int  # 5월에 추가로 낼 것으로 예상되는 금액 (총세액 - 기납부, 음수면 환급)
    expense_method: str = "단순경비율"  # 적용된 경비율 방식


def progressive_tax(tax_base: int) -> int:
    """과세표준에 누진세율을 적용해 산출세액을 계산한다."""
    if tax_base <= 0:
        return 0
    for lower, upper, rate, deduction in TAX_BRACKETS_2025:
        if lower < tax_base <= upper or (upper == float("inf") and tax_base > lower):
            return max(0, round(tax_base * rate - deduction))
    return 0


def compute_tax(annual_income: int, industry_code: str) -> TaxResult:
    """연간 수입금액과 업종코드로 종합소득세 예상액을 계산한다.
    기준수입금액(2,400만원) 이하면 단순경비율, 초과하면 기준경비율 적용.
    (기준경비율 방식은 원래 '주요경비 증빙'을 별도 차감하지만, 증빙 데이터가 없어
    경비율만 적용한 보수적(세금이 많게 나오는 쪽) 추정치다.)"""
    info = INDUSTRY_CODES[industry_code]
    if annual_income <= SIMPLE_EXPENSE_RATE_CAP_INCOME:
        applied_rate = info.simple_expense_rate
        method = "단순경비율"
    else:
        applied_rate = info.standard_expense_rate
        method = "기준경비율"
    expense = round(annual_income * applied_rate)
    taxable_income = max(0, annual_income - expense)
    tax_base = max(0, taxable_income - BASIC_DEDUCTION)
    income_tax = progressive_tax(tax_base)
    income_tax = max(0, income_tax - STANDARD_TAX_CREDIT)  # 표준세액공제 반영
    local_tax = round(income_tax * LOCAL_TAX_RATE)
    total_tax = income_tax + local_tax
    withheld = round(annual_income * WITHHOLDING_RATE)
    additional = total_tax - withheld

    return TaxResult(
        industry_code=industry_code,
        industry_name=info.name,
        annual_income=annual_income,
        expense_rate=applied_rate,
        necessary_expense=expense,
        taxable_income=taxable_income,
        tax_base=tax_base,
        income_tax=income_tax,
        local_tax=local_tax,
        total_tax=total_tax,
        withheld_tax=withheld,
        additional_payment=additional,
        expense_method=method,
    )


def effective_reserve_rate(annual_income: int, industry_code: str) -> float:
    """예상 연소득 기준 추가 납부 대비율.

    지급자가 3.3%를 이미 원천징수해 납부했다는 가정 아래, 예상 추가 납부액만
    연간 수입금액 대비 비율로 환산한다. 원천징수액을 다시 적립하지 않는다.
    """
    if annual_income <= 0:
        return 0.0
    result = compute_tax(annual_income, industry_code)
    shortfall = max(0, result.additional_payment)
    return shortfall / annual_income


def compute_deposit_reserve(deposit_amount: int, predicted_annual_income: int, industry_code: str) -> dict:
    """소득액을 기준으로 추가 납부에 대비할 참고 금액을 산출한다.

    실제 계좌 조회, 자금 보관 또는 이체는 수행하지 않는다.
    """
    rate = effective_reserve_rate(predicted_annual_income, industry_code)
    reserve = round(deposit_amount * rate)
    return {
        "deposit_amount": deposit_amount,
        "reserve_rate": round(rate, 4),
        "reserve_amount": reserve,
        "net_after_reserve": deposit_amount - reserve,
    }


def compare_industry_codes(annual_income: int, candidate_codes: list) -> list:
    """후보 업종코드별 세액을 비교해 정렬된 리스트로 반환한다.
    01. 문제 ① 국정감사 실사례(940909 vs 940918)를 재현하는 로직."""
    results = [compute_tax(annual_income, code) for code in candidate_codes]
    results.sort(key=lambda r: r.total_tax)
    return results


def get_industry_code_candidates(annual_income: int, platform: str) -> dict:
    """플랫폼에 연결된 검토 후보를 반환한다.

    첫 후보는 플랫폼 매핑상 대표 후보일 뿐, 세액이 가장 낮다는 이유로 선택하지 않는다.
    실제 신고 코드는 수행한 용역의 사실관계로 확인해야 한다.
    """
    candidates = get_candidate_codes(platform)
    results = [compute_tax(annual_income, code) for code in candidates]
    representative = results[0]
    return {
        "platform": platform,
        "candidates": results,
        "representative": representative,
    }
