# -*- coding: utf-8 -*-
"""
소득 안정성 진단 — 규칙 기반 정량 지표.

원칙: 안정성 평가는 전부 계산 가능한 정량 지표로 하고, LLM은 결과를
자연어로 풀어주는 역할만 한다 (llm_report.py와 동일한 설계 철학).
이 지표들이 곧 KB 입장에서의 '검증된 대안신용평가 데이터'다.

지표:
  - cv6         : 최근 6개월 변동계수 (표준편차/평균, 낮을수록 안정)
  - trend_pct   : 최근 3개월 평균 vs 이전 3개월 평균 증감률(%)
  - weak_months : 최근 6개월 중 6개월 평균의 50% 미만인 달 수
  - n_platforms : 활성 플랫폼 수 (분산도, 많을수록 외부 충격에 강함)
  - grade       : 종합 등급 (안정 / 보통 / 불안정)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StabilityResult:
    cv6: float           # 변동계수 (0~)
    trend_pct: float     # 최근 3개월 vs 이전 3개월 증감률 (%)
    weak_months: int     # 급감 개월 수 (0~6)
    n_platforms: int
    seasonal_amplitude: float  # 성수기 vs 비수기 편차 (%)
    grade: str           # "안정" | "보통" | "불안정"
    grade_reasons: list  # 등급 판정 근거 문장 리스트


# 등급 기준선 (mock 데이터 패턴별 노이즈 설계 기준: regular 8% / seasonal 12% / irregular 28%)
CV_STABLE = 0.15
CV_UNSTABLE = 0.30


def assess_stability(worker_deposits: pd.DataFrame) -> StabilityResult:
    """월별 입금 이력(worker_deposits: monthly_income, n_platforms_active 컬럼 필요)으로
    소득 안정성을 진단한다. 최소 6개월 이력 필요 — 부족하면 ValueError."""
    if len(worker_deposits) < 6:
        raise ValueError("안정성 진단에는 최소 6개월 이력이 필요합니다.")

    recent6 = worker_deposits["monthly_income"].iloc[-6:].astype(float)
    mean6 = float(recent6.mean())
    cv6 = float(recent6.std(ddof=0) / mean6) if mean6 > 0 else float("inf")

    recent3 = float(worker_deposits["monthly_income"].iloc[-3:].mean())
    prev3 = float(worker_deposits["monthly_income"].iloc[-6:-3].mean())
    trend_pct = ((recent3 - prev3) / prev3 * 100) if prev3 > 0 else 0.0

    weak_months = int((recent6 < mean6 * 0.5).sum())

    n_platforms = int(worker_deposits["n_platforms_active"].iloc[-1])

    # 계절성 진폭: 최근 12개월(부족하면 전체) 중 상위 3개월 평균 vs 하위 3개월 평균 편차
    recent12 = worker_deposits["monthly_income"].iloc[-12:].astype(float)
    top3 = float(recent12.nlargest(3).mean())
    bottom3 = float(recent12.nsmallest(3).mean())
    seasonal_amplitude = ((top3 - bottom3) / recent12.mean() * 100) if recent12.mean() > 0 else 0.0

    # --- 등급 판정 (규칙 기반) ---
    reasons = []
    score = 0  # 높을수록 불안정

    if cv6 <= CV_STABLE:
        reasons.append(f"최근 6개월 변동계수 {cv6:.2f} — 안정 기준({CV_STABLE}) 이내")
    elif cv6 <= CV_UNSTABLE:
        score += 1
        reasons.append(f"최근 6개월 변동계수 {cv6:.2f} — 보통 수준")
    else:
        score += 2
        reasons.append(f"최근 6개월 변동계수 {cv6:.2f} — 불안정 기준({CV_UNSTABLE}) 초과")

    if trend_pct <= -15:
        score += 1
        reasons.append(f"최근 3개월 평균이 이전 3개월 대비 {trend_pct:.1f}% 하락")
    elif trend_pct >= 15:
        reasons.append(f"최근 3개월 평균이 이전 3개월 대비 {trend_pct:.1f}% 상승")

    if weak_months >= 2:
        score += 1
        reasons.append(f"최근 6개월 중 소득 급감(평균 50% 미만) 달이 {weak_months}회")
    elif weak_months == 1:
        reasons.append("최근 6개월 중 소득 급감 달이 1회 있었음")

    if n_platforms >= 2:
        score -= 1  # 분산되어 있으면 한 단계 완화
        reasons.append(f"활성 플랫폼 {n_platforms}개 — 소득원이 분산되어 있음")
    else:
        reasons.append("활성 플랫폼 1개 — 단일 소득원 의존")

    if score <= 0:
        grade = "안정"
    elif score <= 2:
        grade = "보통"
    else:
        grade = "불안정"

    return StabilityResult(
        cv6=round(cv6, 3),
        trend_pct=round(trend_pct, 1),
        weak_months=weak_months,
        n_platforms=n_platforms,
        seasonal_amplitude=round(seasonal_amplitude, 1),
        grade=grade,
        grade_reasons=reasons,
    )