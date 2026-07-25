# -*- coding: utf-8 -*-
"""
피처 엔지니어링 — 기획안 05. AI는 어디에 들어가는가 ①
입력: 과거 월별 소득, 근무일수, 플랫폼 수, 계절/공휴일, (요일 패턴 대용으로 근무일수 비율)
출력: 다음 달 예상 소득을 맞추기 위한 학습용 피처 테이블
"""
import calendar

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "lag1_income", "lag2_income", "lag3_income",
    "roll3_mean_income", "roll6_mean_income", "roll3_std_income",
    "workdays", "workdays_ratio",
    "n_platforms_active",
    "is_holiday_season",
    "month_sin", "month_cos",
    "months_since_start",
    "pattern_regular", "pattern_seasonal", "pattern_irregular",
]

TARGET_COLUMN = "target_next_month_income"


def build_feature_table(deposits_df: pd.DataFrame) -> pd.DataFrame:
    """worker별 월 단위 시계열에서 lag/rolling/계절 피처를 만들고,
    다음 달 소득을 타깃으로 붙인다."""
    df = deposits_df.copy()
    df = df.sort_values(["worker_id", "year", "month"]).reset_index(drop=True)
    df["ym"] = df["year"] * 12 + df["month"]

    grp = df.groupby("worker_id")

    df["lag1_income"] = grp["monthly_income"].shift(1)
    df["lag2_income"] = grp["monthly_income"].shift(2)
    df["lag3_income"] = grp["monthly_income"].shift(3)

    df["roll3_mean_income"] = grp["monthly_income"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["roll6_mean_income"] = grp["monthly_income"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).mean()
    )
    df["roll3_std_income"] = grp["monthly_income"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).std()
    ).fillna(0)

    df["days_in_month"] = df.apply(lambda r: calendar.monthrange(int(r["year"]), int(r["month"]))[1], axis=1)
    df["workdays_ratio"] = (df["workdays"] / df["days_in_month"]).clip(0, 1)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["months_since_start"] = grp.cumcount()

    for p in ("regular", "seasonal", "irregular"):
        df[f"pattern_{p}"] = (df["pattern"] == p).astype(int)

    df[TARGET_COLUMN] = grp["monthly_income"].shift(-1)
    return df


def training_table(deposits_df: pd.DataFrame) -> pd.DataFrame:
    """학습용: lag(최소 3개월 이력)와 타깃(다음 달 실적)이 모두 존재하는 행만 사용."""
    df = build_feature_table(deposits_df)
    return df.dropna(subset=["lag3_income", TARGET_COLUMN]).copy()


def latest_feature_row(deposits_df: pd.DataFrame, worker_id: int) -> pd.DataFrame:
    """대시보드/리포트 화면에서 '지금 시점 기준 다음 달 예측'에 쓸 최신 피처 1행.
    타깃(다음 달 실적)은 아직 모르므로 lag3만 존재하면 된다."""
    df = build_feature_table(deposits_df[deposits_df["worker_id"] == worker_id])
    df = df.dropna(subset=["lag3_income"])
    if df.empty:
        return df
    return df.iloc[[-1]]
