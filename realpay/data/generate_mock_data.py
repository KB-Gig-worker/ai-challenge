# -*- coding: utf-8 -*-
"""
목데이터 생성기 — 기획안 06. 데이터 계획.

가상 긱워커 300~500명 x 24개월 입금 이력을 생성한다.
분포 파라미터는 아래 공개 통계를 참고해 근사치로 잡았다 (실제 마이크로데이터가
아니라 "이 정도 규모/변동성이면 현실적이다"는 감을 잡기 위한 참고치):
  - 고용노동부 "플랫폼종사자 실태조사"(연 1회) — 업종별 월평균 소득, 근무일수 분포
  - 국세청 업종코드별 기준/단순경비율 고시 (공공데이터포털)

소득 유형 3가지:
  regular    : 배달/대리운전 등 근무일수 기반, 변동성 낮음
  seasonal   : 성수기/비수기 뚜렷 (예: 배달 - 여름/명절 성수기)
  irregular  : 플랫폼 다중 등록, 월별 변동성 매우 큼 (크리에이터, 프리랜서형)
"""

import json
import math
import random
from pathlib import Path

import pandas as pd

from industry_codes import list_all_platforms, get_candidate_codes, INDUSTRY_CODES

SEED = 20260723
N_WORKERS = 400
N_MONTHS = 24
OUT_DIR = Path(__file__).parent

INCOME_PATTERNS = ["regular", "seasonal", "irregular"]
PATTERN_WEIGHTS = [0.45, 0.30, 0.25]

# 업종군별 월평균 소득 기준값(원) — 고용노동부 실태조사 상 배달/대리운전 등
# 평균 구간(약 150~280만원)을 참고한 근사치
PLATFORM_BASE_INCOME = {
    "배달의민족": 2_200_000,
    "쿠팡이츠": 2_100_000,
    "요기요": 2_000_000,
    "바로고": 1_900_000,
    "생각대로": 1_850_000,
    "쿠팡플렉스": 2_400_000,
    "로지올": 2_300_000,
    "카카오T대리": 2_600_000,
    "카카오모빌리티": 2_500_000,
    "쿠팡홈서비스": 2_000_000,
    "유튜브": 1_500_000,
    "트위치": 1_100_000,
    "당근알바": 1_300_000,
    "미소": 1_400_000,
    "청소연구소": 1_450_000,
}

SEASONAL_MONTH_MULT = {
    1: 0.90, 2: 0.88, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.08,
    7: 1.15, 8: 1.18, 9: 1.05, 10: 1.02, 11: 0.98, 12: 1.10,
}


def month_index_to_ym(start_year, start_month, idx):
    total = (start_month - 1) + idx
    y = start_year + total // 12
    m = total % 12 + 1
    return y, m


def gen_worker_profile(rng, worker_id):
    pattern = rng.choices(INCOME_PATTERNS, weights=PATTERN_WEIGHTS, k=1)[0]
    n_platforms = 1 if pattern == "regular" else rng.choice([1, 2, 2, 3])
    platforms = rng.sample(list_all_platforms(), k=min(n_platforms, len(list_all_platforms())))
    base = sum(PLATFORM_BASE_INCOME[p] for p in platforms) / len(platforms)
    base *= rng.uniform(0.75, 1.35)
    avg_workdays = {
        "regular": rng.randint(20, 26),
        "seasonal": rng.randint(16, 24),
        "irregular": rng.randint(8, 20),
    }[pattern]
    return {
        "worker_id": worker_id,
        "pattern": pattern,
        "platforms": platforms,
        "n_platforms": len(platforms),
        "base_monthly_income": round(base),
        "avg_workdays": avg_workdays,
        "primary_platform": platforms[0],
    }


def gen_deposit_rows(rng, profile, start_year=2024, start_month=6, n_months=N_MONTHS):
    rows = []
    pattern = profile["pattern"]
    base = profile["base_monthly_income"]
    trend = rng.uniform(-0.004, 0.006)  # 완만한 성장/하락 추세

    noise_sd = {"regular": 0.08, "seasonal": 0.12, "irregular": 0.28}[pattern]

    for idx in range(n_months):
        y, m = month_index_to_ym(start_year, start_month, idx)
        seasonal_mult = SEASONAL_MONTH_MULT[m] if pattern in ("seasonal", "irregular") else 1.0
        trend_mult = (1 + trend) ** idx
        noise = rng.normalvariate(1.0, noise_sd)
        noise = max(0.15, noise)

        # 불규칙형은 가끔 '휴업 달'이 낀다 (플랫폼 다중 등록자의 현실 반영)
        dropout = 1.0
        if pattern == "irregular" and rng.random() < 0.08:
            dropout = rng.uniform(0.1, 0.4)

        monthly_income = base * seasonal_mult * trend_mult * noise * dropout
        monthly_income = max(0, round(monthly_income, -3))  # 천원 단위 반올림

        workdays = max(0, round(profile["avg_workdays"] * rng.uniform(0.7, 1.15) * dropout))
        n_platforms_active = profile["n_platforms"] if dropout > 0.5 else max(1, profile["n_platforms"] - 1)
        is_holiday_season = 1 if m in (1, 2, 9, 12) else 0  # 설/추석/연말 근사

        rows.append({
            "worker_id": profile["worker_id"],
            "year": y,
            "month": m,
            "platform": profile["primary_platform"],
            "monthly_income": int(monthly_income),
            "workdays": workdays,
            "n_platforms_active": n_platforms_active,
            "is_holiday_season": is_holiday_season,
            "pattern": pattern,
        })
    return rows


def main():
    rng = random.Random(SEED)

    profiles = [gen_worker_profile(rng, wid) for wid in range(1, N_WORKERS + 1)]
    deposit_rows = []
    for p in profiles:
        deposit_rows.extend(gen_deposit_rows(rng, p))

    workers_df = pd.DataFrame([{
        "worker_id": p["worker_id"],
        "pattern": p["pattern"],
        "primary_platform": p["primary_platform"],
        "platforms": ";".join(p["platforms"]),
        "n_platforms": p["n_platforms"],
        "avg_workdays": p["avg_workdays"],
        "base_monthly_income": p["base_monthly_income"],
    } for p in profiles])

    deposits_df = pd.DataFrame(deposit_rows)

    workers_path = OUT_DIR / "mock_workers.csv"
    deposits_path = OUT_DIR / "mock_deposits.csv"
    workers_df.to_csv(workers_path, index=False, encoding="utf-8-sig")
    deposits_df.to_csv(deposits_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_workers": len(workers_df),
        "n_deposit_rows": len(deposits_df),
        "pattern_counts": workers_df["pattern"].value_counts().to_dict(),
        "avg_monthly_income": round(deposits_df["monthly_income"].mean(), -2),
        "income_range": [int(deposits_df["monthly_income"].min()), int(deposits_df["monthly_income"].max())],
        "seed": SEED,
    }
    with open(OUT_DIR / "mock_data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] workers -> {workers_path} ({len(workers_df)} rows)")
    print(f"[OK] deposits -> {deposits_path} ({len(deposits_df)} rows)")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
