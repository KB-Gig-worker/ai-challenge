# -*- coding: utf-8 -*-
"""
목데이터 생성기 (v2, 2026-07-30 재구성) — 기획안 06. 데이터 계획.

기존 v1 대비 바뀐 점
--------------------
* 표본 450명(=3직군 × 3소득패턴 × 단일/다중 = 18세그먼트 × 평균 25명) × 24개월.
* 성별/연령/직종/종사형태 구성비를 2023 플랫폼 종사자 실태조사에 맞춤.
* 성별을 "직종 조건부"로 배정(운송=남성↑, 가사돌봄=여성↑) → 비현실 조합 방지.
* 직군별 소득 변동 로직을 다르게 생성(운송=월 충격형, 대리=계절형, 전문/IT/창작=
  프로젝트형, 단순작업=일감 단절형).
* 소득패턴 라벨(regular/seasonal/irregular)을 "생성 후 실제 시계열"의
  변동계수/계절진폭/연도간 반복성으로 재계산하여 부여(정직한 라벨).

하위 호환(중요)
--------------
* 출력 파일명·필수 컬럼은 v1과 동일하므로 model/features.py, train, predict,
  app/streamlit_app.py 가 수정 없이 그대로 동작한다.
    - mock_workers.csv  : worker_id, pattern, primary_platform, platforms(";"),
                          n_platforms, avg_workdays, base_monthly_income  (+ 추가컬럼)
    - mock_deposits.csv : worker_id, year, month, platform, monthly_income,
                          workdays, n_platforms_active, is_holiday_season, pattern (+ 추가컬럼)
* pattern 값은 반드시 "regular" / "seasonal" / "irregular" (features 원핫과 일치).

근거 출처
--------
[S1] 고용노동부·한국고용정보원 2023 플랫폼 종사자 실태조사
     (성별 남70.4/여29.6, 연령·종사형태·직종 구성비, 주업형 월평균 222.2만,
      운송 남87.8, 가사·청소·돌봄 여84.8)
[S3] 쿠팡이츠 주 단위 정산(영업일 3일 후)  [S4] 크몽 구매확정 후 출금  [S5] 카카오 대리 연말 성수
공식 통계(구성비)와 프로젝트 내부 정의(다중플랫폼 비율·소득패턴 임계값·변동성)는 주석으로 구분.
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from industry_codes import INDUSTRY_CODES, list_all_platforms, get_candidate_codes  # noqa
except ImportError:
    from .industry_codes import INDUSTRY_CODES, list_all_platforms, get_candidate_codes  # noqa

# =====================================================================
# 상수 / 파라미터
# =====================================================================
SEED = 20260723
N_WORKERS = 450
N_MONTHS = 24
START_YEAR = 2024
START_MONTH = 6            # v1과 동일한 시작월 유지
OUT_DIR = Path(__file__).parent

# ---- 직종 인원 (합 450). 상위3직군은 실태조사[S1], 하위 세분류는 잔여배분(project) ----
JOB_COUNTS = {
    "delivery_transport": 247,  # 운송·배달·대리
    "professional": 73,         # 전문서비스
    "simple_computer": 45,      # 컴퓨터 단순작업/데이터 라벨링
    "care_cleaning": 27,        # 가사·청소·돌봄
    "creative": 25,             # 창작
    "it_service": 21,           # IT 외주
    "etc": 12,
}
JOB_LABELS_KO = {
    "delivery_transport": "운송·배달·대리", "professional": "전문서비스",
    "simple_computer": "컴퓨터 단순작업", "care_cleaning": "가사·청소·돌봄",
    "creative": "창작활동", "it_service": "IT 서비스", "etc": "기타",
}
AGE_COUNTS = {"10s": 2, "20s": 62, "30s": 129, "40s": 121, "50s": 91, "60s": 45}  # [S1]
WORKTYPE_COUNTS = {"primary": 250, "secondary": 98, "occasional": 102}           # [S1]
# 직종 조건부 남성비율([S1]: 운송87.8, 가사돌봄 여84.8 → 남15.2; 그 외 project 가정)
JOB_MALE_RATIO = {
    "delivery_transport": 0.878, "care_cleaning": 0.152, "professional": 0.55,
    "simple_computer": 0.50, "creative": 0.50, "it_service": 0.68, "etc": 0.60,
}
# 다중 플랫폼 수 — 공식 모집단 비율 아님(다중 이용자 충분 포함용 설계표본). project.
PLATFORM_COUNT_DIST = {1: 0.50, 2: 0.35, 3: 0.15}

# 직군 → 사용 플랫폼(industry_codes에 등록된 이름만 사용해야 업종코드 추천이 동작)
JOB_PLATFORMS = {
    "delivery_transport": ["배달의민족", "쿠팡이츠", "요기요", "바로고", "생각대로",
                           "쿠팡플렉스", "로지올", "카카오T대리", "카카오모빌리티"],
    "professional": ["크몽", "숨고"],
    "simple_computer": ["크라우드웍스", "셀렉트스타", "에이모"],
    "care_cleaning": ["미소", "청소연구소", "당근알바", "대리주부"],
    "creative": ["유튜브", "트위치", "탈잉", "클래스101"],
    "it_service": ["위시켓", "프리모아", "이랜서"],
    "etc": ["쿠팡홈서비스", "당근알바"],
}
PROXY_PLATFORMS = {"카카오T대리", "카카오모빌리티"}  # 대리운전 계열

# ---- 소득 수준(원) : 종사형태 baseline([S1] 앵커) × 직군 배율(project) × 개인차 ----
BASE_MONTHLY_INCOME = {"primary": 2_222_000, "secondary": 1_100_000, "occasional": 550_000}
JOB_INCOME_MULT = {
    "delivery_transport": 1.00, "professional": 1.15, "simple_computer": 0.70,
    "care_cleaning": 0.85, "creative": 0.95, "it_service": 1.20, "etc": 0.90,
}
INDIVIDUAL_LOGNORM_SIGMA = 0.28

# ---- 계절/충격 계수 ----
DELIVERY_MONTH_COEF = {1: 0.90, 2: 0.88, 3: 1.00, 4: 1.03, 5: 1.05, 6: 0.98,
                       7: 1.18, 8: 1.15, 9: 1.02, 10: 1.05, 11: 1.08, 12: 1.20}
PROXY_MONTH_COEF = {1: 0.80, 2: 0.82, 3: 0.95, 4: 1.08, 5: 1.10, 6: 1.05,
                    7: 1.00, 8: 0.98, 9: 1.02, 10: 1.08, 11: 1.30, 12: 1.38}
MONTHLY_SHOCK = {"delivery_transport": 0.15, "professional": 0.35, "simple_computer": 0.45,
                 "care_cleaning": 0.12, "creative": 0.35, "it_service": 0.30, "etc": 0.25}

# ---- 소득패턴 prior(생성 시드) : 실제 라벨은 생성 후 재계산 ----
JOB_PATTERN_PRIOR = {
    "delivery_transport": {"regular": 0.25, "seasonal": 0.50, "irregular": 0.25},
    "professional":       {"regular": 0.15, "seasonal": 0.20, "irregular": 0.65},
    "simple_computer":    {"regular": 0.10, "seasonal": 0.15, "irregular": 0.75},
    "care_cleaning":      {"regular": 0.45, "seasonal": 0.35, "irregular": 0.20},
    "creative":           {"regular": 0.15, "seasonal": 0.25, "irregular": 0.60},
    "it_service":         {"regular": 0.30, "seasonal": 0.20, "irregular": 0.50},
    "etc":                {"regular": 0.30, "seasonal": 0.35, "irregular": 0.35},
}

# ---- 소득패턴 분류 임계값(운영상 정의, 공식 분류 아님) ----
CV_STABLE_MAX = 0.20
ZERO_STABLE_MAX = 1
SEASONAL_AMP_MIN = 0.20
SEASONAL_REPEAT_MIN = 0.35
CV_IRREGULAR_MIN = 0.45
ZERO_IRREGULAR_MIN = 3

# ---- 프로젝트형(불규칙) 파라미터 : unit_lognorm=(mu,sigma) ----
PROJECT_BASED = {
    "professional": {"lam": 2.2, "unit": (13.0, 0.6), "fee": 0.15},
    "creative":     {"lam": 1.8, "unit": (12.7, 0.7), "fee": 0.15},
    "it_service":   {"lam": 1.5, "unit": (13.6, 0.6), "fee": 0.10},
    "simple_computer": {"lam": 3.0, "unit": (11.5, 0.5), "fee": 0.0,
                        "work_prob": (0.60, 0.80), "reject": (0.05, 0.20)},
}
IRREGULAR_ZERO_MONTHS = (2, 5)

# 불규칙형 '일감 momentum'(자기상관): 바쁜/한가한 상태가 몇 달씩 이어지도록.
# rho=지속성(0=독립, 1=완전지속), sigma=월별 충격 크기. 로그-AR(1)로 배율(level) 생성.
# 이걸 넣으면 과거 이력으로 예측 가능한 구조가 생겨 불규칙형 예측오차(MAPE)가 낮아진다.
MOMENTUM_RHO = 0.78
MOMENTUM_SIGMA = 0.38
MOMENTUM_WORK_CUTOFF = 0.75  # level이 이보다 낮은 달은 '일감 없는 달'(단순작업형)

# ---- 종사형태별 평균 근무일수(월) ----
WORKDAYS_BY_WORKTYPE = {"primary": (20, 26), "secondary": (10, 18), "occasional": (4, 12)}

# ---- 타깃 세그먼트 ----
TARGET_ANNUAL_MIN = 24_000_000
TARGET_ANNUAL_MAX = 48_000_000

rng_np = np.random.default_rng(SEED)


# =====================================================================
# 유틸
# =====================================================================
def month_index_to_ym(idx):
    total = (START_MONTH - 1) + idx
    return START_YEAR + total // 12, total % 12 + 1


def _counts_to_list(d):
    out = []
    for k, n in d.items():
        out += [k] * n
    return out


# =====================================================================
# 1. 워커 프로파일
# =====================================================================
def build_profiles():
    jobs = _counts_to_list(JOB_COUNTS); assert len(jobs) == N_WORKERS
    ages = _counts_to_list(AGE_COUNTS); assert len(ages) == N_WORKERS
    wts = _counts_to_list(WORKTYPE_COUNTS); assert len(wts) == N_WORKERS
    rng_np.shuffle(jobs); rng_np.shuffle(ages); rng_np.shuffle(wts)

    profiles = []
    for i in range(N_WORKERS):
        job = jobs[i]
        gender = "male" if rng_np.random() < JOB_MALE_RATIO[job] else "female"
        pcount = int(rng_np.choice(list(PLATFORM_COUNT_DIST), p=list(PLATFORM_COUNT_DIST.values())))
        pool = JOB_PLATFORMS[job]
        k = min(pcount, len(pool))
        platforms = list(rng_np.choice(pool, size=k, replace=False))
        primary = platforms[0]
        submode = ("proxy" if primary in PROXY_PLATFORMS else "delivery") if job == "delivery_transport" else ""
        pr = JOB_PATTERN_PRIOR[job]
        intended = str(rng_np.choice(list(pr), p=list(pr.values())))
        # 개인편차: 로그정규지만 극단 꼬리는 잘라 비현실적 고소득 방지
        indiv = float(np.clip(np.exp(rng_np.normal(0, INDIVIDUAL_LOGNORM_SIGMA)), 0.5, 1.9))
        base = BASE_MONTHLY_INCOME[wts[i]] * JOB_INCOME_MULT[job] * indiv
        wd_lo, wd_hi = WORKDAYS_BY_WORKTYPE[wts[i]]
        avg_workdays = int(rng_np.integers(wd_lo, wd_hi + 1))
        profiles.append({
            "worker_id": i + 1, "job": job, "job_ko": JOB_LABELS_KO[job],
            "gender": gender, "age_band": ages[i], "work_type": wts[i],
            "platforms": platforms, "n_platforms": len(platforms), "primary_platform": primary,
            "submode": submode, "intended_pattern": intended,
            "base_monthly_income": round(base), "avg_workdays": avg_workdays,
            "home_industry_code": get_candidate_codes(primary)[0],
        })
    return profiles


# =====================================================================
# 2. 월별 소득 생성 (직군/패턴별 상이)
# =====================================================================
def _gen_regular(base, n, trend):
    std = rng_np.uniform(0.08, 0.15)
    out = []
    for idx in range(n):
        v = base * (1 + trend) ** idx * max(0.15, rng_np.normal(1.0, std))
        out.append(max(0, v))
    return np.array(out)


def _gen_seasonal(base, n, job, submode, trend):
    if job == "delivery_transport" and submode == "proxy":
        coef = PROXY_MONTH_COEF
    elif job == "delivery_transport":
        coef = DELIVERY_MONTH_COEF
    else:
        coef = {m: 1 + (DELIVERY_MONTH_COEF[m] - 1) * 0.6 for m in range(1, 13)}
    std = rng_np.uniform(0.08, 0.15)
    shock = MONTHLY_SHOCK[job]
    out = []
    for idx in range(n):
        _, m = month_index_to_ym(idx)
        v = base * coef[m] * (1 + trend) ** idx
        v *= max(0.15, rng_np.normal(1.0, std)) * (1 + rng_np.normal(0, shock * 0.4))
        out.append(max(0, v))
    return np.array(out)


def _ar1_levels(n, rho, sigma):
    """로그-AR(1) 배율(level) 시퀀스. 바쁜/한가한 상태가 지속되는 '일감 momentum'.
    level_t = exp(logl_t), logl_t = rho*logl_{t-1} + N(0, sigma). 평균≈1 부근."""
    out = np.empty(n)
    logl = 0.0
    for i in range(n):
        logl = rho * logl + rng_np.normal(0, sigma)
        out[i] = math.exp(logl)
    return out


def _gen_irregular(base, n, job):
    out = np.zeros(n)
    levels = _ar1_levels(n, MOMENTUM_RHO, MOMENTUM_SIGMA)  # 일감 momentum(자기상관)
    cfg = PROJECT_BASED.get(job)
    if cfg:
        lam, (mu, sigma), fee = cfg["lam"], cfg["unit"], cfg["fee"]
        has_work_gate = cfg.get("work_prob") is not None
        reject = cfg.get("reject")
        scale = np.clip(base / (math.exp(mu) * lam), 0.4, 3.0) if lam > 0 else 1.0
        for idx in range(n):
            lvl = levels[idx]
            # 단순작업형: momentum이 낮은 달은 '일감 없는 달'(무소득 streak)
            if has_work_gate and lvl < MOMENTUM_WORK_CUTOFF:
                continue
            lam_eff = max(0.05, lam * lvl)          # 바쁜 달일수록 프로젝트 수↑
            k = rng_np.poisson(lam_eff)
            if k == 0:
                continue
            gross = 0.0
            for _ in range(int(k)):
                unit = math.exp(rng_np.normal(mu, sigma)) * scale
                if reject:
                    unit *= (1 - rng_np.uniform(*reject))
                gross += unit
            out[idx] = gross * (1 - fee)
    else:
        # 프로젝트 파라미터 없는 직군(운송/가사 등)의 불규칙형: momentum 배율 × 소폭 잡음
        out = base * np.clip(levels * rng_np.normal(1.0, 0.25, size=n), 0.0, None)

    # 무소득월 하한 보정: momentum이 자연스러운 0-streak를 만들지만, 부족하면
    # '가장 저조한 달'부터 0 처리(랜덤 대신 → 현실적이고 자기상관 구조를 덜 해침)
    target_zero = int(rng_np.integers(IRREGULAR_ZERO_MONTHS[0], IRREGULAR_ZERO_MONTHS[1] + 1))
    cur = int((out <= 1).sum())
    if cur < target_zero:
        order = np.argsort(out)  # 오름차순: 저조한 달 먼저
        for idx in order[:target_zero - cur]:
            out[idx] = 0
    return out


# 월 소득 상한(이상치 클리핑): 로그정규 꼬리로 인한 비현실적 고액월 제거.
# 절대 상한 900만 & 개인 base의 4.5배 중 큰 값. 라벨 재계산 전에 적용해 일관성 유지.
MONTHLY_INCOME_CAP_ABS = 9_000_000


def gen_monthly(profile):
    base = profile["base_monthly_income"]
    trend = rng_np.uniform(-0.004, 0.006)
    pat = profile["intended_pattern"]
    if pat == "regular":
        monthly = _gen_regular(base, N_MONTHS, trend)
    elif pat == "seasonal":
        monthly = _gen_seasonal(base, N_MONTHS, profile["job"], profile["submode"], trend)
    else:
        monthly = _gen_irregular(base, N_MONTHS, profile["job"])
    return np.minimum(monthly, MONTHLY_INCOME_CAP_ABS)


# =====================================================================
# 3. 소득패턴 라벨 재계산(정직한 라벨)
# =====================================================================
def _amplitude(monthly):
    mean = monthly.mean()
    if mean <= 0:
        return 0.0
    moy = np.zeros(12); cnt = np.zeros(12)
    for i in range(len(monthly)):
        _, m = month_index_to_ym(i)
        moy[m - 1] += monthly[i]; cnt[m - 1] += 1
    cnt[cnt == 0] = 1
    moy /= cnt
    return float((moy.max() - moy.min()) / mean)


def _repeat_corr(monthly):
    if len(monthly) < 24:
        return 0.0
    y1, y2 = monthly[:12], monthly[12:24]
    if y1.std() == 0 or y2.std() == 0:
        return 0.0
    return float(np.corrcoef(y1, y2)[0, 1])


def classify_pattern(monthly):
    mean = monthly.mean()
    zeros = int((monthly <= 1).sum())
    cv = monthly.std() / mean if mean > 0 else 999.0
    if cv >= CV_IRREGULAR_MIN or zeros >= ZERO_IRREGULAR_MIN:
        return "irregular"
    if _amplitude(monthly) >= SEASONAL_AMP_MIN and _repeat_corr(monthly) >= SEASONAL_REPEAT_MIN:
        return "seasonal"
    if cv < CV_STABLE_MAX and zeros <= ZERO_STABLE_MAX:
        return "regular"
    return "seasonal"


# =====================================================================
# 4. 입금(월 집계) 행 생성 — v1과 동일 스키마
# =====================================================================
def gen_deposit_rows(profile, monthly, pattern):
    rows = []
    base = max(profile["base_monthly_income"], 1)
    for idx in range(N_MONTHS):
        y, m = month_index_to_ym(idx)
        income = int(max(0, round(monthly[idx], -3)))  # 천원 단위
        activity = min(monthly[idx] / base, 1.3)
        workdays = max(0, round(profile["avg_workdays"] * activity * rng_np.uniform(0.8, 1.1)))
        if income <= 1:
            workdays = 0
        n_active = profile["n_platforms"] if income > base * 0.4 else max(1, profile["n_platforms"] - 1)
        if income <= 1:
            n_active = 0
        rows.append({
            "worker_id": profile["worker_id"], "year": y, "month": m,
            "platform": profile["primary_platform"], "monthly_income": income,
            "workdays": int(workdays), "n_platforms_active": int(n_active),
            "is_holiday_season": 1 if m in (1, 2, 9, 12) else 0,  # 설/추석/연말 근사
            "pattern": pattern,
            # --- 추가 컬럼(하위호환: features는 무시) ---
            "job": profile["job"], "work_type": profile["work_type"],
        })
    return rows


# =====================================================================
# 5. 메인
# =====================================================================
def main():
    profiles = build_profiles()
    worker_rows, deposit_rows = [], []

    for p in profiles:
        monthly = gen_monthly(p)
        pattern = classify_pattern(monthly)
        deposit_rows.extend(gen_deposit_rows(p, monthly, pattern))

        y1 = int(round(monthly[:12].sum()))
        y2 = int(round(monthly[12:].sum()))
        avg_annual = round((y1 + y2) / 2)
        is_target = (TARGET_ANNUAL_MIN <= avg_annual <= TARGET_ANNUAL_MAX
                     and p["n_platforms"] >= 2 and p["work_type"] == "primary")
        worker_rows.append({
            # --- v1 필수 컬럼(순서 유지) ---
            "worker_id": p["worker_id"], "pattern": pattern,
            "primary_platform": p["primary_platform"], "platforms": ";".join(p["platforms"]),
            "n_platforms": p["n_platforms"], "avg_workdays": p["avg_workdays"],
            "base_monthly_income": p["base_monthly_income"],
            # --- 스펙 반영 추가 컬럼 ---
            "gender": p["gender"], "age_band": p["age_band"], "job": p["job"],
            "job_ko": p["job_ko"], "work_type": p["work_type"], "submode": p["submode"],
            "intended_pattern": p["intended_pattern"], "home_industry_code": p["home_industry_code"],
            "annual_revenue_y1": y1, "annual_revenue_y2": y2, "avg_annual_revenue": avg_annual,
            "is_target_segment": int(is_target),
        })

    workers_df = pd.DataFrame(worker_rows)
    deposits_df = pd.DataFrame(deposit_rows)

    # user 단위 train/valid split(참고용; train_income_model.py는 자체 GroupSplit 사용)
    ids = workers_df["worker_id"].tolist()
    rng_py = random.Random(SEED)
    rng_py.shuffle(ids)
    valid = set(ids[:int(len(ids) * 0.3)])
    workers_df["split"] = workers_df["worker_id"].map(lambda w: "valid" if w in valid else "train")

    workers_path = OUT_DIR / "mock_workers.csv"
    deposits_path = OUT_DIR / "mock_deposits.csv"
    workers_df.to_csv(workers_path, index=False, encoding="utf-8-sig")
    deposits_df.to_csv(deposits_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_workers": len(workers_df),
        "n_deposit_rows": len(deposits_df),
        "pattern_counts": workers_df["pattern"].value_counts().to_dict(),
        "avg_monthly_income": round(float(deposits_df["monthly_income"].mean()), -2),
        "income_range": [int(deposits_df["monthly_income"].min()), int(deposits_df["monthly_income"].max())],
        "seed": SEED,
        # 추가 요약
        "gender_counts": workers_df["gender"].value_counts().to_dict(),
        "age_counts": workers_df["age_band"].value_counts().to_dict(),
        "worktype_counts": workers_df["work_type"].value_counts().to_dict(),
        "job_counts": workers_df["job_ko"].value_counts().to_dict(),
        "platform_count_dist": workers_df["n_platforms"].value_counts().to_dict(),
        "target_segment_n": int(workers_df["is_target_segment"].sum()),
    }
    with open(OUT_DIR / "mock_data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] workers  -> {workers_path} ({len(workers_df)} rows)")
    print(f"[OK] deposits -> {deposits_path} ({len(deposits_df)} rows)")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
