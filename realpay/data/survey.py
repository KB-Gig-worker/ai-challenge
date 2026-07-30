# -*- coding: utf-8 -*-
"""
온보딩 설문 로직 + 콜드스타트 프로파일 변환 — 기획안 06. 콜드스타트 처리 / 07. 화면①.

핵심
----
* questions.json(설문 정의)을 로드하고, 응답(dict)을 mock_workers.csv와 동일한
  스키마의 "초기 프로파일"로 변환한다. → 설문 응답과 실입금 데이터가 같은 프로파일
  스키마로 수렴하므로 대시보드/모델이 동일하게 소비할 수 있다.
* streamlit 대시보드가 콜드스타트에서 쓰는 키(expected_monthly)를 그대로 포함시켜
  기존 app/streamlit_app.py 를 깨지 않는다.

사용 예
------
    from data.survey import load_questions, options_for, survey_to_profile
    qs = load_questions()
    answers = {"q1_job": "delivery_transport", "q2_worktype": "primary",
               "q3_platform_count": 2, "q4_platforms": ["쿠팡이츠", "배달의민족"],
               "q5_workdays": 24, "q6_income_band": 2500000, "q7_regularity": "seasonal"}
    profile = survey_to_profile(answers)
"""
import json
from pathlib import Path

try:
    from industry_codes import get_candidate_codes, INDUSTRY_CODES
except ImportError:
    from .industry_codes import get_candidate_codes, INDUSTRY_CODES

_HERE = Path(__file__).parent
QUESTIONS_PATH = _HERE / "survey_questions.json"

JOB_LABELS_KO = {
    "delivery_transport": "운송·배달·대리", "professional": "전문서비스",
    "simple_computer": "컴퓨터 단순작업", "care_cleaning": "가사·청소·돌봄",
    "creative": "창작활동", "it_service": "IT 서비스", "etc": "기타",
}

# 타깃 세그먼트 기준(generate_mock_data 와 동일)
TARGET_ANNUAL_MIN = 24_000_000
TARGET_ANNUAL_MAX = 48_000_000


def load_questions() -> dict:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def options_for(question_id: str, job: str = None):
    """설문 문항의 선택지 반환. q4_platforms 는 job에 따라 동적."""
    qs = load_questions()["questions"]
    q = next((x for x in qs if x["id"] == question_id), None)
    if q is None:
        return []
    if "options_by_job" in q:
        return q["options_by_job"].get(job, [])
    return q.get("options", [])


def survey_to_profile(answers: dict) -> dict:
    """설문 응답 → 콜드스타트 초기 프로파일(mock_workers 스키마 호환).

    반환 필드
    --------
    job, job_ko, work_type, platform_count, n_platforms, platforms, primary_platform,
    pattern(=intended), intended_pattern, avg_workdays,
    est_monthly_income, expected_monthly(별칭·streamlit 호환), base_monthly_income,
    home_industry_code, est_annual_revenue, is_target_segment_candidate, source, confidence
    """
    job = answers.get("q1_job", "delivery_transport")
    work_type = answers.get("q2_worktype", "primary")
    platform_count = int(answers.get("q3_platform_count", 1) or 1)
    platforms = answers.get("q4_platforms") or []
    if isinstance(platforms, str):
        platforms = [platforms]
    workdays = int(answers.get("q5_workdays", 20) or 0)
    est_monthly = int(answers.get("q6_income_band", 2_000_000) or 0)
    pattern = answers.get("q7_regularity", "regular")

    primary_platform = platforms[0] if platforms else None
    n_platforms = len(platforms) if platforms else platform_count
    home_code = get_candidate_codes(primary_platform)[0] if primary_platform else "940909"
    est_annual = est_monthly * 12
    is_target = (TARGET_ANNUAL_MIN <= est_annual <= TARGET_ANNUAL_MAX
                 and n_platforms >= 2 and work_type == "primary")

    return {
        "job": job,
        "job_ko": JOB_LABELS_KO.get(job, job),
        "work_type": work_type,
        "platform_count": platform_count,
        "n_platforms": n_platforms,
        "platforms": platforms,
        "primary_platform": primary_platform,
        "primary_income": primary_platform,      # streamlit 기존 키 별칭
        "pattern": pattern,                       # regular/seasonal/irregular
        "intended_pattern": pattern,
        "avg_workdays": workdays,
        "est_monthly_income": est_monthly,
        "expected_monthly": est_monthly,          # streamlit 대시보드 콜드스타트 호환 키
        "base_monthly_income": est_monthly,
        "home_industry_code": home_code,
        "est_annual_revenue": est_annual,
        "is_target_segment_candidate": bool(is_target),
        "source": "survey_coldstart",
        "confidence": "low",                      # 3개월 실데이터 누적 시 실데이터 우선
    }


if __name__ == "__main__":
    demo = {"q1_job": "delivery_transport", "q2_worktype": "primary",
            "q3_platform_count": 2, "q4_platforms": ["쿠팡이츠", "배달의민족"],
            "q5_workdays": 24, "q6_income_band": 2_500_000, "q7_regularity": "seasonal"}
    print(json.dumps(survey_to_profile(demo), ensure_ascii=False, indent=2))
