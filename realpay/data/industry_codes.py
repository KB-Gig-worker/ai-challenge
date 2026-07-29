# -*- coding: utf-8 -*-
"""
업종코드 마스터 데이터 (긱워커 / 인적용역 사업소득 중심)
2025년 귀속(2026년 신고분) 국세청 홈택스 "기준(단순)경비율 조회"에서
직접 확인한 실제 고시 수치로 교체 완료 (2026-07-29 검증).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class IndustryCode:
    code: str
    name: str
    category: str  # 업태
    simple_expense_rate: float  # 단순경비율(일반율) (0~1)
    standard_expense_rate: float  # 기준경비율(일반율) (0~1), 2,400만원 초과 시 참고용
    basic_deduction: int
    platforms: tuple


INDUSTRY_CODES = {
    "940909": IndustryCode(
        code="940909",
        name="기타자영업(1인 미디어 포함 포괄코드)",
        category="서비스업",
        simple_expense_rate=0.641,
        standard_expense_rate=0.174,
        basic_deduction=0,
        platforms=("기타", "분류불명"),
    ),
    "940918": IndustryCode(
        code="940918",
        name="퀵서비스배달원(배달 라이더)",
        category="운수업 관련 인적용역",
        simple_expense_rate=0.794,
        standard_expense_rate=0.153,
        basic_deduction=0,
        platforms=("배달의민족", "쿠팡이츠", "요기요", "바로고", "생각대로"),
    ),
    "940919": IndustryCode(
        code="940919",
        name="기타물품운반원(택배/화물 기사)",
        category="운수업 관련 인적용역",
        simple_expense_rate=0.742,
        standard_expense_rate=0.276,
        basic_deduction=0,
        platforms=("쿠팡플렉스", "로지올", "화물맨"),
    ),
    "940913": IndustryCode(
        code="940913",
        name="대리운전기사",
        category="서비스업 관련 인적용역",
        simple_expense_rate=0.737,
        standard_expense_rate=0.322,
        basic_deduction=0,
        platforms=("카카오T대리", "카카오모빌리티"),
    ),
    "940922": IndustryCode(
        code="940922",
        name="대여제품 방문점검원(가전 방문점검/설치 유사업)",
        category="서비스업 관련 인적용역",
        simple_expense_rate=0.750,
        standard_expense_rate=0.299,
        basic_deduction=0,
        platforms=("쿠팡홈서비스", "설치기사매칭"),
    ),
    "940306": IndustryCode(
        code="940306",
        name="크리에이터(1인 미디어 콘텐츠 창작자)",
        category="정보통신업 관련 인적용역",
        simple_expense_rate=0.641,
        standard_expense_rate=0.121,
        basic_deduction=0,
        platforms=("유튜브", "트위치", "네이버클립"),
    ),
    "950001": IndustryCode(
        code="950001",
        name="가사도우미/가정관리사(가구 내 고용활동)",
        category="가구 내 고용활동",
        simple_expense_rate=0.797,
        standard_expense_rate=0.313,
        basic_deduction=0,
        platforms=("당근알바", "미소", "청소연구소"),
    ),
}

# 플랫폼명 -> 후보 업종코드 목록 (분류 단계에서 사용)
PLATFORM_TO_CANDIDATE_CODES = {
    "배달의민족": ["940918", "940909"],
    "쿠팡이츠": ["940918", "940909"],
    "요기요": ["940918", "940909"],
    "바로고": ["940918", "940919"],
    "생각대로": ["940918", "940919"],
    "쿠팡플렉스": ["940919", "940909"],
    "로지올": ["940919", "940909"],
    "카카오T대리": ["940913", "940909"],
    "카카오모빌리티": ["940913", "940909"],
    "쿠팡홈서비스": ["940922", "940909"],
    "유튜브": ["940306", "940909"],
    "트위치": ["940306", "940909"],
    "당근알바": ["950001", "940909"],
    "미소": ["950001", "940909"],
    "청소연구소": ["950001", "940909"],
}


def get_candidate_codes(platform: str) -> list:
    """플랫폼명으로 후보 업종코드(신고 가능 코드) 목록을 반환. 항상 기타자영업(940909)을
    포함시켜 '분류 불명 시 기본값' 역할을 하도록 한다."""
    codes = PLATFORM_TO_CANDIDATE_CODES.get(platform, [])
    if "940909" not in codes:
        codes = list(codes) + ["940909"]
    return codes


def list_all_platforms():
    return list(PLATFORM_TO_CANDIDATE_CODES.keys())