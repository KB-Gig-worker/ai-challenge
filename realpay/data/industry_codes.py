# -*- coding: utf-8 -*-
"""
업종코드 마스터 데이터 (긱워커 / 인적용역 사업소득 중심)

주의: 아래 단순경비율(SIMPLE_EXPENSE_RATE)은 실제 국세청 고시값이 아니라
      공개된 업종별 평균 경비율 통계 및 유사 코드 사례를 참고해 만든
      "근사치(mock)"다. PPT/데모에서는 반드시
      "국세청 고시 단순경비율(홈택스)로 교체 예정"이라고 밝힐 것.
      (기획안 06. 데이터 계획 참고)

구조가 실제 서비스로 갈 때는 이 표만 국세청 공공데이터포털의
"기준경비율/단순경비율 고시자료"로 교체하면 하위 로직은 그대로 재사용된다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IndustryCode:
    code: str
    name: str
    category: str  # 업태
    simple_expense_rate: float  # 단순경비율 (0~1)
    basic_deduction: int  # 기본율 적용시 추가공제 없음(단순화), 원 단위
    platforms: tuple  # 이 코드로 흔히 신고되는 플랫폼/직군 키워드


INDUSTRY_CODES = {
    "940909": IndustryCode(
        code="940909",
        name="기타자영업(1인 미디어 포함 포괄코드)",
        category="서비스업",
        simple_expense_rate=0.641,
        basic_deduction=0,
        platforms=("기타", "분류불명"),
    ),
    "940918": IndustryCode(
        code="940918",
        name="퀵서비스배달원(배달 라이더)",
        category="운수업 관련 인적용역",
        simple_expense_rate=0.794,
        basic_deduction=0,
        platforms=("배달의민족", "쿠팡이츠", "요기요", "바로고", "생각대로"),
    ),
    "940917": IndustryCode(
        code="940917",
        name="화물자동차운송관련 인적용역(퀵/화물 기사)",
        category="운수업 관련 인적용역",
        simple_expense_rate=0.751,
        basic_deduction=0,
        platforms=("쿠팡플렉스", "로지올", "화물맨"),
    ),
    "940920": IndustryCode(
        code="940920",
        name="대리운전기사",
        category="서비스업 관련 인적용역",
        simple_expense_rate=0.771,
        basic_deduction=0,
        platforms=("카카오T대리", "카카오모빌리티"),
    ),
    "940904": IndustryCode(
        code="940904",
        name="가전제품 방문점검/설치기사",
        category="서비스업 관련 인적용역",
        simple_expense_rate=0.686,
        basic_deduction=0,
        platforms=("쿠팡홈서비스", "설치기사매칭"),
    ),
    "940306": IndustryCode(
        code="940306",
        name="크리에이터(1인 미디어 콘텐츠 창작자)",
        category="정보통신업 관련 인적용역",
        simple_expense_rate=0.649,
        basic_deduction=0,
        platforms=("유튜브", "트위치", "네이버클립"),
    ),
    "940502": IndustryCode(
        code="940502",
        name="가사도우미/가정관리사",
        category="개인서비스업",
        simple_expense_rate=0.706,
        basic_deduction=0,
        platforms=("당근알바", "미소", "청소연구소", "대리주부"),
    ),
    # --- 스펙 반영 추가(2026-07-30): 전문서비스/IT/데이터라벨링 인적용역 ---
    # 주의: 아래도 실제 국세청 고시값이 아닌 근사치(mock). 파일 상단 주의사항과 동일.
    "940911": IndustryCode(
        code="940911",
        name="기타 인적용역(프리랜서·전문서비스·IT·데이터작업)",
        category="사업서비스업 관련 인적용역",
        simple_expense_rate=0.642,
        basic_deduction=0,
        platforms=("크몽", "숨고", "위시켓", "프리모아", "이랜서",
                   "크라우드웍스", "셀렉트스타", "에이모"),
    ),
}


# 플랫폼명 -> 후보 업종코드 목록 (분류 단계에서 사용)
PLATFORM_TO_CANDIDATE_CODES = {
    "배달의민족": ["940918", "940909"],
    "쿠팡이츠": ["940918", "940909"],
    "요기요": ["940918", "940909"],
    "바로고": ["940918", "940917"],
    "생각대로": ["940918", "940917"],
    "쿠팡플렉스": ["940917", "940909"],
    "로지올": ["940917", "940909"],
    "카카오T대리": ["940920", "940909"],
    "카카오모빌리티": ["940920", "940909"],
    "쿠팡홈서비스": ["940904", "940909"],
    "유튜브": ["940306", "940909"],
    "트위치": ["940306", "940909"],
    "당근알바": ["940502", "940909"],
    "미소": ["940502", "940909"],
    "청소연구소": ["940502", "940909"],
    "대리주부": ["940502", "940909"],
    # 전문서비스 / IT / 데이터라벨링 (인적용역)
    "크몽": ["940911", "940909"],
    "숨고": ["940911", "940909"],
    "위시켓": ["940911", "940909"],
    "프리모아": ["940911", "940909"],
    "이랜서": ["940911", "940909"],
    "크라우드웍스": ["940911", "940909"],
    "셀렉트스타": ["940911", "940909"],
    "에이모": ["940911", "940909"],
    # 창작·교육 (크리에이터 코드 재사용)
    "탈잉": ["940306", "940909"],
    "클래스101": ["940306", "940909"],
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
