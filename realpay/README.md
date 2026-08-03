# RealPay — MVP 프로토타입

`../kb-ai-challenge-realpay.html` 기획안(제8회 KB A.I Challenge 제출용)의
04. MVP 범위를 실제로 동작하는 코드로 구현한 것.

> 가상 소득 내역으로 다음 달 소득과 세금 대비 참고 금액을 살펴보는 개념검증 데모.

> [!IMPORTANT]
> 이 프로젝트는 목데이터를 사용하는 시뮬레이션입니다. 실제 계좌를 연결·조회하지 않고,
> 돈을 이체하거나 보관하지 않으며, 세무 신고·세무 자문을 제공하지 않습니다.

## 구현 범위

- 가상 긱워커 데이터와 온보딩 설문을 이용한 소득 추정
- 다음 달 소득 점추정치와 단순 연환산 값 표시
- 단순화된 세금 계산에 따른 추가 납부 대비 참고액 계산
- 플랫폼에 연결된 업종코드 **검토 후보** 표시
- 모델 입력 요인의 SHAP 기여도와 선택적 LLM 요약

구현하지 않은 기능:

- 오픈뱅킹·은행 API 연결, 실제 입금 감지, 계좌 잔액 조회
- 자동이체, 자금 보관, 출금·취소·실패 처리
- 본인인증 및 실사용자 금융데이터 처리
- 신고용 업종코드 결정, 세무 신고 대행 또는 개인별 세무 자문

## 빠른 시작

```bash
pip install -r requirements.txt

# 1) 목데이터 생성 + 모델 학습 (최초 1회, 데이터/코드 바뀌면 재실행)
python scripts/run_pipeline.py

# 2) 데모 실행
streamlit run app/streamlit_app.py
```

`requirements.txt`의 버전은 Python 3.9~3.11 기준으로 고정돼 있다. **1단계 설치가 실패하면**
(Python 3.12 이상에서는 `numpy 1.24.4` / `pandas 2.0.3` 휠이 없어 실패한다)
`requirements.txt`에서 `== 버전` 부분을 지우고 패키지 이름만 남긴 뒤 다시 설치하면 된다.
자세한 내용은 파일 상단 주석에 적어두었다.

LLM 리포트를 실제 Claude API로 생성하려면 실행 전 환경변수를 설정하고, 화면에서 외부 전송에
명시적으로 동의해야 한다. 기본값은 외부 전송이 없는 로컬 템플릿이다.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 폴더 구조

```
realpay/
  data/
    industry_codes.py       업종코드 마스터(단순경비율 근사치) + 플랫폼→코드 매핑
    generate_mock_data.py   가상 긱워커 450명 x 24개월 입금 이력 생성기
    survey.py               온보딩 설문 응답 → 프로파일/유사 워커 추정
    survey_questions.json   온보딩 설문 문항 정의
    DATA_SCHEMA.md          목데이터 컬럼 정의와 생성 근거
    mock_workers.csv        (생성됨) 긱워커 프로필
    mock_deposits.csv       (생성됨) 월별 입금 이력
    mock_data_summary.json  (생성됨) 생성 데이터 요약 통계
  engine/
    tax_engine.py           데모 세액 근사 계산, 업종코드 검토 후보,
                             추가 납부 대비 참고액 산출
    stability.py            소득 안정성 자가진단 (최소 6개월 이력 필요)
  model/
    features.py             lag/rolling/계절 피처 엔지니어링
    train_income_model.py   LightGBM 학습 (worker 단위 group split, MAE/MAPE 평가)
    compare_baselines.py    LinearRegression/RandomForest/XGBoost/LightGBM 5-fold 비교
    predict.py               다음달/연간 소득 예측 + SHAP 근거 설명
    artifacts/               (생성됨) income_model.txt, metrics.json,
                             model_comparison.json
  report/
    llm_report.py            수치 → 자연어 리포트 (Claude API 또는 템플릿 폴백)
  app/
    streamlit_app.py         3개 화면: 온보딩 설문 / 대시보드 / 리포트
  scripts/
    run_pipeline.py           데이터 생성 + 학습을 한 번에 실행
```

## 기획안 대비 구현 매핑

| 기획안 섹션 | 구현 |
|---|---|
| 업종별 계산 차이 탐색 | `engine/tax_engine.py::compare_industry_codes` (신고 코드 추천 아님) |
| 소득 추정과 대비액 계산 | 대시보드가 분류·예측·참고 계산을 시뮬레이션. 감지·이체·실행은 미구현 |
| MVP 화면 | 온보딩, 소득 점추정, 대비 참고액, 업종코드 후보, SHAP, 선택적 LLM 요약 |
| 05. AI는 어디에 — ① 소득예측 | `model/train_income_model.py` (LightGBM) + `model/predict.py` (SHAP) |
| 05. AI는 어디에 — ② LLM 리포트 | `report/llm_report.py` |
| 06. 데이터 계획 (Mock + 근거, 콜드스타트) | `data/generate_mock_data.py` + 대시보드의 "콜드스타트 이력 개월 수" 슬라이더 |
| 07. 화면 3개 | `app/streamlit_app.py` |

## 중요한 한계와 안전 가정

- **세액 계산은 참고용 근사치다.** 2,400만원 단일 데모 기준과 기본공제 등 제한된 항목만
  반영한다. 실제 단순·기준경비율 판정과 세액은 업종, 귀속연도, 직전연도 수입, 신규사업자 여부,
  다른 소득·공제 및 증빙에 따라 달라진다. 신고 전 홈택스 자료 또는 세무전문가 확인이 필요하다.
- **업종코드는 세액순으로 추천하지 않는다.** 플랫폼 매핑의 대표 후보와 추가 검토 후보만 보여준다.
  실제 신고 코드는 수행한 업무의 사실관계와 지급명세서 등으로 결정해야 한다.
- **연간 소득 예측은 "다음 달 예측치 × 12"로 단순화**했다(`model/predict.py::predict_annual`,
  `months_remaining` 인자는 있으나 호출부는 모두 기본값 12를 사용). 실서비스에서는 월별 순차 재귀
  예측이 필요하다. 현재 UI의 값은 올해 누적 실적을 반영한 연간 예측이 아니라 단순 연환산 값이다.
- **추가 납부 대비액은 3.3% 원천징수가 이미 이루어졌다고 가정**하고 부족 예상분만 계산한다.
  실제 거래에서는 지급명세서와 원천징수 여부를 입금별로 확인해야 한다.
- **모든 계좌·금고·적립 표시는 가상 계산값이다.** 실제 자금 이동이나 보관은 발생하지 않는다.
- **모델은 합성 목데이터로만 평가했다.** 현재 저장된 5-fold 교차검증 결과는 MAE 약 59.1만원,
  MAPE 약 39.0%이며 실제 긱워커 데이터에서의 성능을 입증하지 않는다
  (`model/artifacts/metrics.json`). SHAP은 모델 내부 기여도이지 인과관계나 정확성 보증이 아니다.
- **LightGBM이 대안 모델 대비 뚜렷하게 우월하지는 않다.** `model/compare_baselines.py`의
  동일 조건 비교에서 RandomForest가 LightGBM과 오차 범위 내에서 비슷하거나 약간 앞선다
  (`model/artifacts/model_comparison.json`). LightGBM 채택은 성능 우위가 아니라 SHAP 연동과
  학습 속도를 고려한 선택이다.

## 개인정보 및 외부 LLM

`ANTHROPIC_API_KEY`가 설정되면 리포트용 재무 추정 컨텍스트가 외부 API로 전송될 수 있다.
현재 저장소는 목데이터만 사용하지만, 실서비스에서는 이 기능을 기본 OFF로 두고 전송 항목·목적·
제공업체·보관정책을 고지한 뒤 명시적 동의를 받아야 한다. 사용자·계좌 식별자는 전송하지 않아야 한다.
