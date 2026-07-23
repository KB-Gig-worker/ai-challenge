# RealPay — MVP 프로토타입

`../kb-ai-challenge-realpay.html` 기획안(제8회 KB A.I Challenge 제출용)의
04. MVP 범위를 실제로 동작하는 코드로 구현한 것.

> 긱워커의 통장 입금을 보고, 올해 낼 세금을 예측해서, 매 입금마다 자동으로 떼어놓는 AI 에이전트.

## 빠른 시작

```bash
pip install -r requirements.txt

# 1) 목데이터 생성 + 모델 학습 (최초 1회, 데이터/코드 바뀌면 재실행)
python scripts/run_pipeline.py

# 2) 데모 실행
streamlit run app/streamlit_app.py
```

LLM 리포트를 실제 Claude API로 생성하려면 실행 전 환경변수만 설정하면 된다.
(`ANTHROPIC_API_KEY` 미설정 시 자동으로 템플릿 폴백 — 데모가 API 키 유무에 흔들리지 않음)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 폴더 구조

```
realpay/
  data/
    industry_codes.py       업종코드 마스터(단순경비율 근사치) + 플랫폼→코드 매핑
    generate_mock_data.py   가상 긱워커 400명 x 24개월 입금 이력 생성기
    mock_workers.csv        (생성됨) 긱워커 프로필
    mock_deposits.csv       (생성됨) 월별 입금 이력
    mock_data_summary.json  (생성됨) 생성 데이터 요약 통계
  engine/
    tax_engine.py           세법 룰 엔진: 종합소득세 계산, 업종코드 최적화 추천,
                             입금 건별 적립액 산출
  model/
    features.py             lag/rolling/계절 피처 엔지니어링
    train_income_model.py   LightGBM 학습 (worker 단위 group split, MAE/MAPE 평가)
    predict.py               다음달/연간 소득 예측 + SHAP 근거 설명
    artifacts/               (생성됨) income_model.txt, metrics.json
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
| 01. 문제 ① (업종코드 하나로 세액 차이) | `engine/tax_engine.py::compare_industry_codes` |
| 03. 핵심 구조 (감지→분류→예측→결정→실행→검증) | 대시보드 화면이 분류/예측/결정 단계를, `compute_deposit_reserve`가 결정/실행 단계를 재현 |
| 04. MVP 범위 표의 8개 항목 | 전부 구현 (온보딩, 분류, 업종코드 추천, 소득예측, 적립액 산출, 자동적립 시뮬레이션, SHAP, LLM 리포트) |
| 05. AI는 어디에 — ① 소득예측 | `model/train_income_model.py` (LightGBM) + `model/predict.py` (SHAP) |
| 05. AI는 어디에 — ② LLM 리포트 | `report/llm_report.py` |
| 06. 데이터 계획 (Mock + 근거, 콜드스타트) | `data/generate_mock_data.py` + 대시보드의 "콜드스타트 이력 개월 수" 슬라이더 |
| 07. 화면 3개 | `app/streamlit_app.py` |

## 알려진 단순화 (발표에서 먼저 말할 부분, 10. 한계와 일치)

- **세액 계산은 근사치다.** 단순경비율 대상자만 다루고, 인적공제/세액공제는 기본공제(150만원)만
  반영한다. `INDUSTRY_CODES`의 단순경비율 수치는 국세청 실제 고시값이 아니라 데모용 근사치이며,
  실서비스 전환 시 국세청 공공데이터포털 고시자료로 교체해야 한다 (`data/industry_codes.py` 상단 주석 참고).
- **연간 소득 예측은 "다음 달 예측치 × 잔여 개월"로 단순화**했다. 실서비스에서는 월별 순차 재귀
  예측이 더 정확하지만, 12일 일정에 맞춰 단일 스텝 모델로 범위를 좁혔다 (05. "복잡한 시계열 모델 금지"와 일치).
- **자동 이체는 시뮬레이션.** 세금 금고 잔고는 계산값이며 실제 계좌 이체는 발생하지 않는다.
- **모델 평가:** worker 단위로 train/test를 분리해 리키지를 막았다. 마지막 학습 결과 기준
  MAE 약 25만원(월 평균소득 약 200만원 대비), 3개월 이동평균 베이스라인 대비 약 13.7% 개선
  (`model/artifacts/metrics.json`에서 재확인 가능).
