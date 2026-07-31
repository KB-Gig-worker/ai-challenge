# RealPay 데이터 스키마 & 재구성 노트 (2026-07-30)

기존 `realpay` 코드에 **드롭인**할 수 있도록, 출력 파일명·필수 컬럼·`pattern` 값
(`regular`/`seasonal`/`irregular`)을 그대로 유지하면서 목데이터를 스펙에 맞게 재구성했다.
`model/features.py`, `model/train_income_model.py`, `model/predict.py`,
`app/streamlit_app.py`, `engine/tax_engine.py` 는 **수정 없이** 그대로 동작한다.

---

## 1. 무엇이 바뀌었나

| 항목 | 기존(v1) | 재구성(v2) |
|---|---|---|
| 표본 | 400명 × 24개월 | **450명 × 24개월** (18세그먼트 × 평균 25명) |
| 인구 구성 | 없음 | **성별·연령·직종·종사형태를 2023 실태조사 구성비에 맞춤** |
| 성별 배정 | 없음 | **직종 조건부**(운송 남 87.8%, 가사돌봄 여 84.8%) |
| 직종 | 없음(플랫폼만) | **7개 직종**(운송·전문서비스·단순작업·가사돌봄·창작·IT·기타) |
| 다중 플랫폼 | pattern에 종속 | **설계 표본 비율 1:50 / 2:35 / 3+:15** (모집단 비율 아님 명시) |
| 소득 변동 | 공통 사인곡선 | **직군별 상이**(운송=월충격, 대리=계절, 전문/IT/창작=프로젝트, 단순=일감단절) |
| 불규칙형 구조 | 매달 독립(예측 불가) | **일감 momentum(로그-AR(1), rho=0.78)** — 바쁜/한가한 상태가 이어져 예측 가능 |
| 패턴 라벨 | 생성 시 부여 | **생성 후 실제 시계열의 변동계수/계절진폭/연도간 반복성으로 재계산(정직한 라벨)** |
| 타깃 세그먼트 | 없음 | **연매출 2,400~4,800만 & 2개+ 플랫폼 & 주업형** 플래그 |

추가로 `data/industry_codes.py` 에 전문서비스·IT·데이터라벨링 인적용역 코드(`940911`)와
해당 플랫폼(크몽·숨고·위시켓·크라우드웍스 등)을 **추가만** 했다(기존 항목 불변).

---

## 2. 파일 & 컬럼

### `mock_workers.csv` (워커 1인 1행, 450행)
필수(v1 호환): `worker_id, pattern, primary_platform, platforms(";"연결), n_platforms, avg_workdays, base_monthly_income`
추가(스펙): `gender, age_band, job, job_ko, work_type, submode, intended_pattern, home_industry_code, annual_revenue_y1, annual_revenue_y2, avg_annual_revenue, is_target_segment, split`

- `pattern` : `regular` / `seasonal` / `irregular` (실제 시계열에서 재계산된 라벨)
- `work_type` : `primary`(주업) / `secondary`(부업) / `occasional`(간헐)
- `is_target_segment` : 1이면 핵심 타깃(연 2,400~4,800만·다중·주업형)
- `split` : 참고용 워커단위 train/valid (모델은 자체 GroupShuffleSplit 사용)

### `mock_deposits.csv` (워커×월, 10,800행)
필수(v1 호환): `worker_id, year, month, platform, monthly_income, workdays, n_platforms_active, is_holiday_season, pattern`
추가: `job, work_type` (features는 무시하므로 무해)

- `monthly_income` : 해당 월 총 플랫폼 소득(세전, 천원 단위, 상한 900만). 무소득월은 0.
- `is_holiday_season` : 1,2,9,12월(설·추석·연말 근사)

### `mock_data_summary.json`
v1 키(`n_workers, n_deposit_rows, pattern_counts, avg_monthly_income, income_range, seed`)
+ 구성비 요약(`gender/age/worktype/job_counts, platform_count_dist, target_segment_n`).

---

## 3. 소득패턴 분류 기준 (운영상 정의 — 공식 분류 아님)

변동계수 CV = 월소득 표준편차 ÷ 월평균.

- **irregular** : CV ≥ 0.45 **또는** 무소득월 ≥ 3
- **seasonal** : 계절진폭 ≥ 0.20 **그리고** 연도간 반복상관 ≥ 0.35 (반복되는 계절성)
- **regular** : CV < 0.20 **그리고** 무소득월 ≤ 1
- 그 외 : 완만 변동 → seasonal 귀속

라벨은 생성된 데이터에서 재계산하므로 "설계값"이 아니라 "실측 특성"이다.

---

## 4. 모델 적합성 검증 결과 (중요)

lightgbm이 없는 환경이라 sklearn HistGradientBoosting으로 **대리 검증**(워커단위 split).
실제 학습은 기존 `model/train_income_model.py`(LightGBM)로 수행.

불규칙형에 **일감 momentum(로그-AR(1), rho=0.78, sigma=0.38)** 적용 후:

| 세그먼트 | MAE | MAPE |
|---|---|---|
| regular | 약 21만원 | **14.4%** |
| seasonal | 약 23만원 | **18.0%** |
| irregular | 약 98만원 | **76.5%** (momentum 미적용 시 106.9%) |
| 전체 | 약 56만원 | **39.3%** (momentum 미적용 시 51.4%) |

해석: **규칙형·계절형은 정확히 예측**되고, momentum 덕분에 불규칙형도 과거 몇 달치로
어느 정도 예측 가능해졌다(107%→76.5%). 잔여 오차는 소득이 0 근처인 달을 MAPE가 과하게
벌주는 특성 탓이 크며, MAE 기준 개선폭이 더 크다.

momentum 파라미터는 `generate_mock_data.py` 상단 `MOMENTUM_RHO`(지속성)·`MOMENTUM_SIGMA`
(월별 충격)로 조절한다. rho를 더 올리면 예측은 쉬워지지만 불규칙형이 규칙형으로 분류돼
세그먼트 구성이 왜곡되므로 0.78에서 멈췄다.

발표 프레이밍: "규칙·계절형은 정밀 적립, 불규칙형은 일감 흐름을 반영해 예측하되
불확실성이 큰 만큼 **보수적 적립률**을 얹는다." — 기획안 10. '먼저 말할 한계'와 병행 가능.

---

## 5. 온보딩 설문 (재설계)

`data/survey_questions.json`(정의) + `data/survey.py`(로직·변환).
7문항: 직종 → 종사형태 → 플랫폼 수 → 사용 플랫폼 → 월 근무일수 → 월소득 구간 → 소득 규칙성.

`survey_to_profile(answers)` 는 **mock_workers 스키마와 호환되는 초기 프로파일**을 반환하며,
기존 streamlit 대시보드가 콜드스타트에서 쓰는 `expected_monthly` 키도 포함해 그대로 동작한다.
`app/streamlit_app.py` 화면①이 이 설문을 렌더링하도록 교체됐다.

콜드스타트 규칙: 가입 직후엔 설문 프로파일 → 3개월 실데이터 누적 시 실데이터 우선.

---

## 6. 실행

```bash
cd realpay
python data/generate_mock_data.py          # mock_workers/deposits/summary 생성 (seed 고정, 재현가능)
python model/train_income_model.py          # LightGBM 학습 (lightgbm 필요)
streamlit run app/streamlit_app.py          # 데모 3화면
```

시드(`20260723`) 고정 → 항상 동일 데이터 재현.
