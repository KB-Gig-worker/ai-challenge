# ai-challenge

제8회 KB Future Finance A.I. Challenge 2026 — 도깨비방망이

실제 데모 코드는 하위 폴더에 있습니다. 아래 순서로 실행하세요.

## 시작하기

```bash
# 1) 프로젝트 폴더로 이동
cd realpay

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 목데이터 생성 + 모델 학습 (최초 1회)
python scripts/run_pipeline.py

# 4) 데모 실행
streamlit run app/streamlit_app.py
```

## 설치 오류가 날 때

`pip install` 중 버전 충돌 오류가 나면, `requirements.txt`에서 버전 표기(`==1.2.3` 부분)를 지우고 패키지 이름만 남긴 뒤 다시 설치하세요.

```bash
# 예시 — 버전을 뺀 형태
# lightgbm==4.3.0   →   lightgbm
pip install -r requirements.txt
```
