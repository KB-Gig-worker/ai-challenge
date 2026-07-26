# -*- coding: utf-8 -*-
"""
전체 파이프라인 한 번에 실행: 목데이터 생성 -> 모델 학습 -> 평가 결과 출력.
09. 일정의 "7/23~24 스펙 확정" 게이트 통과 여부를 한 번에 확인하는 용도.

사용법 (realpay/ 디렉토리에서):
    python scripts/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, cwd):
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    run([sys.executable, "generate_mock_data.py"], cwd=ROOT / "data")
    run([sys.executable, "train_income_model.py"], cwd=ROOT / "model")
    run([sys.executable, "compare_baselines.py"], cwd=ROOT / "model")
    print("\n[DONE] 목데이터 + 모델 아티팩트 + 베이스라인 비교 준비 완료.")
    print("데모 실행: streamlit run app/streamlit_app.py  (realpay/ 디렉토리에서)")


if __name__ == "__main__":
    main()