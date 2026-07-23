# -*- coding: utf-8 -*-
"""
예측 + SHAP 근거 — 기획안 05. ①, 04. MVP 범위 'XAI 근거 표시'.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.features import FEATURE_COLUMNS  # noqa: E402

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

FEATURE_LABEL_KO = {
    "lag1_income": "지난달 소득",
    "lag2_income": "2개월 전 소득",
    "lag3_income": "3개월 전 소득",
    "roll3_mean_income": "최근 3개월 평균",
    "roll6_mean_income": "최근 6개월 평균",
    "roll3_std_income": "최근 3개월 변동성",
    "workdays": "근무일수",
    "workdays_ratio": "근무일 비율",
    "n_platforms_active": "활성 플랫폼 수",
    "is_holiday_season": "명절/연말 성수기 여부",
    "month_sin": "계절성(월, sin)",
    "month_cos": "계절성(월, cos)",
    "months_since_start": "활동 개월 수",
    "pattern_regular": "규칙형 소득 패턴",
    "pattern_seasonal": "계절형 소득 패턴",
    "pattern_irregular": "불규칙형 소득 패턴",
}


class IncomePredictor:
    def __init__(self, model_path: Path = None):
        model_path = model_path or (ARTIFACT_DIR / "income_model.txt")
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} 가 없습니다. 먼저 `python model/train_income_model.py` 를 실행하세요."
            )
        self.model = lgb.Booster(model_file=str(model_path))
        self.explainer = shap.TreeExplainer(self.model)

    def predict_next_month(self, feature_row) -> float:
        X = feature_row[FEATURE_COLUMNS]
        pred = self.model.predict(X)[0]
        return float(max(0, pred))

    def predict_annual(self, feature_row, months_remaining: int = 12) -> float:
        """다음 달 예측치를 향후 남은 개월에 단순 반복 적용해 연간 누적치를 추정한다.
        (실제 서비스에서는 월별로 순차 재귀 예측하지만, MVP 데모에서는 단일 스텝
        예측 x 개월수로 단순화 — 06. 데이터 계획의 '평가 방식'과 별개로 데모용 근사)"""
        next_month = self.predict_next_month(feature_row)
        return next_month * months_remaining

    def explain(self, feature_row, top_k: int = 5) -> list:
        """SHAP 기여도를 한국어 설명 문장으로 변환한다.
        예: '비수기 -18%', '플랫폼 정산주기 변동 -7%' 같은 리포트 문구의 소스."""
        X = feature_row[FEATURE_COLUMNS]
        shap_values = self.explainer.shap_values(X)
        base_value = self.explainer.expected_value
        prediction = self.model.predict(X)[0]

        contributions = []
        for i, col in enumerate(FEATURE_COLUMNS):
            val = float(shap_values[0][i])
            pct_of_pred = (val / prediction * 100) if prediction else 0
            contributions.append({
                "feature": col,
                "label": FEATURE_LABEL_KO.get(col, col),
                "value": float(X.iloc[0][col]),
                "shap": val,
                "pct_of_prediction": round(pct_of_pred, 1),
            })

        contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)
        return {
            "base_value": round(float(base_value), 0),
            "prediction": round(float(prediction), 0),
            "top_factors": contributions[:top_k],
        }
