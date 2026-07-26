# -*- coding: utf-8 -*-
"""
모델 비교 — LightGBM을 선택한 근거를 뒷받침하기 위한 베이스라인 비교.
같은 5-fold GroupKFold(worker 단위, train_income_model.py의 교차검증과 동일 방식)로
LinearRegression / RandomForest / XGBoost / LightGBM을 공정하게 비교한다.

사용법 (realpay/ 디렉토리에서):
    python model/compare_baselines.py

결과는 model/artifacts/model_comparison.json 에 저장되고 터미널에도 표로 출력된다.
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from model.features import FEATURE_COLUMNS, TARGET_COLUMN, training_table  # noqa: E402

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

SEED = 20260723

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": SEED,
}


def load_deposits() -> pd.DataFrame:
    path = ROOT / "data" / "mock_deposits.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `python data/generate_mock_data.py` 를 실행하세요."
        )
    return pd.read_csv(path)


def _inner_val_split(X_tr, y_tr, val_frac=0.15):
    """early stopping용 내부 검증셋 분리 (train fold 안에서만)."""
    n_val = max(1, int(len(X_tr) * val_frac))
    return X_tr.iloc[:-n_val], X_tr.iloc[-n_val:], y_tr.iloc[:-n_val], y_tr.iloc[-n_val:]


def fit_predict_linear(X_tr, y_tr, X_te):
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def fit_predict_rf(X_tr, y_tr, X_te):
    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def fit_predict_xgb(X_tr, y_tr, X_te):
    X_fit, X_val, y_fit, y_val = _inner_val_split(X_tr, y_tr)
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.85, colsample_bytree=0.85,
        random_state=SEED, n_jobs=-1,
        early_stopping_rounds=30, eval_metric="mae",
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model.predict(X_te)


def fit_predict_lgb(X_tr, y_tr, X_te):
    X_fit, X_val, y_fit, y_val = _inner_val_split(X_tr, y_tr)
    train_set = lgb.Dataset(X_fit, label=y_fit)
    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        LGB_PARAMS, train_set, num_boost_round=500,
        valid_sets=[valid_set], valid_names=["valid"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model.predict(X_te, num_iteration=model.best_iteration)


def cv_mae(fit_predict_fn, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    maes = []
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        preds = np.clip(fit_predict_fn(X_tr, y_tr, X_te), 0, None)
        maes.append(float(mean_absolute_error(y_te, preds)))
    return maes


def main():
    deposits_df = load_deposits()
    table = training_table(deposits_df)
    X = table[FEATURE_COLUMNS]
    y = table[TARGET_COLUMN]
    groups = table["worker_id"]

    models = {
        "LinearRegression": fit_predict_linear,
        "RandomForest": fit_predict_rf,
        "XGBoost": fit_predict_xgb,
        "LightGBM": fit_predict_lgb,
    }

    results = {}
    print(f"{'model':16s} {'cv_mae_mean':>14s}   {'cv_mae_std':>12s}")
    for name, fn in models.items():
        maes = cv_mae(fn, X, y, groups, n_splits=5)
        mean_mae = float(np.mean(maes))
        std_mae = float(np.std(maes))
        results[name] = {
            "cv_mae_per_fold": [round(m, 0) for m in maes],
            "cv_mae_mean": round(mean_mae, 0),
            "cv_mae_std": round(std_mae, 0),
        }
        print(f"{name:16s} {mean_mae:>14,.0f} ± {std_mae:>10,.0f}")

    out_path = ARTIFACT_DIR / "model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] saved -> {out_path}")


if __name__ == "__main__":
    main()