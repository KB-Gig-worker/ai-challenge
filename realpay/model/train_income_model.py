# -*- coding: utf-8 -*-
"""
소득 예측 모델 학습 — 기획안 05. AI는 어디에 들어가는가 ①
"12일이니 복잡한 시계열 모델 금지" -> LightGBM 회귀 모델 하나로 승부.

평가는 반드시 워커 단위(group) 분할로 한다. 같은 사람의 다른 달이 train/test에
동시에 섞이면 리키지가 생겨 성능이 뻥튀기된다 (06. 데이터 계획: 시계열 예측
성능(MAE/MAPE)으로 평가 — 클러스터 라벨 복원이 아님, 이라는 원칙과 일치).
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from model.features import FEATURE_COLUMNS, TARGET_COLUMN, training_table  # noqa: E402

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_deposits() -> pd.DataFrame:
    path = ROOT / "data" / "mock_deposits.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `python data/generate_mock_data.py` 를 실행하세요."
        )
    return pd.read_csv(path)


def main():
    deposits_df = load_deposits()
    table = training_table(deposits_df)

    X = table[FEATURE_COLUMNS]
    y = table[TARGET_COLUMN]
    groups = table["worker_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=20260723)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 20260723,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    preds = model.predict(X_test, num_iteration=model.best_iteration)
    preds = np.clip(preds, 0, None)

    # MAPE는 실제값 0 근처에서 폭주하므로, 너무 작은 실제값(1만원 미만)은 제외하고 계산
    mask = y_test > 10_000
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test[mask], preds[mask]) if mask.any() else float("nan")

    baseline_preds = X_test["roll3_mean_income"].values  # naive baseline: 최근 3개월 평균
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_mape = mean_absolute_percentage_error(y_test[mask], baseline_preds[mask]) if mask.any() else float("nan")

    # --- 교차검증: MAE가 특정 split에 우연히 좋게/나쁘게 나온 게 아닌지 확인 (worker 단위 5-fold) ---
    gkf = GroupKFold(n_splits=5)
    cv_maes = []
    cv_mapes = []
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        fold_train_set = lgb.Dataset(X_tr, label=y_tr)
        fold_valid_set = lgb.Dataset(X_te, label=y_te, reference=fold_train_set)
        fold_model = lgb.train(
            params,
            fold_train_set,
            num_boost_round=500,
            valid_sets=[fold_valid_set],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        fold_preds = np.clip(fold_model.predict(X_te, num_iteration=fold_model.best_iteration), 0, None)
        cv_maes.append(float(mean_absolute_error(y_te, fold_preds)))
        fold_mask = y_te > 10_000
        if fold_mask.any():
            cv_mapes.append(float(mean_absolute_percentage_error(y_te[fold_mask], fold_preds[fold_mask])))

    cv_mae_mean = float(np.mean(cv_maes))
    cv_mae_std = float(np.std(cv_maes))
    cv_mape_mean = float(np.mean(cv_mapes)) if cv_mapes else float("nan")
    cv_mape_std = float(np.std(cv_mapes)) if cv_mapes else float("nan")

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_iteration": int(model.best_iteration),
        "mae": round(float(mae), 0),
        "mape": round(float(mape), 4),
        "baseline_mae_roll3mean": round(float(baseline_mae), 0),
        "baseline_mape_roll3mean": round(float(baseline_mape), 4),
        "improvement_vs_baseline_pct": round(float(1 - mae / baseline_mae) * 100, 1) if baseline_mae else None,
        "cv_folds": 5,
        "cv_mae_per_fold": [round(v, 0) for v in cv_maes],
        "cv_mae_mean": round(cv_mae_mean, 0),
        "cv_mae_std": round(cv_mae_std, 0),
        "cv_mape_mean": round(cv_mape_mean, 4),
        "cv_mape_std": round(cv_mape_std, 4),
        "feature_columns": FEATURE_COLUMNS,
    }

    model_path = ARTIFACT_DIR / "income_model.txt"
    model.save_model(str(model_path))
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] model -> {model_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
