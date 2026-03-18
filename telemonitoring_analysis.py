import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from xgboost import DMatrix, XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
SUBJECT_COL = "subject#"
TIME_COL = "test_time"
TARGET_MOTOR = "motor_UPDRS"
TARGET_TOTAL = "total_UPDRS"


class TwoStageResidualRegressor(BaseEstimator, RegressorMixin):
    """Baseline + residual model with stability controls for production."""

    def __init__(
        self,
        n_estimators=260,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.3,
        reg_alpha=1e-3,
        reg_lambda=1.0,
        objective="reg:squarederror",
        early_stopping_rounds=30,
        clip_percentile_low=1.0,
        clip_percentile_high=99.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.early_stopping_rounds = early_stopping_rounds
        self.clip_percentile_low = clip_percentile_low
        self.clip_percentile_high = clip_percentile_high
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _as_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return pd.DataFrame(X)

    def _monotone_constraints(self, columns):
        constraints = [0] * len(columns)
        if "days_since_first" in columns:
            constraints[columns.index("days_since_first")] = 1
        if "baseline_pred" in columns:
            constraints[columns.index("baseline_pred")] = 1
        return "(" + ",".join(str(v) for v in constraints) + ")"

    def fit(self, X, y):
        X_df = self._as_dataframe(X)
        y_arr = np.asarray(y, dtype=float)
        self.feature_names_in_ = list(X_df.columns)

        self.target_min_ = np.percentile(y_arr, self.clip_percentile_low)
        self.target_max_ = np.percentile(y_arr, self.clip_percentile_high)

        self.baseline_model_ = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("model", LinearRegression()),
            ]
        )
        self.baseline_model_.fit(X_df, y_arr)
        baseline_pred = self.baseline_model_.predict(X_df)

        residual = y_arr - baseline_pred
        self.residual_min_ = np.percentile(residual, self.clip_percentile_low)
        self.residual_max_ = np.percentile(residual, self.clip_percentile_high)

        n_quantiles = max(10, min(1000, len(y_arr)))
        self.residual_transformer_ = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=self.random_state,
        )
        residual_t = self.residual_transformer_.fit_transform(residual.reshape(-1, 1)).ravel()

        self.residual_t_min_ = float(residual_t.min())
        self.residual_t_max_ = float(residual_t.max())

        X_stage2 = X_df.copy()
        X_stage2["baseline_pred"] = baseline_pred
        stage2_cols = list(X_stage2.columns)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_stage2,
            residual_t,
            test_size=0.15,
            random_state=self.random_state,
        )

        self.residual_model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            monotone_constraints=self._monotone_constraints(stage2_cols),
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )

        self.residual_model_.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict(self, X):
        X_df = self._as_dataframe(X)
        baseline_pred = self.baseline_model_.predict(X_df)

        X_stage2 = X_df.copy()
        X_stage2["baseline_pred"] = baseline_pred

        residual_t_pred = self.residual_model_.predict(X_stage2)
        residual_t_pred = np.clip(residual_t_pred, self.residual_t_min_, self.residual_t_max_)

        residual_pred = self.residual_transformer_.inverse_transform(
            residual_t_pred.reshape(-1, 1)
        ).ravel()
        residual_pred = np.clip(residual_pred, self.residual_min_, self.residual_max_)

        y_pred = baseline_pred + residual_pred
        y_pred = np.clip(y_pred, self.target_min_, self.target_max_)
        return y_pred


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def build_subject_meta_features(feature_df, subject_series):
    """Create subject-level acoustic meta-features from X only (no target leakage)."""
    tmp = feature_df.copy()
    tmp[SUBJECT_COL] = subject_series.values

    meta = tmp.groupby(SUBJECT_COL).agg(["mean", "std"]).fillna(0.0)
    meta.columns = [f"meta_{col}_{agg}" for col, agg in meta.columns]

    out = tmp[[SUBJECT_COL]].join(meta, on=SUBJECT_COL).drop(columns=[SUBJECT_COL])
    out.index = feature_df.index
    return out


def choose_blend_weight(y_true, pred_a, pred_b):
    best_alpha = 1.0
    best_rmse = float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        pred = alpha * pred_a + (1.0 - alpha) * pred_b
        rmse, _, _ = evaluate_regression(y_true, pred)
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
    return best_alpha


def engineer_features(df, include_target_history=False):
    df = df.sort_values([SUBJECT_COL, TIME_COL]).reset_index(drop=True)

    base_cols = [
        col
        for col in df.columns
        if col not in [SUBJECT_COL, TIME_COL, TARGET_MOTOR, TARGET_TOTAL]
    ]

    X = df[base_cols].copy()
    X["days_since_first"] = df.groupby(SUBJECT_COL)[TIME_COL].transform(lambda x: x - x.min())

    for col in base_cols:
        grouped = df.groupby(SUBJECT_COL)[col]
        X[f"{col}_delta1"] = grouped.diff().fillna(0.0)
        X[f"{col}_rollmean3"] = (
            grouped.rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    if include_target_history:
        y_group = df.groupby(SUBJECT_COL)[TARGET_MOTOR]
        prev_updrs = y_group.shift(1)
        prev2_updrs = y_group.shift(2)
        rolling_prev3 = (
            prev_updrs.groupby(df[SUBJECT_COL])
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        X["prev_updrs"] = prev_updrs.fillna(0.0)
        X["prev_updrs_change"] = (prev_updrs - prev2_updrs).fillna(0.0)
        X["rolling_mean_3"] = rolling_prev3.fillna(0.0)

    return df, X


def temporal_warm_split_indices(df_sorted, min_history=3, test_fraction=0.3):
    train_idx = []
    test_idx = []

    for _, group in df_sorted.groupby(SUBJECT_COL):
        idx = group.index.to_list()
        if len(idx) <= min_history:
            train_idx.extend(idx)
            continue

        split_point = max(min_history, int(len(idx) * (1 - test_fraction)))
        split_point = min(split_point, len(idx) - 1)

        train_idx.extend(idx[:split_point])
        test_idx.extend(idx[split_point:])

    return np.array(train_idx), np.array(test_idx)


def build_benchmark_models():
    return {
        "LinearRegression": Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "HuberRegressor": Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("model", HuberRegressor()),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "TwoStageResidualXGB": TwoStageResidualRegressor(
            objective="reg:squarederror",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }


def run_group_cv(X, y, groups, label):
    gkf = GroupKFold(n_splits=5)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows = []
    print(f"\n[GroupKFold CV Results] Target: {label}")

    for name, model in build_benchmark_models().items():
        cv_scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=gkf,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        rmse_values = -cv_scores["test_rmse"]
        mae_values = -cv_scores["test_mae"]
        r2_values = cv_scores["test_r2"]

        summary = {
            "model": name,
            "rmse_mean": rmse_values.mean(),
            "rmse_std": rmse_values.std(),
            "mae_mean": mae_values.mean(),
            "r2_mean": r2_values.mean(),
            "r2_std": r2_values.std(),
        }
        rows.append(summary)

        print(
            f"{name}: RMSE = {summary['rmse_mean']:.3f} +/- {summary['rmse_std']:.3f}, "
            f"MAE = {summary['mae_mean']:.3f}, R2 = {summary['r2_mean']:.3f}"
        )

    result_df = pd.DataFrame(rows).sort_values(by="rmse_mean", ascending=True)
    print("\nCV leaderboard:")
    print(result_df[["model", "rmse_mean", "rmse_std", "mae_mean", "r2_mean"]])
    return result_df


def tune_two_stage_optuna_cold(X, y, groups, n_trials=20):
    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        model = TwoStageResidualRegressor(
            n_estimators=trial.suggest_int("n_estimators", 120, 320),
            max_depth=trial.suggest_int("max_depth", 2, 4),
            learning_rate=trial.suggest_float("learning_rate", 0.008, 0.05, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 0.95),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.85),
            min_child_weight=trial.suggest_int("min_child_weight", 4, 12),
            gamma=trial.suggest_float("gamma", 0.2, 2.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
            objective="reg:squarederror",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )

        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=gkf,
            scoring={"rmse": "neg_root_mean_squared_error"},
            n_jobs=1,
            error_score="raise",
        )
        return (-scores["test_rmse"]).mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("\n[Optuna Tuning - TwoStageResidualXGB | cold-start aggressive]")
    print("Best RMSE:", round(study.best_value, 4))
    print("Best params:", study.best_params)
    return study.best_params


def run_native_shap(two_stage_model, X_train, title):
    try:
        sample = X_train.sample(n=min(300, len(X_train)), random_state=RANDOM_STATE)
        baseline_pred = two_stage_model.baseline_model_.predict(sample)

        X_stage2 = sample.copy()
        X_stage2["baseline_pred"] = baseline_pred

        dmat = DMatrix(X_stage2, feature_names=list(X_stage2.columns))
        shap_values = two_stage_model.residual_model_.get_booster().predict(
            dmat, pred_contribs=True
        )[:, :-1]

        ranking = (
            pd.DataFrame(
                {
                    "feature": X_stage2.columns,
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                }
            )
            .sort_values(by="mean_abs_shap", ascending=False)
            .head(12)
        )

        print(f"\n[Top SHAP Features - {title}]")
        print(ranking.to_string(index=False))
        return ranking
    except Exception as exc:
        print("\nSHAP summary skipped:", exc)
        return None


def plot_predictions(y_true, y_pred, label):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color="teal")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"Actual vs Predicted: {label}")
    plt.tight_layout()
    plt.show()


def evaluate_holdout_track(
    X,
    y,
    subject_series,
    meta_source,
    train_idx,
    test_idx,
    best_params,
    track_name,
):
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    subjects_train = subject_series.iloc[train_idx].copy()

    model = TwoStageResidualRegressor(
        **best_params,
        objective="reg:squarederror",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Secondary model on subject-level acoustic meta-features for blending.
    meta_train = build_subject_meta_features(
        meta_source.iloc[train_idx].copy(),
        subject_series.iloc[train_idx],
    )
    meta_test = build_subject_meta_features(
        meta_source.iloc[test_idx].copy(),
        subject_series.iloc[test_idx],
    )

    X_train_aug = pd.concat([X_train.reset_index(drop=True), meta_train.reset_index(drop=True)], axis=1)
    X_test_aug = pd.concat([X_test.reset_index(drop=True), meta_test.reset_index(drop=True)], axis=1)

    meta_model = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("model", HuberRegressor(alpha=0.002, epsilon=1.35, max_iter=2000)),
        ]
    )

    # Estimate blend weight on a group-safe validation split.
    inner_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    inner_train_idx, inner_val_idx = next(inner_split.split(X_train, y_train, groups=subjects_train))

    base_inner = TwoStageResidualRegressor(
        **best_params,
        objective="reg:squarederror",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    base_inner.fit(X_train.iloc[inner_train_idx], y_train.iloc[inner_train_idx])
    pred_base_val = base_inner.predict(X_train.iloc[inner_val_idx])

    meta_model.fit(X_train_aug.iloc[inner_train_idx], y_train.iloc[inner_train_idx])
    pred_meta_val = meta_model.predict(X_train_aug.iloc[inner_val_idx])

    blend_alpha = choose_blend_weight(y_train.iloc[inner_val_idx], pred_base_val, pred_meta_val)

    # Refit meta model on full training data and run blended inference.
    meta_model.fit(X_train_aug, y_train)
    pred_base = model.predict(X_test)
    pred_meta = meta_model.predict(X_test_aug)
    y_pred = blend_alpha * pred_base + (1.0 - blend_alpha) * pred_meta

    y_pred = np.clip(
        y_pred,
        np.percentile(y_train.values, 1),
        np.percentile(y_train.values, 99),
    )

    rmse, mae, r2 = evaluate_regression(y_test, y_pred)

    print(f"\n[Holdout Results - {track_name}]")
    print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
    print(f"Blend alpha (TwoStage weight): {blend_alpha:.2f}")

    errors = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": y_pred,
            "abs_error": np.abs(y_test.values - y_pred),
        },
        index=y_test.index,
    ).sort_values(by="abs_error", ascending=False)

    print(f"\n[Top 10 highest-error samples - {track_name}]")
    print(errors.head(10).to_string())

    shap_ranking = run_native_shap(model, X_train, title=track_name)

    return {
        "model": "TwoStageResidualXGB",
        "track": track_name,
        "blend_alpha": blend_alpha,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }, errors, shap_ranking


def main():
    print("\n==============================")
    print("Parkinson Telemonitoring Regression (Stabilized Production-grade)")
    print("==============================\n")

    data_path = "parkinsons/telemonitoring/parkinsons_updrs.data"
    df = pd.read_csv(data_path)

    required = {SUBJECT_COL, TIME_COL, TARGET_MOTOR, TARGET_TOTAL}
    if not required.issubset(set(df.columns)):
        print("Required columns missing. Please check input file format.")
        return

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET_MOTOR], bins=30, ax=axes[0], kde=True, color="skyblue")
    axes[0].set_title("Distribution of motor_UPDRS")
    sns.histplot(df[TARGET_TOTAL], bins=30, ax=axes[1], kde=True, color="salmon")
    axes[1].set_title("Distribution of total_UPDRS")
    plt.tight_layout()
    plt.show()

    df_sorted, X_cold = engineer_features(df, include_target_history=False)
    _, X_warm = engineer_features(df, include_target_history=True)

    # Subject-level sound meta features (target-free) for blending.
    meta_cols = [
        col
        for col in df_sorted.columns
        if col not in [SUBJECT_COL, TIME_COL, TARGET_MOTOR, TARGET_TOTAL, "age", "sex"]
    ]
    meta_source = df_sorted[meta_cols].copy()

    groups = df_sorted[SUBJECT_COL].copy()
    y_motor = df_sorted[TARGET_MOTOR].copy()
    y_total = df_sorted[TARGET_TOTAL].copy()

    # Cold-start CV (unseen subject setting).
    motor_cv = run_group_cv(X_cold, y_motor, groups, label=f"{TARGET_MOTOR} [cold]")
    total_cv = run_group_cv(X_cold, y_total, groups, label=f"{TARGET_TOTAL} [cold]")

    cold_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    cold_train_idx, cold_test_idx = next(cold_splitter.split(X_cold, y_motor, groups=groups))

    best_params = tune_two_stage_optuna_cold(
        X_cold.iloc[cold_train_idx],
        y_motor.iloc[cold_train_idx],
        groups.iloc[cold_train_idx],
        n_trials=20,
    )

    cold_metrics, cold_errors, cold_shap = evaluate_holdout_track(
        X_cold,
        y_motor,
        groups,
        meta_source,
        cold_train_idx,
        cold_test_idx,
        best_params,
        track_name="cold_start_unseen_subject",
    )

    # Warm-start split (same subject in train/test, chronological split).
    warm_train_idx, warm_test_idx = temporal_warm_split_indices(
        df_sorted,
        min_history=3,
        test_fraction=0.3,
    )

    warm_metrics, warm_errors, warm_shap = evaluate_holdout_track(
        X_warm,
        y_motor,
        groups,
        meta_source,
        warm_train_idx,
        warm_test_idx,
        best_params,
        track_name="warm_start_subject_history_available",
    )

    plot_predictions(
        y_motor.iloc[cold_test_idx],
        np.clip(
            cold_errors.sort_index()["predicted"].values,
            y_motor.min(),
            y_motor.max(),
        ),
        f"{TARGET_MOTOR} (cold)",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    motor_cv.to_csv(os.path.join(OUTPUT_DIR, "regression_cv_motor_updrs.csv"), index=False)
    total_cv.to_csv(os.path.join(OUTPUT_DIR, "regression_cv_total_updrs.csv"), index=False)

    cold_errors.head(50).to_csv(
        os.path.join(OUTPUT_DIR, "regression_top50_errors_motor_updrs_cold.csv")
    )
    warm_errors.head(50).to_csv(
        os.path.join(OUTPUT_DIR, "regression_top50_errors_motor_updrs_warm.csv")
    )

    # Backward-compatible filename keeps latest cold-start result.
    cold_errors.head(50).to_csv(
        os.path.join(OUTPUT_DIR, "regression_top50_errors_motor_updrs.csv")
    )

    if cold_shap is not None:
        cold_shap.to_csv(
            os.path.join(OUTPUT_DIR, "regression_shap_top12_motor_updrs_cold.csv"),
            index=False,
        )
    if warm_shap is not None:
        warm_shap.to_csv(
            os.path.join(OUTPUT_DIR, "regression_shap_top12_motor_updrs_warm.csv"),
            index=False,
        )

    with open(
        os.path.join(OUTPUT_DIR, "regression_best_params_motor_updrs.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best_params, f, indent=2)

    with open(
        os.path.join(OUTPUT_DIR, "regression_holdout_metrics_motor_updrs_cold.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cold_metrics, f, indent=2)

    with open(
        os.path.join(OUTPUT_DIR, "regression_holdout_metrics_motor_updrs_warm.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(warm_metrics, f, indent=2)

    # Backward-compatible filename keeps latest cold-start metrics.
    with open(
        os.path.join(OUTPUT_DIR, "regression_holdout_metrics_motor_updrs.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cold_metrics, f, indent=2)

    print("\nSaved regression artifacts to outputs/")
    print("\nTelemonitoring regression workflow completed.")


if __name__ == "__main__":
    main()
