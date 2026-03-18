import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import DMatrix, XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"


def specificity_score(y_true, y_pred):
	tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
	return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def build_pipeline(model):
	return ImbPipeline(
		steps=[
			("scaler", MinMaxScaler()),
			("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
			("model", model),
		]
	)


def compare_models_with_cv(X, y):
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
	models = {
		"LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
		"RandomForest": RandomForestClassifier(
			n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
		),
		"XGBoost": XGBClassifier(
			n_estimators=250,
			max_depth=4,
			learning_rate=0.1,
			subsample=0.9,
			colsample_bytree=0.9,
			eval_metric="logloss",
			random_state=RANDOM_STATE,
			n_jobs=-1,
		),
	}

	scoring = {
		"accuracy": "accuracy",
		"roc_auc": "roc_auc",
		"precision": "precision",
		"recall": "recall",
		"f1": "f1",
	}

	rows = []
	print("\n[Stratified 5-Fold CV Comparison]")
	for name, model in models.items():
		pipeline = build_pipeline(model)
		scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
		summary = {
			"model": name,
			"roc_auc_mean": scores["test_roc_auc"].mean(),
			"roc_auc_std": scores["test_roc_auc"].std(),
			"accuracy_mean": scores["test_accuracy"].mean(),
			"recall_mean": scores["test_recall"].mean(),
			"precision_mean": scores["test_precision"].mean(),
			"f1_mean": scores["test_f1"].mean(),
		}
		rows.append(summary)
		print(
			f"{name}: ROC-AUC = {summary['roc_auc_mean']:.3f} +/- {summary['roc_auc_std']:.3f}, "
			f"Accuracy = {summary['accuracy_mean']:.3f}, Recall = {summary['recall_mean']:.3f}"
		)

	results_df = pd.DataFrame(rows).sort_values(by="roc_auc_mean", ascending=False)
	print("\nCV leaderboard:")
	print(results_df[["model", "roc_auc_mean", "roc_auc_std", "accuracy_mean", "f1_mean"]])
	return results_df


def tune_xgboost_with_optuna(X, y, n_trials=25):
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

	def objective(trial):
		params = {
			"n_estimators": trial.suggest_int("n_estimators", 120, 450),
			"max_depth": trial.suggest_int("max_depth", 3, 8),
			"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
			"subsample": trial.suggest_float("subsample", 0.6, 1.0),
			"colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
			"min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
			"gamma": trial.suggest_float("gamma", 0.0, 2.0),
			"reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
			"reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
			"eval_metric": "logloss",
			"random_state": RANDOM_STATE,
			"n_jobs": -1,
		}
		model = XGBClassifier(**params)
		pipeline = build_pipeline(model)
		cv_results = cross_validate(
			pipeline,
			X,
			y,
			cv=cv,
			scoring={"roc_auc": "roc_auc"},
			n_jobs=-1,
		)
		return cv_results["test_roc_auc"].mean()

	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
	print("\n[Optuna Tuning]")
	print("Best ROC-AUC:", round(study.best_value, 4))
	print("Best params:", study.best_params)
	return study.best_params


def report_test_metrics(model, X_test, y_test):
	y_prob = model.predict_proba(X_test)[:, 1]
	y_pred = (y_prob >= 0.5).astype(int)

	metrics = {
		"accuracy": accuracy_score(y_test, y_pred),
		"roc_auc": roc_auc_score(y_test, y_prob),
		"precision": precision_score(y_test, y_pred),
		"recall_sensitivity": recall_score(y_test, y_pred),
		"f1": f1_score(y_test, y_pred),
		"specificity": specificity_score(y_test, y_pred),
	}

	print("\n[Test Metrics]")
	for key, value in metrics.items():
		print(f"{key}: {value:.4f}")

	print("\nClassification Report:")
	print(classification_report(y_test, y_pred, digits=4))

	conf_matrix = confusion_matrix(y_test, y_pred)
	plt.figure(figsize=(6, 4))
	sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
	plt.title("Confusion Matrix (Test)")
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.tight_layout()
	plt.show()

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=8, ax=axes[0])
	axes[0].set_title("Calibration Curve")

	fpr, tpr, _ = roc_curve(y_test, y_prob)
	axes[1].plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
	axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
	axes[1].set_xlabel("False Positive Rate")
	axes[1].set_ylabel("True Positive Rate")
	axes[1].set_title("ROC Curve")
	axes[1].legend(loc="lower right")
	plt.tight_layout()
	plt.show()

	# Show uncertain wrong predictions for targeted error analysis.
	analysis_df = X_test.copy()
	analysis_df["y_true"] = y_test.values
	analysis_df["y_pred"] = y_pred
	analysis_df["positive_probability"] = y_prob
	analysis_df["uncertainty"] = np.abs(y_prob - 0.5)
	misclassified = analysis_df[analysis_df["y_true"] != analysis_df["y_pred"]]
	misclassified = misclassified.sort_values(by="uncertainty", ascending=True)

	print("\n[Top 5 uncertain misclassified samples]")
	if misclassified.empty:
		print("No misclassifications on the test split.")
	else:
		print(
			misclassified[["y_true", "y_pred", "positive_probability", "uncertainty"]]
			.head(5)
			.to_string()
		)

	return metrics, misclassified


def run_shap_summary(best_pipeline, X_train):
	try:
		scaler = best_pipeline.named_steps["scaler"]
		model = best_pipeline.named_steps["model"]
		sample = X_train.sample(n=min(250, len(X_train)), random_state=RANDOM_STATE)
		sample_scaled = scaler.transform(sample)
		dmat = DMatrix(sample_scaled, feature_names=list(sample.columns))
		# XGBoost native TreeSHAP contributions; last column is the bias term.
		shap_values = model.get_booster().predict(dmat, pred_contribs=True)[:, :-1]
		mean_abs_shap = np.abs(shap_values).mean(axis=0)

		ranking = (
			pd.DataFrame(
				{
					"feature": sample.columns,
					"mean_abs_shap": mean_abs_shap,
				}
			)
			.sort_values(by="mean_abs_shap", ascending=False)
			.head(10)
		)
		print("\n[Top 10 SHAP Features]")
		print(ranking.to_string(index=False))
		return ranking
	except Exception as exc:
		print("\nSHAP summary skipped:", exc)
		return None


def main():
	df = pd.read_csv("parkinsons/parkinsons.data")
	print("First 5 rows:")
	print(df.head())
	print("\nShape:", df.shape)
	print("\nClass distribution:")
	print(df["status"].value_counts())

	X = df.drop(["name", "status"], axis=1)
	y = df["status"].astype(int)

	cv_results = compare_models_with_cv(X, y)
	best_params = tune_xgboost_with_optuna(X, y, n_trials=25)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=RANDOM_STATE,
		stratify=y,
	)

	tuned_model = XGBClassifier(
		**best_params,
		eval_metric="logloss",
		random_state=RANDOM_STATE,
		n_jobs=-1,
	)
	best_pipeline = build_pipeline(tuned_model)
	best_pipeline.fit(X_train, y_train)

	metrics, misclassified = report_test_metrics(best_pipeline, X_test, y_test)
	shap_ranking = run_shap_summary(best_pipeline, X_train)

	os.makedirs(OUTPUT_DIR, exist_ok=True)
	cv_results.to_csv(os.path.join(OUTPUT_DIR, "classification_cv_results.csv"), index=False)
	misclassified.head(25).to_csv(
		os.path.join(OUTPUT_DIR, "classification_misclassified_top25.csv"),
		index=False,
	)
	if shap_ranking is not None:
		shap_ranking.to_csv(
			os.path.join(OUTPUT_DIR, "classification_shap_top10.csv"),
			index=False,
		)
	with open(os.path.join(OUTPUT_DIR, "classification_best_params.json"), "w", encoding="utf-8") as f:
		json.dump(best_params, f, indent=2)
	with open(os.path.join(OUTPUT_DIR, "classification_test_metrics.json"), "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	print("\nSaved classification artifacts to outputs/")

	print("\nClassification workflow completed.")


if __name__ == "__main__":
	main()
