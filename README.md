# Parkinson's Disease Detection and Telemonitoring Analysis

## Overview
This project contains two machine learning workflows built on UCI Parkinson datasets:

1. **Classification** (`parkinsons_detection.py`)
   - Detects Parkinson's disease status from voice features.
2. **Regression** (`telemonitoring_analysis.py`)
   - Predicts clinical symptom scores (`motor_UPDRS`, `total_UPDRS`) from telemonitoring voice signals.

The codebase is updated for stronger methodological validity, better reproducibility, and richer model diagnostics.

## Key Improvements Implemented
- **Data leakage fixes**
  - Scaling and sampling are applied inside model pipelines.
- **Imbalance-safe model validation**
  - Classification uses `StratifiedKFold` with `RandomOverSampler` inside an `imblearn` pipeline.
- **Patient-level robust evaluation**
  - Telemonitoring regression uses `GroupShuffleSplit` and `GroupKFold` with `subject#` to avoid patient leakage.
- **Expanded model benchmarking**
  - Classification: Logistic Regression, Random Forest, XGBoost
  - Regression: Linear Regression, Random Forest, XGBoost, HistGradientBoosting, optional CatBoost
- **Advanced tuning**
  - Optuna-based hyperparameter optimization for XGBoost.
- **Richer diagnostics**
  - Sensitivity/specificity, calibration curve, ROC curve, error analysis.
  - SHAP-based feature importance summary (if SHAP is available).

## Why These Changes Were Necessary
This repository is intended to answer a clinically meaningful question:

**"Can we estimate Parkinson symptom severity for a new patient?"**

To make the evaluation trustworthy for this use-case, the workflow was upgraded as follows:

1. **Subject-safe validation**
  - Same patient should not leak between train and test in cold-start evaluation.
2. **Two deployment scenarios**
  - **Cold-start:** brand-new patient (no prior target history).
  - **Warm-start:** existing patient with prior follow-up history.
3. **Stability and robustness**
  - Robust objectives, clipping/capping, early stopping, and monotonic constraints were added to avoid unstable predictions.
4. **Explainability and diagnostics**
  - SHAP and detailed error analysis are exported for model behavior auditing.

## Technologies Used
- Python 3
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- optuna
- shap
- catboost (optional)

## Datasets

### 1) Classification Dataset
- **Source:** UCI Machine Learning Repository
- **File:** `parkinsons/parkinsons.data`
- **Samples:** 195
- **Features:** 22 voice features + metadata
- **Target:** `status` (0: healthy, 1: Parkinson)

### 2) Telemonitoring Dataset (Regression)
- **Source:** UCI Machine Learning Repository
- **File:** `parkinsons/telemonitoring/parkinsons_updrs.data`
- **Samples:** 5,875
- **Targets:** `motor_UPDRS`, `total_UPDRS`
- **Grouping variable:** `subject#`

## Installation
Install all dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

### Classification Pipeline
```bash
python parkinsons_detection.py
```

Expected outputs:
- 5-fold stratified CV model comparison
- Optuna best hyperparameters for XGBoost
- Test metrics (Accuracy, ROC-AUC, Precision, Recall/Sensitivity, F1, Specificity)
- Confusion matrix, ROC curve, calibration curve
- Top uncertain misclassified samples
- SHAP feature ranking (if available)
- Saved artifacts in `outputs/`:
  - `classification_cv_results.csv`
  - `classification_best_params.json`
  - `classification_test_metrics.json`
  - `classification_misclassified_top25.csv`

### Telemonitoring Regression Pipeline
```bash
python telemonitoring_analysis.py
```

Expected outputs:
- Group-aware CV results (mean +/- std) for both targets
- Cold-start tuning for `motor_UPDRS` with aggressive regularization
- Separate holdout metrics for:
  - `cold_start_unseen_subject`
  - `warm_start_subject_history_available`
- Actual vs predicted plot
- Top highest-error samples
- SHAP feature ranking (if available)
- Saved artifacts in `outputs/`:
  - `regression_cv_motor_updrs.csv`
  - `regression_cv_total_updrs.csv`
  - `regression_best_params_motor_updrs.json`
  - `regression_holdout_metrics_motor_updrs.json`
  - `regression_holdout_metrics_motor_updrs_cold.json`
  - `regression_holdout_metrics_motor_updrs_warm.json`
  - `regression_top50_errors_motor_updrs.csv`
  - `regression_top50_errors_motor_updrs_cold.csv`
  - `regression_top50_errors_motor_updrs_warm.csv`
  - `regression_shap_top12_motor_updrs_cold.csv`
  - `regression_shap_top12_motor_updrs_warm.csv`

## Results Comparison

### Classification (latest run)
- Accuracy: **0.9231**
- ROC-AUC: **0.9828**
- Precision: **0.9643**
- Recall (Sensitivity): **0.9310**
- F1: **0.9474**
- Specificity: **0.9000**

### Telemonitoring Regression (motor_UPDRS)

| Setup | RMSE | MAE | R2 | Notes |
|---|---:|---:|---:|---|
| Earlier single holdout pipeline | 8.040 | 6.991 | -0.109 | Group-aware but single-track reporting |
| **Current cold-start (unseen subject)** | **10.125** | **8.901** | **-0.758** | Harder and more realistic new-patient scenario |
| **Current warm-start (history available)** | **0.772** | **0.642** | **0.991** | Follow-up scenario with prior patient trajectory |

### Interpretation
- The **cold-start** score is lower than warm-start because predicting symptom severity for completely unseen subjects is significantly harder.
- The **warm-start** score is very strong, indicating the model is effective when longitudinal patient history exists.
- Reporting both tracks provides a production-realistic view instead of one mixed score.

## Reproducibility Notes
- Random seeds are fixed (`random_state=42`) across workflows.
- Group-aware splitting is used for telemonitoring to avoid subject leakage.
- Pipelines are used to ensure transformations are learned only from training data.

## Future Directions
- Time-aware/longitudinal modeling for telemonitoring trajectories.
- Probability threshold optimization for classification according to clinical cost.
- Feature selection and interaction modeling.
- External validation on independent cohorts.

## References
- [UCI Parkinson's Disease Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- [UCI Parkinson Telemonitoring Data Set](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)
