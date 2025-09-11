# Parkinson's Disease Detection with Machine Learning

## Overview
This project uses machine learning to detect Parkinson's Disease from voice measurements. The dataset is sourced from the UCI Machine Learning Repository and contains 24 features extracted from voice recordings of 195 individuals.

## Technologies Used
- Python 3
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn

## Dataset

### Additional Dataset: Telemonitoring (Regression)
- **Source:** UCI Machine Learning Repository
- **File:** `parkinsons/telemonitoring/parkinsons_updrs.data`
- **Samples:** 5,875
- **Features:** 16 voice features (+ subject info)
- **Targets:** `motor_UPDRS`, `total_UPDRS` (clinician's Parkinson's symptom scores)

## Project Steps
1. **Library Installation**
   - All required libraries are installed via pip.
2. **Data Loading**
   - The dataset is loaded using pandas.
3. **Exploratory Data Analysis (EDA)**
   - Data info, statistical summary, and class distribution are displayed.
4. **Preprocessing**
   - The `name` column is dropped.
   - Features are scaled using MinMaxScaler.
   - Class imbalance is handled with RandomOverSampler.
5. **Model Training**
   - XGBoost Classifier is trained on the resampled data.
6. **Evaluation**
   - Accuracy, classification report, AUC score, confusion matrix, and ROC curve are reported.
7. **Model Improvement**
   - Hyperparameter tuning is performed with GridSearchCV.
8. **Reporting**
   - The best model's performance is summarized and suggestions for future improvements are provided.

---

### Telemonitoring Dataset (Regression)
1. **Data Loading**
   - The telemonitoring dataset is loaded using pandas.
2. **EDA**
   - Target variable distributions are visualized.
3. **Preprocessing**
   - Subject info columns are excluded; features are scaled.
4. **Model Training**
   - Linear Regression, Random Forest, and XGBoost Regressor are trained to predict `motor_UPDRS` and `total_UPDRS`.
5. **Evaluation**
   - RMSE, MAE, and R2 metrics are reported for each model.
6. **Model Improvement**
   - Hyperparameter tuning is performed for XGBoost using GridSearchCV.
7. **Reporting**
   - Best model's results and predicted vs actual plots are provided.

## How to Run
1. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
   ```
2. Make sure the dataset is in `parkinsons/parkinsons.data`.
3. Run the main script:
   ```bash
   python parkinsons_detection.py
   ```

## Results
- **Test Accuracy:** ~0.92
- **AUC Score:** ~0.95
- The model detects Parkinson's Disease with high accuracy.


### Telemonitoring Regression Results
**Regression Results for motor_UPDRS:**
- Linear Regression: RMSE = 7.68, MAE = 6.56, R² = 0.08
- Random Forest: RMSE = 6.52, MAE = 5.17, R² = 0.33
- XGBoost: RMSE = 6.86, MAE = 5.45, R² = 0.26

**Regression Results for total_UPDRS:**
- Linear Regression: RMSE = 10.10, MAE = 8.30, R² = 0.08
- Random Forest: RMSE = 8.45, MAE = 6.53, R² = 0.36
- XGBoost: RMSE = 8.89, MAE = 6.91, R² = 0.29

**Best XGBoost (tuned) for motor_UPDRS:**
- RMSE = 6.63, MAE = 5.37, R² = 0.31
- Best params: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}

**Interpretation:**
Random Forest and XGBoost models outperform linear regression for both targets, but the R² values indicate moderate predictive power. Hyperparameter tuning improves XGBoost performance slightly. Further feature engineering or advanced models may improve results.

## Future Work
- Try different models (Random Forest, SVM)
- Feature engineering and selection
- More advanced hyperparameter optimization

- Extend regression analysis with more advanced models or feature selection
- Explore time-series or longitudinal modeling for telemonitoring data

## References
- [UCI Parkinson's Disease Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- GeeksforGeeks, TechVidvan, PythonGeeks: Parkinson's Disease ML Projects
