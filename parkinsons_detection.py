import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
df = pd.read_csv('parkinsons/parkinsons.data')
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 2. Exploratory Data Analysis (EDA)
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nTarget variable distribution (status):")
print(df['status'].value_counts())

# 3. Feature Selection and Preprocessing
X = df.drop(['name', 'status'], axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)

# 4. Model Training (XGBoost Classifier)
model = XGBClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 5. Model Evaluation
y_pred = model.predict(X_test_scaled)
print("\n[Initial Model Evaluation]")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()
plt.show()

# 6. Model Improvement: Hyperparameter Tuning with GridSearchCV
param_grid = {
	'n_estimators': [50, 100, 200],
	'max_depth': [3, 5, 7],
	'learning_rate': [0.01, 0.1, 0.2]
}
grid = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_resampled, y_train_resampled)
print('\nBest Hyperparameters:', grid.best_params_)
print('Best Cross-Validation Score:', grid.best_score_)

# Test the best model
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print("\n[Evaluation After Hyperparameter Tuning]")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))
print("AUC Score:", roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1]))

# 7. Summary and Reporting
print("\n--- Model Performance Summary ---")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")
print(f"AUC Score: {roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1]):.2f}")
print("The model detects Parkinson's Disease with high accuracy.")
print("Future improvements can include trying different models (Random Forest, SVM) and feature engineering.")
print("This model shows promise for early diagnosis in healthcare applications.")
y_pred_best = best_model.predict(X_test_scaled)
print("\n[GridSearchCV Sonrası Test Sonuçları]")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))
print("AUC Score:", roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1]))

# 8. Sonuçların Sunumu ve Raporlama
print("\n--- Model Performans Özeti ---")
print(f"Test Doğruluk Oranı: {accuracy_score(y_test, y_pred_best):.2f}")
print(f"AUC Skoru: {roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1]):.2f}")
print("Model, Parkinson hastalarını yüksek doğrulukla tespit etmektedir.\n")
print("Gelecekte, farklı modeller (Random Forest, SVM) ve özellik mühendisliği ile daha da iyileştirilebilir.")
print("Bu model, sağlık alanında erken teşhis için umut vadetmektedir.")
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

