
# ==============================
# PART 2: Telemonitoring Dataset (Regression)
# ==============================
def main():
	print("\n==============================\n")
	print("Parkinson's Disease Telemonitoring Dataset Analysis (Regression)")
	print("==============================\n")

	import os
	import warnings
	warnings.filterwarnings('ignore')
	import matplotlib.pyplot as plt
	import seaborn as sns
	import numpy as np
	import pandas as pd
	from sklearn.model_selection import train_test_split, GridSearchCV
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.linear_model import LinearRegression
	from sklearn.ensemble import RandomForestRegressor
	from xgboost import XGBRegressor
	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

	# Load telemonitoring dataset robustly
	tele_path = os.path.join("parkinsons", "telemonitoring", "parkinsons_updrs.data")
	try:
		# Try with header
		tele_df = pd.read_csv(tele_path)
		if 'motor_UPDRS' not in tele_df.columns:
			# Try with header=0 (sometimes header is not detected)
			tele_df = pd.read_csv(tele_path, header=0)
	except Exception as e:
		print(f"Error loading data: {e}")
		return

	print(f"Telemonitoring dataset shape: {tele_df.shape}")
	print("Columns:", list(tele_df.columns))
	print(tele_df.head())

	# Check for required columns
	required_cols = {'motor_UPDRS', 'total_UPDRS'}
	if not required_cols.issubset(set(tele_df.columns)):
		print("ERROR: Required columns not found in the dataset. Check the file format and header row.")
		return

	# EDA: Target distributions
	try:
		fig, axes = plt.subplots(1, 2, figsize=(12, 4))
		sns.histplot(tele_df['motor_UPDRS'], bins=30, ax=axes[0], kde=True, color='skyblue')
		axes[0].set_title('Distribution of motor_UPDRS')
		sns.histplot(tele_df['total_UPDRS'], bins=30, ax=axes[1], kde=True, color='salmon')
		axes[1].set_title('Distribution of total_UPDRS')
		plt.tight_layout()
		plt.show()
	except Exception as e:
		print(f"Plotting error: {e}")

	# Feature selection (exclude subject#, age, sex, test_time, targets)
	features = [col for col in tele_df.columns if col not in ['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS']]
	X = tele_df[features]
	y_motor = tele_df['motor_UPDRS']
	y_total = tele_df['total_UPDRS']

	# Scaling
	scaler = MinMaxScaler()
	X_scaled = scaler.fit_transform(X)

	# Train/test split
	X_train, X_test, y_motor_train, y_motor_test = train_test_split(X_scaled, y_motor, test_size=0.2, random_state=42)
	_, _, y_total_train, y_total_test = train_test_split(X_scaled, y_total, test_size=0.2, random_state=42)

	# --- Regression Models ---
	models = {
		'LinearRegression': LinearRegression(),
		'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
		'XGBoost': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
	}

	def evaluate_regression(y_true, y_pred, label):
		# Compute RMSE in a way compatible with all sklearn versions
		rmse = np.sqrt(mean_squared_error(y_true, y_pred))
		mae = mean_absolute_error(y_true, y_pred)
		r2 = r2_score(y_true, y_pred)
		print(f"{label} - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
		return rmse, mae, r2

	print("\nRegression Results for motor_UPDRS:")
	for name, model in models.items():
		model.fit(X_train, y_motor_train)
		y_pred = model.predict(X_test)
		evaluate_regression(y_motor_test, y_pred, name)

	print("\nRegression Results for total_UPDRS:")
	for name, model in models.items():
		model.fit(X_train, y_total_train)
		y_pred = model.predict(X_test)
		evaluate_regression(y_total_test, y_pred, name)

	# --- Hyperparameter Tuning for XGBoost (motor_UPDRS) ---
	print("\nGridSearchCV for XGBoost (motor_UPDRS)...")
	param_grid = {
		'n_estimators': [50, 100],
		'max_depth': [3, 5],
		'learning_rate': [0.05, 0.1]
	}
	xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
	grid = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
	grid.fit(X_train, y_motor_train)
	print(f"Best params: {grid.best_params_}")
	y_pred = grid.predict(X_test)
	print("XGBoost (tuned) results for motor_UPDRS:")
	evaluate_regression(y_motor_test, y_pred, "XGBoost (tuned)")

	# --- Plot predicted vs actual ---
	try:
		plt.figure(figsize=(6, 6))
		plt.scatter(y_motor_test, y_pred, alpha=0.3, color='teal')
		plt.plot([y_motor_test.min(), y_motor_test.max()], [y_motor_test.min(), y_motor_test.max()], 'r--')
		plt.xlabel('Actual motor_UPDRS')
		plt.ylabel('Predicted motor_UPDRS')
		plt.title('XGBoost (tuned): Actual vs Predicted motor_UPDRS')
		plt.tight_layout()
		plt.show()
	except Exception as e:
		print(f"Plotting error: {e}")

	print("\nTelemonitoring regression analysis completed.\n")


if __name__ == "__main__":
	main()
