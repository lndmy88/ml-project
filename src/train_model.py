import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def train_model(X_train, y_train):
    """
    Train a machine learning model.
    """
  
    # Prepare data
    X_train = pd.read_csv("data/training/X_train_scaled.csv")
    X_test = pd.read_csv("data/test/X_test_scaled.csv")
    y_train_df = pd.read_csv("data/training/y_train.csv")
    y_test_df = pd.read_csv("data/test/y_test.csv")

    # Remove rows with NaN in y_train and corresponding X_train rows
    train_not_nan = y_train_df['mean_temp'].notna()
    X_train = X_train[train_not_nan]
    y_train = y_train_df.loc[train_not_nan, 'mean_temp']

    # Remove rows with NaN in y_test and corresponding X_test rows (for evaluation)
    test_not_nan = y_test_df['mean_temp'].notna()
    X_test = X_test[test_not_nan]
    y_test = y_test_df.loc[test_not_nan, 'mean_temp']

    # Start MLflow run
    with mlflow.start_run(run_name="RandomForestRegressor-Weather"):
    # Train model with tuned hyperparameters
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Export model to joblib
    MODEL_DIR = "registered_models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    output_path = os.path.join(MODEL_DIR, "best_weather_regressor.joblib")
    joblib.dump(rf,  output_path)

if __name__ == "__main__":
    train_model("data/training/X_train_scaled.csv", "data/training/y_train.csv")
    print("Model training completed and model saved to registered_models/best_weather_regressor.joblib")