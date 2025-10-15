from email import utils
import mlflow
import joblib
import pandas as pd
import os
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import load_config, load_params

# Load configuration
config = load_config()
TEST_DIR = config["data"]["test_path"]
TEST_FILE_X = config["data"]["test_x"]
TEST_FILE_Y = config["data"]["test_y"]
MODEL_DIR =  config["artifacts"]["model_path"]
MODEL_NAME = config["artifacts"]["model_name"]
MODEL_FILENAME = config["artifacts"]["model_filename"]
REPORTS_DIR = config["artifacts"]["reports_path"]
METRICS_FILE = config["artifacts"]["metrics_filename"]

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set and log metrics to MLflow.
    """
    model = joblib.load("registered_models/best_weather_regressor.joblib")

    # Prepare test data
    X_test = pd.read_csv(X_test)
    y_test = pd.read_csv(y_test)
    test_not_nan = y_test['mean_temp'].notna()
    X_test = X_test[test_not_nan]
    y_test = y_test.loc[test_not_nan, 'mean_temp']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)

    # Save metrics locally for DVC
    metrics = {
        "test_mse": mse,
        "test_mae": mae,
        "test_r2": r2
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, METRICS_FILE), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate_model(os.path.join(MODEL_DIR, MODEL_FILENAME), os.path.join(TEST_DIR, TEST_FILE_X), os.path.join(TEST_DIR, TEST_FILE_Y))
    print("Model evaluation completed and metrics logged to MLflow.")