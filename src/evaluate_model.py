from email import utils
import mlflow
import joblib
import pandas as pd
import os
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.utils import load_config, load_params
from utils.mlflow_utils import log_data_version_info, log_dict_as_artifact, log_input_data, log_model_metrics

# Load configuration
config = load_config()
EXPERIMENT_NAME = config["mlflow"]["experiment_name"]   
TEST_DIR = config["data"]["test_path"]
TEST_FILE_X = config["data"]["test_x"]
TEST_FILE_Y = config["data"]["test_y"]
MODEL_DIR =  config["artifacts"]["model_path"]
MODEL_NAME = config["artifacts"]["model_name"]
MODEL_FILENAME = config["artifacts"]["model_filename"]
REPORTS_DIR = config["artifacts"]["reports_path"]
METRICS_FILE = config["artifacts"]["metrics_filename"]
EVALUATE_MODEL_STAGE = "evaluate_model"

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

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="evaluate_model"):
        # Log model info
        mlflow.log_param("model_name", MODEL_NAME)

        # Log data version
        log_input_data("data/test", artifact_name="test_data_info")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics to MLflow
        metrics = {
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2
        }
        log_model_metrics(metrics)
        log_dict_as_artifact(metrics, "evaluation_metrics")

        # Save metrics locally for DVC
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(os.path.join(REPORTS_DIR, METRICS_FILE), 'w') as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate_model(os.path.join(MODEL_DIR, MODEL_FILENAME), os.path.join(TEST_DIR, TEST_FILE_X), os.path.join(TEST_DIR, TEST_FILE_Y))
    print("Model evaluation completed and metrics logged to MLflow.")