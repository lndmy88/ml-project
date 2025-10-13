import mlflow
import joblib
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

if __name__ == "__main__":
    evaluate_model("models/best_weather_regressor.joblib", "data/test/X_test_scaled.csv", "data/test/y_test.csv")
    print("Model evaluation completed and metrics logged to MLflow.")