import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from mlflow.models import infer_signature
from utils import load_config, load_params

# Load configuration
config = load_config()
TRAINING_DIR = config["data"]["train_path"]
TRAINING_FILE_X = config["data"]["train_x"]
TRAINING_FILE_Y = config["data"]["train_y"]
TEST_DIR = config["data"]["test_path"]
TEST_FILE_X = config["data"]["test_x"]
TEST_FILE_Y = config["data"]["test_y"]
MODEL_DIR =  config["artifacts"]["model_path"]
MODEL_NAME = config["artifacts"]["model_name"]
MODEL_FILENAME = config["artifacts"]["model_filename"]

# Load params
params = load_params()
N_ESTIMATORS=params["model"]["n_estimators"]
MAX_DEPTH=params["model"]["max_depth"]
RANDOM_STATE=params["train"]["random_state"]
MIN_SAMPLES_SPLIT=params["model"]["min_samples_split"]
MIN_SAMPLES_LEAF=params["model"]["min_samples_leaf"]
N_JOBS=params["model"]["n_jobs"]

def train_model():
    """
    Train a machine learning model.
    """
  
    # Prepare data
    X_train = pd.read_csv(os.path.join(TRAINING_DIR, TRAINING_FILE_X))
    X_test = pd.read_csv(os.path.join(TEST_DIR, TEST_FILE_X))
    y_train_df = pd.read_csv(os.path.join(TRAINING_DIR, TRAINING_FILE_Y))
    y_test_df = pd.read_csv(os.path.join(TEST_DIR, TEST_FILE_Y))

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
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Export model to joblib
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    output_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(rf,  output_path)

    # Log the model with MLflow
    signature = infer_signature(X_test, y_pred)
    logged_model = mlflow.sklearn.log_model(rf, name=MODEL_NAME, signature=signature, input_example=X_test.head(1))

    # Register the model
    mlflow.register_model(
        #model_uri =   f"runs:/{mlflow.active_run().info.run_id}/model",
        model_uri = logged_model.model_uri,
        name=MODEL_NAME
    )

if __name__ == "__main__":
    train_model()
    print(F"Model training completed and model saved to {MODEL_DIR}/{MODEL_FILENAME}")