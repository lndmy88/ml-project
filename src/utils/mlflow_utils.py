import mlflow
import os
import json
import psutil
import platform
from datetime import datetime
import yaml
import subprocess
from utils.utils import load_config

# Load configuration
config = load_config()
LOCK_PATH = config["data"]["lock_file"]
PROCESSED_DATA_PATH = config["data"]["processed_path"]
TRAINING_DATA_PATH = config["data"]["train_path"]
TEST_DATA_PATH = config["data"]["test_path"]
PREPARE_DATA_STAGE = "prepare_data"
ENGINEER_FEATURES_STAGE = "engineer_features"
TRAINING_STAGE = "train_model"

def log_model_metrics(metrics: dict):
    """
    Log evaluation metrics to MLflow.
    Example: metrics = {"rmse": 0.45, "r2": 0.89}
    """
    try:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        print(f"Model metrics logged: {metrics}")    
    except Exception as e:
        print(f"Error logging model metrics: {e}")

def log_system_metrics():
    """
    Log system-level metrics (CPU, memory, platform info).
    """
    try:
        mlflow.log_metric("cpu_percent", psutil.cpu_percent())
        mlflow.log_metric("memory_percent", psutil.virtual_memory().percent)
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("python_version", platform.python_version())

        print("System metrics logged.")
    except Exception as e:
        print(f"Error logging system metrics: {e}")

def log_model_info(model_path: str, data_version: str = None, commit_hash: str = None):
    """
    Log information about the model artifact and its associated data.
    """
    try:
        mlflow.log_param("model_path", model_path)
        if data_version:
            mlflow.log_param("data_version", data_version)
        if commit_hash:
            mlflow.log_param("git_commit", commit_hash)
        
        print("Model info logged.")
    except Exception as e:
        print(f"Error logging model info: {e}")

def log_dict_as_artifact(data: dict, artifact_name: str = "metadata"):
    """
    Log a dictionary as an artifact JSON file to MLflow.
    """
    artifact_path = f"{artifact_name}.json"
    try:
        with open(artifact_path, "w") as f:
            json.dump(data, f, indent=4)
        mlflow.log_artifact(artifact_path)

        print(f"Artifact {artifact_name} logged.")
    except Exception as e:
        print(f"Error logging artifact {artifact_name}: {e}")
    finally:
        os.remove(artifact_path)  # clean up local file
   
def get_git_commit_hash():
    """Return current Git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception as e:
        print("Error retrieving Git commit hash:", e)
        return None

def get_dvc_data_hash(data_path):
    try:
        dvc_output = subprocess.check_output(["dvc", "list", ".", data_path, "--show-hash"]).decode("utf-8").strip()
        if dvc_output:
            result = {}
            for line in dvc_output.split("\n"):
                parts = line.split()
                if len(parts) == 2:
                    hash_value, file_path = parts
                    result[file_path] = hash_value
            return result
    except Exception as e:
        print("Error retrieving DVC data hash (list):", e)
        return None

def log_input_data(data_path: str, artifact_name: str = "input_data_info"):
    """
    Log input dataset info (path, Git commit, and DVC hash) to MLflow.
    """
    git_commit = get_git_commit_hash()
    dvc_hash = get_dvc_data_hash(data_path)

    input_info = {
        "data_path": data_path,
        "git_commit": git_commit,
        "dvc_hash": dvc_hash,
    }

    # log all as params
    try:
        mlflow.log_param("data_path", data_path)
        if git_commit:
            mlflow.log_param("git_commit", git_commit)
        if dvc_hash:
            mlflow.log_param("dvc_hash", dvc_hash)
    except Exception as e:
        print("Error logging input data:", e)

    # also save full info as artifact
    artifact_path = artifact_name + ".json"
    try:
        with open(artifact_path, "w") as f:
            json.dump(input_info, f, indent=4)
        mlflow.log_artifact(artifact_path)
    except Exception as e:
        print("Error logging input data artifact:", e)
    finally:    
        os.remove(artifact_path)

