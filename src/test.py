from utils.mlflow_utils import get_dvc_data_hash, log_data_version_info, log_training_data_info

print(get_dvc_data_hash("data/training"))