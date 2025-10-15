import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import load_config, load_params

# Load configuration
config = load_config()

PROCESSED_DIR = config["data"]["processed_path"]
PROCESSED_FILE = config["data"]["processed_file"]  # e.g., "l
TRAINING_DIR = config["data"]["train_path"]
TRAINING_FILE_X = config["data"]["train_x"]
TRAINING_FILE_Y = config["data"]["train_y"]
TEST_DIR = config["data"]["test_path"]
TEST_FILE_X = config["data"]["test_x"]
TEST_FILE_Y = config["data"]["test_y"]

# Load params
params = load_params()
TEST_SIZE=params["train"]["test_size"]
RANDOM_STATE=params["train"]["random_state"]

def engineer_features(weather):
    """
    Perform feature engineering on the London weather DataFrame.
    """

    # Define X and y
    X = weather[['sunshine', 'global_radiation', 'cloud_cover']]
    y = weather['mean_temp']

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Convert back to DataFrame for saving
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=['sunshine', 'global_radiation', 'cloud_cover'], index=X_train.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=['sunshine', 'global_radiation', 'cloud_cover'], index=X_test.index
    )
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    # Save to files
   
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    train_output_path_x = os.path.join(TRAINING_DIR, TRAINING_FILE_X)
    X_train_scaled_df.to_csv(train_output_path_x)
    print(f"✅ Training features saved to {train_output_path_x}")
    train_output_path_y = os.path.join(TRAINING_DIR, TRAINING_FILE_Y)
    y_train_df.to_csv(train_output_path_y)
    print(f"✅ Training target saved to {train_output_path_y}")
    test_output_path_x = os.path.join(TEST_DIR, TEST_FILE_X)
    X_test_scaled_df.to_csv(test_output_path_x)
    print(f"✅ Test features saved to {test_output_path_x}")
    test_output_path_y = os.path.join(TEST_DIR, TEST_FILE_Y)
    y_test_df.to_csv(test_output_path_y)
    print(f"✅ Test target saved to {test_output_path_y}")
    
if __name__ == "__main__":
    # Load the processed data
    weather = pd.read_csv(os.path.join(PROCESSED_DIR, PROCESSED_FILE))

    # Perform feature engineering
    engineer_features(weather)
    print(f"Feature engineering completed and data saved to {TRAINING_DIR} and {TEST_DIR} directories.")