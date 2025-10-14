import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd


def engineer_features(weather):
    """
    Perform feature engineering on the London weather DataFrame.
    """

    # Define X and y
    X = weather[['sunshine', 'global_radiation', 'cloud_cover']]
    y = weather['mean_temp']

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    TRAINING_DIR = "data/training"
    TRAINING_FILE_X = "X_train_scaled.csv"
    TRAINING_FILE_Y = "y_train.csv"
    TEST_DIR = "data/test"
    TEST_FILE_X = "X_test_scaled.csv"
    TEST_FILE_Y = "y_test.csv"
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
    weather = pd.read_csv("data/processed/london_weather__processed.csv")

    # Perform feature engineering
    engineer_features(weather)
    print("Feature engineering completed and data saved to data/training and data/test directories.")