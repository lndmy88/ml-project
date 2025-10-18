import os
import pandas as pd
from utils.utils import load_config

config = load_config()

RAW_DIR = config["data"]["raw_path"]
PROCESSED_DIR = config["data"]["processed_path"]
PROCESSED_FILE = config["data"]["processed_file"]  # e.g., "processed_data.csv"

def load_raw_data():
    """Load all CSV files in data/raw/ into a single DataFrame."""
    dataframes = []
    for file in os.listdir(RAW_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, fill missing values, etc."""
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")  # forward-fill missing values
    return df


def save_processed_data(df: pd.DataFrame):
    """Save the cleaned and processed data to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, PROCESSED_FILE)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")

def main():
    df = load_raw_data()
    df = clean_data(df)
    save_processed_data(df)

if __name__ == "__main__":
    main()
