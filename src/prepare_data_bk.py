import pandas as pd

def load_and_prepare_data(csv_path):
    """
    Load and prepare the London weather data from a CSV file.
    """

    # Read the data
    weather = pd.read_csv(csv_path)

    # Display the first few rows
    print(weather.head())

    # Convert the 'date' column to datetime format
    weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')

    # Extract year, month, and day from the 'date' column into new columns
    weather['year'] = weather['date'].dt.year
    weather['month'] = weather['date'].dt.month
    weather['day'] = weather['date'].dt.day



    return weather
    

if __name__ == "__main__":
    df = load_and_prepare_data("data/raw/london_weather.csv")
    df.to_csv("data/processed/london_weather__processed.csv", index=False)
    print("Processed data saved to data/processed/london_weather__processed.csv")