import os
import pandas as pd
from datetime import datetime


def convert_nifty_data(txt_file):
    """
    Convert NIFTY data format from .txt file to include datetime column and extract date components

    Expected input format: SYMBOL,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOLUME1,VOLUME2
    Output format: datetime,year,month,day,open,high,low,close (without volume columns)
    """
    # Define column names for the input data
    columns = [
        "symbol",
        "date",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume1",
        "volume2",
    ]

    # Read the .txt file (comma-separated)
    df = pd.read_csv(txt_file, names=columns, header=None)

    # Convert date from YYYYMMDD format to datetime
    df["date_str"] = df["date"].astype(str)

    # Combine date and time to create datetime column
    df["datetime"] = pd.to_datetime(
        df["date_str"] + " " + df["time"], format="%Y%m%d %H:%M"
    )

    # Extract year, month, day
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day

    # Select only required columns and reorder
    result_df = df[["datetime", "year", "month", "day", "open", "high", "low", "close"]]

    return result_df


def process_csv_data(csv_file, date_cutoff="2025-02-01"):
    """
    Process CSV data with date filtering and date component extraction

    Args:
        csv_file (str): Path to CSV file
        date_cutoff (str): Date cutoff in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Read the CSV data
    df = pd.read_csv(csv_file)

    # Convert date column to datetime
    df["datetime"] = pd.to_datetime(df["date"])

    # Extract year, month, day
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day

    # Drop the volume column if it exists
    if "volume" in df.columns:
        df = df.drop("volume", axis=1)

    # Reorder columns
    df = df[["datetime", "year", "month", "day", "open", "high", "low", "close"]]

    # Filter data before cutoff date
    df = df[df["datetime"] < date_cutoff]

    return df


def process_txt_files(directory="datasets"):
    """
    Process all .txt files in the given directory

    Args:
        directory (str): Directory path to search for .txt files

    Returns:
        pd.DataFrame: Combined dataframe from all .txt files
    """
    # Initialize empty dataframe
    combined_df = pd.DataFrame(
        columns=["datetime", "year", "month", "day", "open", "high", "low", "close"]
    )

    # Get all .txt files in directory
    txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

    print(f"Found {len(txt_files)} .txt files to process")

    # Process each .txt file
    for txt_file in txt_files:
        print(f"Processing: {txt_file}")
        try:
            temp_df = convert_nifty_data(os.path.join(directory, txt_file))
            combined_df = pd.concat([combined_df, temp_df], axis=0, ignore_index=True)
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")

    # Sort by datetime
    combined_df = combined_df.sort_values(by=["datetime"], ignore_index=True)

    return combined_df


def main():
    """
    Main function to orchestrate the data processing
    """
    print("Starting NIFTY data processing...")

    # Process CSV data
    csv_file = "datasets/NIFTY 50_minute_data.csv"
    if os.path.exists(csv_file):
        print(f"Processing CSV file: {csv_file}")
        df_csv = process_csv_data(csv_file)
        print(f"CSV data shape: {df_csv.shape}")
    else:
        print(f"CSV file {csv_file} not found")
        df_csv = pd.DataFrame()

    # Process TXT files
    print("\nProcessing .txt files...")
    df_txt = process_txt_files()
    print(f"Combined TXT data shape: {df_txt.shape}")

    # Combine all data if both exist
    if not df_csv.empty and not df_txt.empty:
        print("\nCombining CSV and TXT data...")
        final_df = pd.concat([df_csv, df_txt], axis=0, ignore_index=True)
        final_df = final_df.sort_values(by=["datetime"], ignore_index=True)
        print(f"Final combined data shape: {final_df.shape}")
    elif not df_csv.empty:
        final_df = df_csv
        print("Using only CSV data")
    elif not df_txt.empty:
        final_df = df_txt
        print("Using only TXT data")
    else:
        print("No data found to process")
        return

    # Display summary
    print(f"\nData Summary:")
    print(f"Date range: {final_df['datetime'].min()} to {final_df['datetime'].max()}")
    print(f"Total records: {len(final_df)}")
    print(f"Columns: {list(final_df.columns)}")

    # Save processed data
    output_file = "datasets/processed_nifty_data.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")

    return final_df


# Run the main function
if __name__ == "__main__":
    processed_data = main()
