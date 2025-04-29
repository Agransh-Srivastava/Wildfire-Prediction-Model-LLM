# scripts/prepare_data.py

import os
import pandas as pd

RAW_DATA_PATH = '../data/raw'
PROCESSED_DATA_PATH = '../data/processed'
OUTPUT_FILE = 'cleaned_data.csv'

def load_csvs(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def clean_data(df):
    # Remove duplicate rows entirely
    print(f"Original records: {len(df)}")
    df = df.drop_duplicates()
    print(f"After full row duplicate removal: {len(df)}")
    
    # Remove duplicate based on main text columns if exists
    possible_text_columns = ['text', 'description', 'content', 'case_details']
    for col in possible_text_columns:
        if col in df.columns:
            df = df.drop_duplicates(subset=[col])
            print(f"After duplicate removal based on '{col}': {len(df)}")
            break
    else:
        print("Warning: No obvious text column found for advanced deduplication.")
    
    df = df.reset_index(drop=True)
    return df

def save_clean_data(df, output_folder, filename):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    df = load_csvs(RAW_DATA_PATH)
    cleaned_df = clean_data(df)
    save_clean_data(cleaned_df, PROCESSED_DATA_PATH, OUTPUT_FILE)
