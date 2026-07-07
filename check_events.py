import pandas as pd
import os

def inspect_file(file_path, label):
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return

    print(f"\n--- INSPECTING {label} ---")
    df = pd.read_csv(file_path)
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print("\nFirst 5 rows (raw data):")
    print(df.head())
    print("-" * 30)

if __name__ == "__main__":
    # Pointing to the data directory based on your previous logs
    inspect_file('data/Master_HEL1OS.csv', 'HEL1OS')
    inspect_file('data/Master_SoLEXS.csv', 'SoLEXS')