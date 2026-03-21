import pandas as pd
import os

def inspect_dataset():
    # The exact path where your dataset was saved
    csv_path = r"D:\long_doc_agent\data_store\eval_dataset_store\testset_output.csv"

    # 1. Check if the file is actually there
    if not os.path.exists(csv_path):
        print(f"❌ Could not find the file at {csv_path}")
        return

    # 2. Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # 3. Print high-level stats
    print("=========================================")
    print(f"✅ Successfully loaded {len(df)} questions!")
    print(f"📊 Available Columns: {list(df.columns)}")
    print("=========================================\n")

    # 4. Display EVERY column and its full content for the first 3 rows
    for index, row in df.head(3).iterrows():
        print(f"📝 ROW {index + 1}")
        print("=" * 40)
        
        # Loop through all columns dynamically
        for col in df.columns:
            # We convert the value to string just in case there are numeric or null values
            val = str(row[col])
            print(f"📌 {col.upper()}:\n{val}\n")
            
        print("-" * 60 + "\n")

if __name__ == "__main__":
    inspect_dataset()