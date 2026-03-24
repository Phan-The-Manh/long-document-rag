import pandas as pd
import os

# Define the local path (ensure 'r' prefix for Windows paths)
file_path = r"D:\long_doc_agent\data\eval_dataset_store\retrieval_evaluation_results.csv"

if os.path.exists(file_path):
    # Load the CSV
    df = pd.read_csv(file_path)
    
    print(f"📋 Found {len(df)} records in evaluation dataset.\n")
    print("=" * 80)

    # Iterate through every row in the dataframe
    for index, row in df.iterrows():
        print(f"RECORD #{index + 1}")
        print("-" * 30)
        
        # Print each column and its full value
        print(f"▶ USER INPUT:\n{row['user_input']}\n")
        
        print(f"▶ REFERENCE (Ground Truth):\n{row['reference']}\n")
        
        # Note: These are usually lists or long strings
        print(f"▶ RETRIEVED CONTEXTS:\n{row['retrieved_contexts']}\n")
        
        # If 'reference_contexts' exists and is different from reference
        if 'reference_contexts' in df.columns:
            print(f"▶ REFERENCE CONTEXTS:\n{row['reference_contexts']}\n")
        
        print(f"📈 METRICS:")
        print(f"   - Context Precision: {row['context_precision']}")
        print(f"   - Context Recall:    {row['context_recall']}")
        
        print("=" * 80) # Large separator between records

else:
    print(f"❌ File not found at: {file_path}")