import os
import pandas as pd
import csv

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Check if symptom files exist
required_files = [
    'symptom_severity.csv',
    'symptom_Description.csv',
    'symptom_precaution.csv',
    'Training.csv',
    'Testing.csv',
    'healifyLLM-QA-scraped-dataset.csv'
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join('data', f))]

if missing_files:
    print(f"Missing data files: {', '.join(missing_files)}")
    print("Please download them to the 'data' directory before running the application.")
else:
    print("All required data files are present.")
    
    # Display data summary
    for file in required_files:
        try:
            df = pd.read_csv(os.path.join('data', file))
            print(f"{file}: {len(df)} records, {len(df.columns)} columns")
        except Exception as e:
            print(f"Error reading {file}: {e}")