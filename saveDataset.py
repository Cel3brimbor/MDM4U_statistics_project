import os
import pandas as pd
from datasets import load_dataset
import numpy as np

DATASET_NAME = "imdbLimited"
HF_DATASET_ID = "imdb"
RANDOM_STATE = 42
OUTPUT_FOLDER = DATASET_NAME + "_datasets"
MAX_SAMPLES = 5000

try:
    raw_datasets = load_dataset(HF_DATASET_ID)

    train_df_full = raw_datasets['train'].to_pandas()
    test_df_full = raw_datasets['test'].to_pandas()
    
    train_df = train_df_full.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)
    
    test_df = test_df_full.copy()

    train_df = train_df.rename(columns={'label': 'label'})
    test_df = test_df.rename(columns={'label': 'label'})
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_FOLDER, DATASET_NAME+"_train.csv")
    test_path = os.path.join(OUTPUT_FOLDER, DATASET_NAME + "_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Successfully created files in the '{OUTPUT_FOLDER}' folder:")
    print(f"Train data saved at: {train_path} ({len(train_df):,} samples)")
    print(f"Test data saved at: {test_path} ({len(test_df):,} samples)")
    
except Exception as e:
    print(f"Error during dataset download or processing: {e}")