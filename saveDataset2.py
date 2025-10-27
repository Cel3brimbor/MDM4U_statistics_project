from datasets import load_dataset
import os

DATASET_NAME = "amazon_polarity"
NUM_SAMPLES = 100000

print(f"Downloading the first {NUM_SAMPLES:,} samples from the '{DATASET_NAME}' training split...")

raw_dataset_slice = load_dataset(
    DATASET_NAME, 
    split=f"train[:{NUM_SAMPLES}]"
)

test_dataset = load_dataset(DATASET_NAME, split=f"test[:{NUM_SAMPLES}]")

train_df_slice = raw_dataset_slice.to_pandas()
test_df = test_dataset.to_pandas()

output_folder = DATASET_NAME + "_datasets_100k"
os.makedirs(output_folder, exist_ok=True)

train_path = os.path.join(output_folder, DATASET_NAME + f"_{NUM_SAMPLES//1000}k_train.csv")
test_path = os.path.join(output_folder, DATASET_NAME + "_test.csv")

train_df_slice.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\nSuccessfully downloaded and saved files to the '{output_folder}' folder:")
print(f"Custom Train data saved at: {train_path} ({len(train_df_slice):,} rows)")
print(f"Test data saved at: {test_path} ({len(test_df):,} rows)")