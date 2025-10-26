from datasets import load_dataset
import os

DATASET_NAME = "imdb"

print("Downloading dataset")
raw_datasets = load_dataset(DATASET_NAME)

train_df = raw_datasets['train'].to_pandas()
test_df = raw_datasets['test'].to_pandas()

output_folder = DATASET_NAME + "_datasets"
os.makedirs(output_folder, exist_ok=True)

train_path = os.path.join(output_folder, DATASET_NAME+"_train.csv")
test_path = os.path.join(output_folder, DATASET_NAME + "_test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\nSuccessfully downloaded and saved files to the '{output_folder}' folder:")
print(f"Train data saved at: {train_path}")
print(f"Test data saved at: {test_path}")