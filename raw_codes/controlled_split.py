import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "custom_dataset.csv"  # Replace with your dataset file path
data = pd.read_csv(file_path)

# Group data by sentence1 to ensure pairs are kept together
grouped_data = data.groupby('sentence1')

# Assign a group ID to each pair
pairs = [group for _, group in grouped_data]

# Split the groups into training and testing sets
train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

# Concatenate the pairs back into DataFrames
train_data = pd.concat(train_pairs).reset_index(drop=True)
test_data = pd.concat(test_pairs).reset_index(drop=True)

# Save the results to separate files
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print(f"Training data saved to train_data.csv with {len(train_data)} rows.")
print(f"Testing data saved to test_data.csv with {len(test_data)} rows.")
