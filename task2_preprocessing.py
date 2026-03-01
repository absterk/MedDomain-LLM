# Read the Excel file "train-test-data.xlsx" and perform the following operations:
# Split it into training and testing sets (80% training, 20% testing). Use random shuffle to ensure that the data is mixed well before splitting.
# For each set, save it as a JSON file named "train.json" and "test.json" respectively.
# {"prompt": <input_findings>, "response": <output_impression>}

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from config.path_config import TEST_JSON, TRAIN_JSON, TRAIN_TEST_XLSX
from config.training_config import RANDOM_STATE, TRAIN_TEST_SPLIT

# Read the Excel file
data = pd.read_excel(TRAIN_TEST_XLSX)
# Shuffle the data and split into training and testing sets
train_data, test_data = train_test_split(data, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)
# Prepare the training data in the required format
train_json = []
for index, row in train_data.iterrows():
    train_json.append({"prompt": row['input_findings'], "response": row['output_impression']})
# Prepare the testing data in the required format
test_json = []
for index, row in test_data.iterrows():
    test_json.append({"prompt": row['input_findings'], "response": row['output_impression']})
# Save the training data to train.json
with open(TRAIN_JSON, "w", encoding="utf-8") as train_file:
    json.dump(train_json, train_file, indent=4)
# Save the testing data to test.json
with open(TEST_JSON, "w", encoding="utf-8") as test_file:
    json.dump(test_json, test_file, indent=4)
