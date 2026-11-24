"""Step 2 - Load data from the HF hub, clean it, split it, and upload the splits back."""
import os

import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

REPO_ID = os.getenv("HF_DATASET_REPO", "cadambank/superkart-sales-data")
TOKEN = os.getenv("HF_TOKEN")
SPLIT_DIR = "superkart_project/data/splits"
TARGET = "Product_Store_Sales_Total"

# 1. Load directly from the Hugging Face dataset hub
df = pd.read_csv(f"hf://datasets/{REPO_ID}/SuperKart.csv", storage_options={"token": TOKEN})
print("Loaded from hub:", df.shape)

# 2. Clean
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
df = df.drop(columns=["Product_Id", "Store_Id"])   # identifiers, no predictive information
df = df.drop_duplicates().reset_index(drop=True)
print("After cleaning:", df.shape)
print("Sugar content levels:", sorted(df["Product_Sugar_Content"].unique()))

# 3. Split into train / test
X, y = df.drop(columns=[TARGET]), df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Save locally
os.makedirs(SPLIT_DIR, exist_ok=True)
splits = {"Xtrain.csv": Xtrain, "Xtest.csv": Xtest, "ytrain.csv": ytrain, "ytest.csv": ytest}
for name, frame in splits.items():
    frame.to_csv(os.path.join(SPLIT_DIR, name), index=False)
print("Train:", Xtrain.shape, "| Test:", Xtest.shape)

# 5. Upload the splits back to the dataset hub
HfApi(token=TOKEN).upload_folder(
    folder_path=SPLIT_DIR,
    path_in_repo="splits",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Upload cleaned train/test splits",
)
print(f"Uploaded {list(splits)} -> https://huggingface.co/datasets/{REPO_ID}/tree/main/splits")
