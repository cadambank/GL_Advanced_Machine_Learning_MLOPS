"""Step 1 - Register the raw SuperKart dataset on the Hugging Face dataset hub."""
import os

from huggingface_hub import HfApi

REPO_ID = os.getenv("HF_DATASET_REPO", "cadambank/superkart-sales-data")
RAW_FILE = "superkart_project/data/SuperKart.csv"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the dataset repository (public) if it does not exist yet
api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)

# Upload the raw data file
api.upload_file(
    path_or_fileobj=RAW_FILE,
    path_in_repo="SuperKart.csv",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Register raw SuperKart dataset",
)
print(f"Registered {RAW_FILE} -> https://huggingface.co/datasets/{REPO_ID}")
