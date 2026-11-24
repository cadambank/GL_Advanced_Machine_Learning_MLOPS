"""Step 4 - Push the deployment files to the Hugging Face Space."""
import os

from huggingface_hub import HfApi

SPACE_REPO = os.getenv("HF_SPACE_REPO", "cadambank/superkart-sales-forecast")

api = HfApi(token=os.getenv("HF_TOKEN"))
api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy SuperKart Streamlit app",
)
print(f"Deployed -> https://huggingface.co/spaces/{SPACE_REPO}")
