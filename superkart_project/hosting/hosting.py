"""Step 4 - Push the deployment files to the Hugging Face Space."""
import os

from huggingface_hub import HfApi

SPACE_REPO = os.getenv("HF_SPACE_REPO", "cadambank/superkart-sales-forecast")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the Space only if it does not exist yet. Hugging Face now requires a PRO plan to create
# new Docker Spaces on free hardware, so this project deploys into an existing Docker Space.
if not api.repo_exists(SPACE_REPO, repo_type="space"):
    api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker", private=False)

# Upload the deployment files; delete_patterns removes any stale files from earlier deployments
api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    delete_patterns=["*", "**/*"],
    commit_message="Deploy SuperKart Streamlit app",
)
print(f"Deployed -> https://huggingface.co/spaces/{SPACE_REPO}")
