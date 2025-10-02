import os
from huggingface_hub import HfApi

from dotenv import load_dotenv
load_dotenv()

# Ensure OPENAI_API_KEY exists even if .env uses 'openai_api_key'
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

api = HfApi(token=HF_TOKEN)
# me = api.whoami()
# user = me.get("name") or me.get("user", {}).get("name")
# all_spaces = set(list_spaces_for(user))

target = "trekcrew/career_digital_twin"  # e.g. "alice/my-space"
try:
    info = api.repo_info(repo_id=target, repo_type="space")
    print("Repo exists:", info)

    # Delete if present (use the normalized IDs above)
    api.delete_repo(repo_id=target, repo_type="space")
    print(f"Deleted Space: {target}")

except Exception as e:
    print(f"Repository {target} does not exist or cannot be accessed: {e}")
    exit(1)
