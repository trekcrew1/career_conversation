


from huggingface_hub import HfApi
api = HfApi(token="hf_your_token")
api.delete_space_storage(repo_id="owner/space-name")