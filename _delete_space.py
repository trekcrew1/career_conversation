from huggingface_hub import HfApi

# Import HfHubHTTPError in a version-tolerant way
try:
    from huggingface_hub.utils import HfHubHTTPError  # preferred import
except Exception:
    try:
        from huggingface_hub.errors import HfHubHTTPError  # fallback on some versions
    except Exception:
        HfHubHTTPError = Exception  # ultimate fallback: still catches API errors

SPACE = "4robmorrow/career_digital_twin"
api = HfApi()

print(f"Target Space: {SPACE}")

# 1) Try to wipe persistent storage (safe to skip if unsupported/not enabled)
if hasattr(api, "delete_space_storage"):
    try:
        api.delete_space_storage(repo_id=SPACE)
        print("✓ Storage wiped (or nothing to wipe).")
    except Exception as e:
        print(f"• Skipped storage wipe (likely not enabled / no permission): {e}")
else:
    print("• delete_space_storage not available in your huggingface_hub version.")

# 2) Delete the Space repo itself
try:
    api.delete_repo(repo_id=SPACE, repo_type="space")
    print("✓ Space repository deleted.")
except HfHubHTTPError as e:
    print(f"✗ Delete failed (maybe already gone or permission issue): {e}")

# 3) Verify it’s gone
me = api.whoami()
user = me.get("name") or me.get("user", {}).get("name")
remaining = []
for s in api.list_spaces(author=user):
    rid = getattr(s, "id", "")
    remaining.append(rid if "/" in rid else f"{user}/{rid}")

print("✓ Verified removed from your Spaces." if SPACE not in remaining else "✗ Still present in listing.")
