from huggingface_hub import HfApi

api = HfApi()  # uses your saved token
me = api.whoami()
user = me.get("name") or me.get("user", {}).get("name")
print("Authenticated as:", user)

def norm_id(space_info, default_author):
    rid = getattr(space_info, "id", "") or ""
    author = getattr(space_info, "author", None) or default_author
    return rid if "/" in rid else f"{author}/{rid}"

# User-owned spaces
spaces_user = [norm_id(s, user) for s in api.list_spaces(author=user)]

# Org-owned spaces (if any)
orgs = [o.get("name") for o in me.get("orgs", [])]
spaces_orgs = []
for org in orgs:
    spaces_orgs.extend([norm_id(s, org) for s in api.list_spaces(author=org)])

all_spaces = sorted(set(spaces_user + spaces_orgs))

print("\nYour Spaces:")
if not all_spaces:
    print(" - (none)")
else:
    for rid in all_spaces:
        print(" -", rid)
