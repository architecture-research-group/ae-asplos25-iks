import sys
from huggingface_hub import snapshot_download, login

login()

def download(size):
    hf_path = f"meta-llama/Llama-3.1-{size}-Instruct"
    snapshot_download(repo_id=hf_path, local_dir=f"models/{size}",
                      local_dir_use_symlinks=False, revision="main")

if len(sys.argv) != 2 or sys.argv[1] not in ["8b", "70b", "405b", "all"]:
    print("Usage: python download.py {8b|70b|405b|all}")
    sys.exit(1)

if sys.argv[1] == "all":
    download("8b")
    download("70b")
    download("405b")
    sys.exit(0)

else:
    download(sys.argv[1])
    sys.exit(0)


