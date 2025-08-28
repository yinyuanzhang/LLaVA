from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="AILab-CVC/SEED-Bench",
    repo_type="dataset"
)

print(f"仓库已下载到: {local_dir}")