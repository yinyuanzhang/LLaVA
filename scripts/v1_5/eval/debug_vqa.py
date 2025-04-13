import os
import subprocess
from pathlib import Path

# 配置参数
gpu_list = os.getenv("CUDA_VISIBLE_DEVICES", "0")  # 获取 GPU 列表
GPULIST = gpu_list.split(",")  # 按逗号分割 GPU 列表
CHUNKS = len(GPULIST)  # GPU 数量

CKPT = "llava-v1.5-7b-lora-noprefusion"
SPLIT = "llava_vqav2_mscoco_test-dev2015"

AUTO_DL_TMP = "/users/zyy/autodl-tmp"

# 定义输出目录
output_dir = Path(f"{AUTO_DL_TMP}/playground/data/eval/vqav2/answers/{SPLIT}/{CKPT}").expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

# 存储子进程
processes = []

# 启动并行任务
for IDX in range(CHUNKS):
    CUDA_VISIBLE_DEVICES = GPULIST[IDX]
    answers_file = output_dir / f"{CHUNKS}_{IDX}.jsonl"

    cmd = [
        "python", "-m", "llava.eval.model_vqa_loader",
        "--model-path", f"{AUTO_DL_TMP}/cache/hub/models--imagecache--llava-v1.5-7b-lora-noprefusion",
        "--model-base", "lmsys/vicuna-7b-v1.5",
        # "--model-path", "liuhaotian/llava-v1.5-7b",
        "--question-file", f"{AUTO_DL_TMP}/playground/data/eval/vqav2/{SPLIT}.jsonl",
        "--image-folder", f"{AUTO_DL_TMP}/playground/data/eval/vqav2/test2015",
        "--answers-file", str(answers_file),
        "--num-chunks", str(CHUNKS),
        "--chunk-idx", str(IDX),
        "--temperature", "0",
        "--conv-mode", "vicuna_v1"
    ]

    # 启动子进程
    process = subprocess.Popen(
        cmd,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES},
    )
    processes.append(process)

# 等待所有子进程完成
for process in processes:
    process.wait()

# 合并结果文件
output_file = output_dir / "merge.jsonl"
with open(output_file, "w") as outfile:
    for IDX in range(CHUNKS):
        chunk_file = output_dir / f"{CHUNKS}_{IDX}.jsonl"
        with open(chunk_file, "r") as infile:
            outfile.write(infile.read())

# 调用转换脚本
convert_script = "scripts/convert_vqav2_for_submission.py"
subprocess.run(
    ["python", convert_script, "--split", SPLIT, "--ckpt", CKPT],
    check=True,
)