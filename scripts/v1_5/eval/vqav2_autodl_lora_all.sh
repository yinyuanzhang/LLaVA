#!/bin/bash

# 定义 GPU 列表
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# 定义检查点列表
CHECKPOINTS=(5500 5000 4500 4000 3000 2000 1000 500)
SPLIT="llava_vqav2_mscoco_test-dev2015"

# 遍历每个检查点
for CKPT_NUM in "${CHECKPOINTS[@]}"; do
    CKPT="checkpoint-${CKPT_NUM}"

    echo "Processing checkpoint: $CKPT"

    # 并行推理
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path ~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/$CKPT \
            --model-base liuhaotian/llava-v1.5-7b \
            --question-file ~/autodl-tmp/playground/data/eval/vqav2/$SPLIT.jsonl \
            --image-folder ~/autodl-tmp/playground/data/eval/vqav2/test2015 \
            --answers-file ~/autodl-tmp/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    # 等待所有 GPU 完成推理
    wait

    # 合并结果文件
    output_file=~/autodl-tmp/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

    # 清空输出文件
    > "$output_file"

    # 合并分块结果
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ~/autodl-tmp/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    echo "Merged results for checkpoint: $CKPT"

    # 转换为提交格式
    python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT
done

echo "All checkpoints processed."