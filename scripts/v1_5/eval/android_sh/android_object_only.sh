#!/bin/bash

# --- 配置 ---
MODEL_PATH="$HOME/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500"
MODEL_BASE="liuhaotian/llava-v1.5-7b"
EVAL_FILE="/home/zyy/AndroidControl/android_control_test.json"
# IMAGE_ROOT="/home/zyy/AndroidControl/android_control_images"
IMAGE_ROOT="/home/zyy/AndroidControl/android_control_images1"
SAVE_DIR="/data/zyy/LLaVA/evaluation/android_control" 
YOLO_MODEL_PATH="/data/zyy/LLaVA/checkpoints/yolov/yolov8n-seg.pt" 

# --- 评估参数 ---
EVAL_TYPE="high" # 或 "low"
TEMPERATURE=0.0
SEED=42

# 为每个阶段创建带有时间戳的唯一输出文件
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ANSWERS_FILE="$SAVE_DIR/answers-android-control-object-only-${EVAL_TYPE}-${TIMESTAMP}.jsonl"

# 确保输出目录存在
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="5"
export LD_LIBRARY_PATH=""

# --model-path $MODEL_PATH \
# --model-base $MODEL_BASE \

# --- 运行评估 ---
echo "--- Running LLaVA AndroidControl Evaluation (Object-Only Method) ---"
python llava/eval/llava_android_control.py \
    --model-path $MODEL_BASE \
    --eval-file "$EVAL_FILE" \
    --image-root "$IMAGE_ROOT" \
    --output-dir "$SAVE_DIR" \
    --eval-type "$EVAL_TYPE" \
    --answers-file "$ANSWERS_FILE" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --dataset "android_control" \
    --cache-mode "read-only" \
    --method-type "object-only" \
    --conv-mode "vicuna_v1"
    
if [ $? -ne 0 ]; then echo "Error in object-only evaluation. Exiting."; exit 1; fi

echo "--- Object-only evaluation completed successfully ---"