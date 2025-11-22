#!/bin/bash

# --- 配置 ---
MODEL_PATH="$HOME/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500"
MODEL_BASE="liuhaotian/llava-v1.5-7b"
EVAL_FILE="/home/zyy/AndroidControl/android_control_test.json"
FUZZY_IMAGE_ROOT="/home/zyy/AndroidControl/android_control_images2"
IMAGE_ROOT="/home/zyy/AndroidControl/android_control_images1"
SAVE_DIR="/data/zyy/LLaVA/evaluation/android_control" 
YOLO_MODEL_PATH="/data/zyy/LLaVA/checkpoints/yolov/yolov8n-seg.pt" 

# --- 评估参数 ---
EVAL_TYPE="high" # 或 "low"
TEMPERATURE=0.0
SEED=42

# 为每个阶段创建带有时间戳的唯一输出文件
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ANSWERS_FILE_WRITE="$SAVE_DIR/answers-android-control-segmentation-cache-write-${EVAL_TYPE}-${TIMESTAMP}.jsonl"
ANSWERS_FILE_READ="$SAVE_DIR/answers-android-control-segmentation-cache-read-load-${EVAL_TYPE}-${TIMESTAMP}.jsonl"

# 确保输出目录存在
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="6"
export LD_LIBRARY_PATH=""

# --- Write-only 阶段: 构建缓存 ---
echo "--- Step 1: Building cache with segmentation-cache (write-only) ---"
python llava/eval/llava_android_control.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --eval-file "$EVAL_FILE" \
    --image-root "$FUZZY_IMAGE_ROOT" \
    --output-dir "$SAVE_DIR" \
    --eval-type "$EVAL_TYPE" \
    --answers-file "$ANSWERS_FILE_WRITE" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --dataset "android_control" \
    --cache-mode "write-only" \
    --method-type "segmentation-cache" \
    --conv-mode "vicuna_v1" \
    --use-lightweight-query-key \
    --query-key-extractor-type "resnet18" \
    --similarity-threshold 0.1

if [ $? -ne 0 ]; then echo "Error in write-only phase. Exiting."; exit 1; fi

# --- Read-load 阶段: 使用缓存 ---
echo "--- Step 2: Using cache with segmentation-cache (read-load) ---"
python llava/eval/llava_android_control.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --eval-file "$EVAL_FILE" \
    --image-root "$IMAGE_ROOT" \
    --output-dir "$SAVE_DIR" \
    --eval-type "$EVAL_TYPE" \
    --answers-file "$ANSWERS_FILE_READ" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --dataset "android_control" \
    --cache-mode "read-load" \
    --method-type "segmentation-cache" \
    --conv-mode "vicuna_v1" \
    --use-lightweight-query-key \
    --query-key-extractor-type "resnet18" \
    --similarity-threshold 0.2 \
    --is-flexible-route    
    
if [ $? -ne 0 ]; then echo "Error in read-load phase. Exiting."; exit 1; fi

echo "--- Segmentation-cache evaluation completed successfully ---"