#!/bin/bash

# --- 配置 ---
MODEL_PATH="$HOME/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500"
MODEL_BASE="liuhaotian/llava-v1.5-7b"
EVAL_FILE="/home/zyy/AndroidControl/android_control_test.json"
FUZZY_IMAGE_ROOT="/home/zyy/AndroidControl/android_control_images2"
IMAGE_ROOT="/home/zyy/AndroidControl/android_control_images1"
SAVE_DIR="/data/zyy/LLaVA/evaluation/android_control"
YOLO_MODEL_PATH="/data/zyy/LLaVA/checkpoints/yolov/yolov8n-seg.pt"
EXTRACTOR_TYPE="resnet18"

# --- 评估参数 ---
EVAL_TYPE="high" # 或 "low"
TEMPERATURE=0.0
SEED=42

# Similarity thresholds to test
THRESHOLDS=(0.2)

# 为每个阶段创建带有时间戳的唯一输出文件
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ANSWERS_FILE_WRITE="$SAVE_DIR/answers-android-control-cacheblend-${EXTRACTOR_TYPE}-write-${EVAL_TYPE}-${TIMESTAMP}.jsonl"

# 确保输出目录存在
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="5"
export LD_LIBRARY_PATH=""

# --- Step 1: Write-only 阶段: 构建缓存 ---
echo "=========================================="
echo "Step 1: Building CacheBlend cache with WRITE-ONLY mode using ${EXTRACTOR_TYPE}..."
echo "=========================================="

python llava/eval/llava_android_control.py \
    --model-path $MODEL_BASE \
    --eval-file "$EVAL_FILE" \
    --image-root "$FUZZY_IMAGE_ROOT" \
    --output-dir "$SAVE_DIR" \
    --eval-type "$EVAL_TYPE" \
    --answers-file "$ANSWERS_FILE_WRITE" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --dataset "android_control" \
    --cache-mode "write-only" \
    --method-type "cacheblend" \
    --conv-mode "vicuna_v1" \
    --use-lightweight-query-key \
    --query-key-extractor-type "$EXTRACTOR_TYPE" \
    --similarity-threshold 0.1

if [ $? -ne 0 ]; then
    echo "Error: CacheBlend write-only mode failed. Cannot proceed with read-load mode."
    exit 1
fi

echo "CacheBlend cache building completed."
echo ""

# --- Step 2: Read-Load Mode with Multiple Thresholds ---
for THRESHOLD in "${THRESHOLDS[@]}"; do
    echo "=========================================="
    echo "Step 2: Running CacheBlend inference with READ-LOAD mode using ${EXTRACTOR_TYPE} and threshold ${THRESHOLD}..."
    echo "=========================================="

    ANSWERS_FILE_READ="$SAVE_DIR/answers-android-control-cacheblend-${EXTRACTOR_TYPE}-read-threshold-${THRESHOLD}-${EVAL_TYPE}-${TIMESTAMP}.jsonl"

    python llava/eval/llava_android_control.py \
        --model-path $MODEL_BASE \
        --eval-file "$EVAL_FILE" \
        --image-root "$IMAGE_ROOT" \
        --output-dir "$SAVE_DIR" \
        --eval-type "$EVAL_TYPE" \
        --answers-file "$ANSWERS_FILE_READ" \
        --temperature "$TEMPERATURE" \
        --seed "$SEED" \
        --dataset "android_control" \
        --cache-mode "read-load" \
        --method-type "cacheblend" \
        --conv-mode "vicuna_v1" \
        --use-lightweight-query-key \
        --query-key-extractor-type "$EXTRACTOR_TYPE" \
        --similarity-threshold "$THRESHOLD"
        # --is-flexible-route

    if [ $? -ne 0 ]; then
        echo "Warning: CacheBlend read-load mode failed for threshold ${THRESHOLD}. Skipping."
        continue
    fi

    echo "CacheBlend inference completed for threshold ${THRESHOLD}. Results saved to $ANSWERS_FILE_READ"
    echo ""
done

echo "=========================================="
echo "All CacheBlend steps completed for ${EXTRACTOR_TYPE} with all thresholds!"
echo "Results are saved with timestamp: ${TIMESTAMP}"
echo "=========================================="

# --- Summary ---
echo ""
echo "=========================================="
echo "SUMMARY: Generated CacheBlend result files"
echo "=========================================="
echo "Write cache file: $ANSWERS_FILE_WRITE"
echo ""
echo "Read-load result files:"
for THRESHOLD in "${THRESHOLDS[@]}"; do
    ANSWERS_FILE_READ="$SAVE_DIR/answers-android-control-cacheblend-${EXTRACTOR_TYPE}-read-threshold-${THRESHOLD}-${EVAL_TYPE}-${TIMESTAMP}.jsonl"
    if [ -f "$ANSWERS_FILE_READ" ]; then
        echo "  - Threshold ${THRESHOLD}: $(basename $ANSWERS_FILE_READ)"
    fi
done
echo "=========================================="
