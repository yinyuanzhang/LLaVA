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
SIMILARITY_THRESHOLD=0.2

# Window sizes to test
WINDOW_SIZES=(84)

# 为实验创建带有时间戳的唯一输出文件
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# 确保输出目录存在
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="6"
export LD_LIBRARY_PATH=""

echo "=========================================="
echo "Android Control Window Size Ablation Study"
echo "Testing window sizes: ${WINDOW_SIZES[@]}"
echo "Extractor: ${EXTRACTOR_TYPE}"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

# 遍历所有window_size值
for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing Window Size: ${WINDOW_SIZE}"
    echo "=========================================="

    # --- Step 1: Write-only 阶段: 构建缓存 ---
    echo "Step 1: Building cache with WRITE-ONLY mode for window_size=${WINDOW_SIZE}..."

    ANSWERS_FILE_WRITE="$SAVE_DIR/answers-android-control-window-${WINDOW_SIZE}-${EXTRACTOR_TYPE}-write-${EVAL_TYPE}-${TIMESTAMP}.jsonl"

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
        --query-key-extractor-type "$EXTRACTOR_TYPE" \
        --similarity-threshold "$SIMILARITY_THRESHOLD" \
        --window-size "$WINDOW_SIZE"

    if [ $? -ne 0 ]; then
        echo "Error: Write-only mode failed for window_size=${WINDOW_SIZE}. Skipping to next window size."
        continue
    fi

    echo "Cache building completed for window_size=${WINDOW_SIZE}."

    # --- Step 2: Read-Load 阶段: 使用缓存推理 ---
    echo "Step 2: Running inference with READ-LOAD mode for window_size=${WINDOW_SIZE}..."

    ANSWERS_FILE_READ="$SAVE_DIR/answers-android-control-window-${WINDOW_SIZE}-${EXTRACTOR_TYPE}-read-${EVAL_TYPE}-${TIMESTAMP}.jsonl"

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
        --query-key-extractor-type "$EXTRACTOR_TYPE" \
        --similarity-threshold "$SIMILARITY_THRESHOLD" \
        --window-size "$WINDOW_SIZE" \
        --is-flexible-route

    if [ $? -ne 0 ]; then
        echo "Warning: Read-load mode failed for window_size=${WINDOW_SIZE}."
    else
        echo "Inference completed for window_size=${WINDOW_SIZE}. Results saved to $ANSWERS_FILE_READ"
    fi

    echo "Completed processing for window_size=${WINDOW_SIZE}"
    echo ""
done

echo "=========================================="
echo "All window sizes completed!"
echo "Results are saved with timestamp: ${TIMESTAMP}"
echo "=========================================="

# --- Summary ---
echo ""
echo "=========================================="
echo "SUMMARY: Generated result files"
echo "=========================================="

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    ANSWERS_FILE_WRITE="$SAVE_DIR/answers-android-control-window-${WINDOW_SIZE}-${EXTRACTOR_TYPE}-write-${EVAL_TYPE}-${TIMESTAMP}.jsonl"
    ANSWERS_FILE_READ="$SAVE_DIR/answers-android-control-window-${WINDOW_SIZE}-${EXTRACTOR_TYPE}-read-${EVAL_TYPE}-${TIMESTAMP}.jsonl"

    echo ""
    echo "Window Size ${WINDOW_SIZE}:"

    if [ -f "$ANSWERS_FILE_WRITE" ]; then
        echo "  ✓ Write cache: $(basename $ANSWERS_FILE_WRITE)"
    else
        echo "  ✗ Write cache: FAILED"
    fi

    if [ -f "$ANSWERS_FILE_READ" ]; then
        echo "  ✓ Read results: $(basename $ANSWERS_FILE_READ)"
    else
        echo "  ✗ Read results: FAILED"
    fi
done

echo ""
echo "=========================================="
echo "Android Control Window Size Ablation Study Completed!"
echo "=========================================="