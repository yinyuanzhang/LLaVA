#!/bin/bash

# --- Configuration ---
AUTO_DL_TMP="$HOME/autodl-tmp"
export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES="7"

# Model Configuration
MODEL_PATH="~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500"
MODEL_BASE="liuhaotian/llava-v1.5-7b"
CONV_MODE="vicuna_v1"
EXTRACTOR_TYPE="resnet18"
SIMILARITY_THRESHOLD=0.2

# POPE Dataset Paths
FUZZY_IMAGE_FOLDER="$AUTO_DL_TMP/playground/data/eval/pope/val2014_bagle"
IMAGE_FOLDER="$AUTO_DL_TMP/playground/data/eval/pope/val2014"
QUESTION_FILE="$AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl"
POPE_ANNOTATION_DIR="$AUTO_DL_TMP/playground/data/eval/pope/coco"

# Window sizes to test
WINDOW_SIZES=(84)

# 为实验创建带有时间戳的唯一输出文件
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "=========================================="
echo "POPE Window Size Ablation Study"
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

    # --- Step 1: Write-Only Mode (Build Cache) ---
    echo "Step 1: Building cache with WRITE-ONLY mode for window_size=${WINDOW_SIZE}..."

    ANSWERS_FILE_WRITE="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-window-${WINDOW_SIZE}-${EXTRACTOR_TYPE}-write-${TIMESTAMP}.jsonl"

    mkdir -p $(dirname "$ANSWERS_FILE_WRITE")

    python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --model-base "$MODEL_BASE" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$FUZZY_IMAGE_FOLDER" \
        --answers-file "$ANSWERS_FILE_WRITE" \
        --temperature 0 \
        --conv-mode "$CONV_MODE" \
        --cache-mode write-only \
        --method-type segmentation-cache \
        --dataset pope \
        --query-key-extractor-type "$EXTRACTOR_TYPE" \
        --similarity-threshold "$SIMILARITY_THRESHOLD" \
        --use-lightweight-query-key \
        --window-size "$WINDOW_SIZE"

    if [ ! -f "$ANSWERS_FILE_WRITE" ] || [ ! -s "$ANSWERS_FILE_WRITE" ]; then
        echo "Error: Write-only mode failed for window_size=${WINDOW_SIZE}. Skipping to next window size."
        continue
    fi

    echo "Cache building completed for window_size=${WINDOW_SIZE}."

    # --- Step 2: Read-Load Mode ---
    echo "Step 2: Running inference with READ-LOAD mode for window_size=${WINDOW_SIZE}..."

    ANSWERS_FILE_READ="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-window-${WINDOW_SIZE}-${EXTRACTOR_TYPE}-read-${TIMESTAMP}.jsonl"

    python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --model-base "$MODEL_BASE" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$ANSWERS_FILE_READ" \
        --temperature 0 \
        --conv-mode "$CONV_MODE" \
        --cache-mode read-load \
        --method-type segmentation-cache \
        --dataset pope \
        --query-key-extractor-type "$EXTRACTOR_TYPE" \
        --use-lightweight-query-key \
        --similarity-threshold "$SIMILARITY_THRESHOLD" \
        --window-size "$WINDOW_SIZE" \
        --is-flexible-route

    if [ ! -f "$ANSWERS_FILE_READ" ] || [ ! -s "$ANSWERS_FILE_READ" ]; then
        echo "Warning: Read-load mode failed for window_size=${WINDOW_SIZE}. Skipping evaluation."
        continue
    fi

    echo "Inference completed for window_size=${WINDOW_SIZE}. Results saved to $ANSWERS_FILE_READ"

    # --- Step 3: Evaluation for this window size ---
    echo "Step 3: Running POPE evaluation for window_size=${WINDOW_SIZE}..."

    python llava/eval/eval_pope.py \
        --annotation-dir "$POPE_ANNOTATION_DIR" \
        --question-file "$QUESTION_FILE" \
        --result-file "$ANSWERS_FILE_READ"

    echo ""
    echo "Evaluation completed for window_size=${WINDOW_SIZE}!"
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
    ANSWERS_FILE_WRITE="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-window-${WINDOW_SIZE}-${EXTRACTOR_TYPE}-write-${TIMESTAMP}.jsonl"
    ANSWERS_FILE_READ="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-window-${WINDOW_SIZE}-${EXTRACTOR_TYPE}-read-${TIMESTAMP}.jsonl"

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
echo "POPE Window Size Ablation Study Completed!"
echo "=========================================="