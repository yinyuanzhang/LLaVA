#!/bin/bash

# --- Configuration ---
AUTO_DL_TMP="$HOME/autodl-tmp"
export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES="7"

# Model Configuration
MODEL_PATH="~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500"
MODEL_BASE="liuhaotian/llava-v1.5-7b"
CONV_MODE="vicuna_v1"
EXTRACTOR_TYPE="resnet50"

# POPE Dataset Paths
FUZZY_IMAGE_FOLDER="$AUTO_DL_TMP/playground/data/eval/pope/val2014_bagle"
IMAGE_FOLDER="$AUTO_DL_TMP/playground/data/eval/pope/val2014"
QUESTION_FILE="$AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl"
POPE_ANNOTATION_DIR="$AUTO_DL_TMP/playground/data/eval/pope/coco"

# Similarity thresholds to test
THRESHOLDS=(0.0 0.1 0.2 0.3)

# --- Step 1: Write-Only Mode (Build Cache) ---
echo "=========================================="
echo "Step 1: Building cache with WRITE-ONLY mode using ${EXTRACTOR_TYPE}..."
echo "=========================================="

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ANSWERS_FILE_WRITE="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-segmentation-cache-${EXTRACTOR_TYPE}-write-${TIMESTAMP}.jsonl"

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
    --use-lightweight-query-key \
    --query-key-extractor-type "$EXTRACTOR_TYPE" \
    --similarity-threshold 0.1

if [ ! -f "$ANSWERS_FILE_WRITE" ] || [ ! -s "$ANSWERS_FILE_WRITE" ]; then
    echo "Error: Write-only mode failed. Cannot proceed with read-load mode."
    exit 1
fi

echo "Cache building completed."
echo ""

# --- Step 2: Read-Load Mode with Multiple Thresholds ---
for THRESHOLD in "${THRESHOLDS[@]}"; do
    echo "=========================================="
    echo "Step 2: Running inference with READ-LOAD mode using ${EXTRACTOR_TYPE} and threshold ${THRESHOLD}..."
    echo "=========================================="

    ANSWERS_FILE_READ="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-segmentation-cache-${EXTRACTOR_TYPE}-read-threshold-${THRESHOLD}-${TIMESTAMP}.jsonl"

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
        --use-lightweight-query-key \
        --query-key-extractor-type "$EXTRACTOR_TYPE" \
        --similarity-threshold "$THRESHOLD"

    if [ ! -f "$ANSWERS_FILE_READ" ] || [ ! -s "$ANSWERS_FILE_READ" ]; then
        echo "Warning: Read-load mode failed for threshold ${THRESHOLD}. Skipping evaluation."
        continue
    fi

    echo "Inference completed for threshold ${THRESHOLD}. Results saved to $ANSWERS_FILE_READ"
    echo ""

    # --- Step 3: Evaluation for this threshold ---
    echo "=========================================="
    echo "Step 3: Running POPE evaluation for threshold ${THRESHOLD}..."
    echo "=========================================="

    python llava/eval/eval_pope.py \
        --annotation-dir "$POPE_ANNOTATION_DIR" \
        --question-file "$QUESTION_FILE" \
        --result-file "$ANSWERS_FILE_READ"

    echo ""
    echo "Evaluation completed for threshold ${THRESHOLD}!"
    echo ""
done

echo "=========================================="
echo "All steps completed for ${EXTRACTOR_TYPE} with all thresholds!"
echo "Results are saved with timestamp: ${TIMESTAMP}"
echo "=========================================="

# --- Summary ---
echo ""
echo "=========================================="
echo "SUMMARY: Generated result files"
echo "=========================================="
echo "Write cache file: $ANSWERS_FILE_WRITE"
echo ""
echo "Read-load result files:"
for THRESHOLD in "${THRESHOLDS[@]}"; do
    ANSWERS_FILE_READ="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-segmentation-cache-${EXTRACTOR_TYPE}-read-threshold-${THRESHOLD}-${TIMESTAMP}.jsonl"
    if [ -f "$ANSWERS_FILE_READ" ]; then
        echo "  - Threshold ${THRESHOLD}: $(basename $ANSWERS_FILE_READ)"
    fi
done
echo "=========================================="