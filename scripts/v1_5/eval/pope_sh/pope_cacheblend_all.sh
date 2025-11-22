#!/bin/bash

# --- Configuration ---
AUTO_DL_TMP="$HOME/autodl-tmp"
export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES="2"

# Model Configuration
MODEL_PATH="~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500"
MODEL_BASE="liuhaotian/llava-v1.5-7b"
CONV_MODE="vicuna_v1"
EXTRACTOR_TYPE="resnet18"

# POPE Dataset Paths
FUZZY_IMAGE_FOLDER="$AUTO_DL_TMP/playground/data/eval/pope/val2014_bagle"
IMAGE_FOLDER="$AUTO_DL_TMP/playground/data/eval/pope/val2014"
QUESTION_FILE="$AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl"
POPE_ANNOTATION_DIR="$AUTO_DL_TMP/playground/data/eval/pope/coco"

# Similarity thresholds to test
THRESHOLDS=(0.0 0.1 0.2)

# --- Step 1: Write-Only Mode (Build Cache) ---
echo "=========================================="
echo "Step 1: Building CacheBlend cache with WRITE-ONLY mode using ${EXTRACTOR_TYPE}..."
echo "=========================================="

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ANSWERS_FILE_WRITE="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-cacheblend-${EXTRACTOR_TYPE}-write-${TIMESTAMP}.jsonl"

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
    --method-type cacheblend \
    --dataset pope \
    --use-lightweight-query-key \
    --query-key-extractor-type "$EXTRACTOR_TYPE" \
    --similarity-threshold 0.1

if [ ! -f "$ANSWERS_FILE_WRITE" ] || [ ! -s "$ANSWERS_FILE_WRITE" ]; then
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

    ANSWERS_FILE_READ="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-cacheblend-${EXTRACTOR_TYPE}-read-threshold-${THRESHOLD}-${TIMESTAMP}.jsonl"

    python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --model-base "$MODEL_BASE" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$ANSWERS_FILE_READ" \
        --temperature 0 \
        --conv-mode "$CONV_MODE" \
        --cache-mode read-load \
        --method-type cacheblend \
        --dataset pope \
        --use-lightweight-query-key \
        --query-key-extractor-type "$EXTRACTOR_TYPE" \
        --similarity-threshold "$THRESHOLD" \
        --is-flexible-route

    if [ ! -f "$ANSWERS_FILE_READ" ] || [ ! -s "$ANSWERS_FILE_READ" ]; then
        echo "Warning: CacheBlend read-load mode failed for threshold ${THRESHOLD}. Skipping evaluation."
        continue
    fi

    echo "CacheBlend inference completed for threshold ${THRESHOLD}. Results saved to $ANSWERS_FILE_READ"
    echo ""

    # --- Step 3: Evaluation for this threshold ---
    echo "=========================================="
    echo "Step 3: Running POPE evaluation for CacheBlend threshold ${THRESHOLD}..."
    echo "=========================================="

    python llava/eval/eval_pope.py \
        --annotation-dir "$POPE_ANNOTATION_DIR" \
        --question-file "$QUESTION_FILE" \
        --result-file "$ANSWERS_FILE_READ"

    echo ""
    echo "CacheBlend evaluation completed for threshold ${THRESHOLD}!"
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
    ANSWERS_FILE_READ="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-cacheblend-${EXTRACTOR_TYPE}-read-threshold-${THRESHOLD}-${TIMESTAMP}.jsonl"
    if [ -f "$ANSWERS_FILE_READ" ]; then
        echo "  - Threshold ${THRESHOLD}: $(basename $ANSWERS_FILE_READ)"
    fi
done
echo "=========================================="

# --- Performance Comparison Summary ---
echo ""
echo "=========================================="
echo "NEXT STEPS: Performance Analysis"
echo "=========================================="
echo "To analyze the performance impact of different thresholds, you can:"
echo ""
echo "1. Compare cache hit rates across thresholds:"
echo "   grep 'cache_hits\\|cache_searches' /path/to/log/files"
echo ""
echo "2. Compare POPE accuracy results:"
echo "   Check the evaluation output above for each threshold"
echo ""
echo "3. Analyze computational savings:"
echo "   Compare 'img_len' values in with_cache vs no_cache statistics"
echo ""
echo "Generated files can be found in:"
echo "  $AUTO_DL_TMP/playground/data/eval/pope/answers/"
echo "  Pattern: llava-v1.5-7b-cacheblend-${EXTRACTOR_TYPE}-*-${TIMESTAMP}.jsonl"
echo "=========================================="