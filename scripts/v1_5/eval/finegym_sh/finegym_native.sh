#!/bin/bash

# --- Configuration ---
MODEL_PATH="$HOME/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500"
MODEL_BASE="liuhaotian/llava-v1.5-7b"
MODEL_NAME=$(basename "$MODEL_PATH")

# FineGym Dataset Paths
DATA_ROOT="/home/zyy/autodl-tmp/playground/data/eval/fineGym"
IMAGE_ROOT="$DATA_ROOT/extracted_frames"
OUTPUT_DIR="/data/zyy/LLaVA/evaluation/finegym"
VAL_ELEMENT_FILE="$DATA_ROOT/gym99_val_element.txt"
CATEGORIES_FILE="$DATA_ROOT/gym99_categories.txt"
SET_CATEGORIES_FILE="$DATA_ROOT/set_categories.txt"
YOLO_MODEL_PATH="/data/zyy/LLaVA/checkpoints/yolov/yolov8n-seg.pt"

# Parameters
N_FRAMES=5
TEMPERATURE=0.0
SEED=42
DATASET_NAME="finegym"

# Generate unique output file name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ANSWERS_FILE="$OUTPUT_DIR/finegym-llava-native-${TIMESTAMP}.json"

export CUDA_VISIBLE_DEVICES="2"
export LD_LIBRARY_PATH=""

# --- Create Output Directory ---
mkdir -p "$OUTPUT_DIR"

# --- Run LLaVA Inference (Native Method) ---
echo "Starting LLaVA inference on FineGym dataset using NATIVE method..."
python llava/eval/llava_finegym.py \
    --model-path "$MODEL_BASE" \
    --image-folder "$IMAGE_ROOT" \
    --answers-file "$ANSWERS_FILE" \
    --val-element-txt "$VAL_ELEMENT_FILE" \
    --categories-txt "$CATEGORIES_FILE" \
    --set-categories-txt "$SET_CATEGORIES_FILE" \
    --n-frames "$N_FRAMES" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --dataset "$DATASET_NAME" \
    --method-type "native" \
    --cache-mode "read-only" \
    --conv-mode "vicuna_v1" \
    --yolo-model-path "$YOLO_MODEL_PATH"

# --- Check if inference was successful ---
if [ ! -f "$ANSWERS_FILE" ] || [ ! -s "$ANSWERS_FILE" ]; then
    echo "Error: Inference output file not found or is empty."
    exit 1
fi

echo "FineGym evaluation completed. Results saved to $ANSWERS_FILE"