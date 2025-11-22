#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"
export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES="7"

ANSWERS_FILE="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-native-${EVAL_TYPE}-${TIMESTAMP}.jsonl"


python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/pope/val2014 \
    --answers-file "$ANSWERS_FILE" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --cache-mode read-only \
    --method-type native \
    --dataset pope

python llava/eval/eval_pope.py \
    --annotation-dir $AUTO_DL_TMP/playground/data/eval/pope/coco \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $ANSWERS_FILE



ANSWERS_FILE="$AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-native-${EVAL_TYPE}-${TIMESTAMP}.jsonl"

python -m llava.eval.model_vqa_loader \
    --model-path ~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500 \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/pope/val2014 \
    --answers-file "$ANSWERS_FILE" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --cache-mode read-only \
    --method-type native \
    --dataset pope

python llava/eval/eval_pope.py \
    --annotation-dir $AUTO_DL_TMP/playground/data/eval/pope/coco \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $ANSWERS_FILE