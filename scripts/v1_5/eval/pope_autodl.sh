#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"

python -m llava.eval.model_vqa_loader \
    --model-path ~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500 \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/pope/val2014 \
    --answers-file $AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-task-lora-window-16-5500.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --cache-load-way write-only \
    --dataset default_dataset

python llava/eval/eval_pope.py \
    --annotation-dir $AUTO_DL_TMP/playground/data/eval/pope/coco \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-task-lora-window-16-5500.jsonl

