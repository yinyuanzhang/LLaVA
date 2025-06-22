#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"

python -m llava.eval.model_vqa_loader \
    --model-path ~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500 \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file $AUTO_DL_TMP/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/vizwiz/test \
    --answers-file $AUTO_DL_TMP/playground/data/eval/vizwiz/answers/llava-v1.5-7b-task-lora-window-16-5500.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $AUTO_DL_TMP/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file $AUTO_DL_TMP/playground/data/eval/vizwiz/answers/llava-v1.5-7b-task-lora-window-16-5500.jsonl \
    --result-upload-file $AUTO_DL_TMP/playground/data/eval/vizwiz/answers_upload/llava-v1.5-7b-task-lora-window-16-5500.json
