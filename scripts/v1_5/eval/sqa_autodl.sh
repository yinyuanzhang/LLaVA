#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"
python -m llava.eval.model_vqa_science \
    --model-path ~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500 \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file $AUTO_DL_TMP/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $AUTO_DL_TMP/playground/data/eval/scienceqa/images/test \
    --answers-file $AUTO_DL_TMP/playground/data/eval/scienceqa/answers/llava-v1.5-7b-task-lora-window-16-5500.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir $AUTO_DL_TMP/playground/data/eval/scienceqa \
    --result-file $AUTO_DL_TMP/playground/data/eval/scienceqa/answers/llava-v1.5-7b-task-lora-window-16-5500.jsonl \
    --output-file $AUTO_DL_TMP/playground/data/eval/scienceqa/answers/llava-v1.5-7b-task-lora-window-16-5500_output.jsonl \
    --output-result $AUTO_DL_TMP/playground/data/eval/scienceqa/answers/llava-v1.5-7b-task-lora-window-16-5500_result.json
