#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ~/.cache/huggingface/hub/models--imagecache--llava-v1.5-7b-lora-noprefusion \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $AUTO_DL_TMP/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file $AUTO_DL_TMP/playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b-lora-noprefusion.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $AUTO_DL_TMP/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir $AUTO_DL_TMP/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir $AUTO_DL_TMP/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b-lora-noprefusion
