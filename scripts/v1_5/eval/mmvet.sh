#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"

python -m llava.eval.model_vqa \
    --model-path ~/.cache/huggingface/hub/models--imagecache--llava-v1.5-7b-lora-noprefusion \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $AUTO_DL_TMP/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/mm-vet/images \
    --answers-file $AUTO_DL_TMP/playground/data/eval/mm-vet/answers/llava-v1.5-7b-lora-noprefusion.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $AUTO_DL_TMP/playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $AUTO_DL_TMP/playground/data/eval/mm-vet/answers/llava-v1.5-7b-lora-noprefusion.jsonl \
    --dst $AUTO_DL_TMP/playground/data/eval/mm-vet/results/llava-v1.5-7b-lora-noprefusion.json

