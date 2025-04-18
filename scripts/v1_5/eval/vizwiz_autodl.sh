#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"

python -m llava.eval.model_vqa_loader \
    --model-path ~/.cache/huggingface/hub/models--imagecache--llava-v1.5-7b-lora-noprefusion \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $AUTO_DL_TMP/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/vizwiz/test \
    --answers-file $AUTO_DL_TMP/playground/data/eval/vizwiz/answers/llava-v1.5-7b-lora-noprefusion.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $AUTO_DL_TMP/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file $AUTO_DL_TMP/playground/data/eval/vizwiz/answers/llava-v1.5-7b-lora-noprefusion.jsonl \
    --result-upload-file $AUTO_DL_TMP/playground/data/eval/vizwiz/answers_upload/llava-v1.5-7b-lora-noprefusion.json
