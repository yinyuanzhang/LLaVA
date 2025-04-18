#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"
python -m llava.eval.model_vqa_science \
    --model-path ~/.cache/huggingface/hub/models--imagecache--llava-v1.5-7b-lora-noprefusion \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $AUTO_DL_TMP/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $AUTO_DL_TMP/playground/data/eval/scienceqa/images/test \
    --answers-file $AUTO_DL_TMP/playground/data/eval/scienceqa/answers/llava-v1.5-7b-lora-noprefusion.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir $AUTO_DL_TMP/playground/data/eval/scienceqa \
    --result-file $AUTO_DL_TMP/playground/data/eval/scienceqa/answers/llava-v1.5-7b-lora-noprefusion.jsonl \
    --output-file $AUTO_DL_TMP/playground/data/eval/scienceqa/answers/llava-v1.5-7b-lora-noprefusion_output.jsonl \
    --output-result $AUTO_DL_TMP/playground/data/eval/scienceqa/answers/llava-v1.5-7b-lora-noprefusion_result.json
