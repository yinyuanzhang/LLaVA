#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"

python -m llava.eval.model_vqa_loader \
    --model-path ~/.cache/huggingface/hub/models--imagecache--llava-v1.5-7b-lora-noprefusion \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $AUTO_DL_TMP/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file $AUTO_DL_TMP/playground/data/eval/MME/answers/llava-v1.5-7b-lora-noprefusion.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd $AUTO_DL_TMP/playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-lora-noprefusion

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-lora-noprefusion
