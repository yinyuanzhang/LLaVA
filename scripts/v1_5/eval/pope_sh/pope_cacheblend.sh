#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"
export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES="5"


# Write-only phase: Build cache using val2014_bagle
python -m llava.eval.model_vqa_loader \
    --model-path ~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500 \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/pope/val2014_bagle \
    --answers-file $AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-segmentation-cache-write.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --cache-mode write-only \
    --method-type cacheblend \
    --dataset pope \
    --use-lightweight-query-key \
    --query-key-extractor-type resnet18 \
    --similarity-threshold 0.1

# Read-load phase: Use cache with val2014
python -m llava.eval.model_vqa_loader \
    --model-path ~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500 \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/pope/val2014 \
    --answers-file $AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-segmentation-cache-read-load.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --cache-mode read-load \
    --method-type cacheblend \
    --dataset pope \
    --use-lightweight-query-key \
    --query-key-extractor-type resnet18 \
    --similarity-threshold 0.1


python llava/eval/eval_pope.py \
    --annotation-dir $AUTO_DL_TMP/playground/data/eval/pope/coco \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-segmentation-cache-read-load.jsonl