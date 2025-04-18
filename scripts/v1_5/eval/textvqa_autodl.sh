#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"
python -m llava.eval.model_vqa_loader \
    --model-path ~/.cache/huggingface/hub/models--imagecache--llava-v1.5-7b-lora-noprefusion \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $AUTO_DL_TMP/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/textvqa/train_images \
    --answers-file $AUTO_DL_TMP/playground/data/eval/textvqa/answers/llava-v1.5-7b-lora-noprefusion.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file $AUTO_DL_TMP/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $AUTO_DL_TMP/playground/data/eval/textvqa/answers/llava-v1.5-7b-lora-noprefusion.jsonl


# python -m llava.eval.model_vqa_loader \
#     --model-path ~/.cache/huggingface/hub/models--imagecache--llava-v1.5-7b-lora-noprefusion2 \
#     --model-base liuhaotian/llava-v1.5-7b \
#     --question-file $AUTO_DL_TMP/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder $AUTO_DL_TMP/playground/data/eval/textvqa/train_images \
#     --answers-file $AUTO_DL_TMP/playground/data/eval/textvqa/answers/llava-v1.5-7b-task-lora.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file $AUTO_DL_TMP/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file $AUTO_DL_TMP/playground/data/eval/textvqa/answers/llava-v1.5-7b-task-lora.jsonl