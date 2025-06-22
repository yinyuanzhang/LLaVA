#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-task-lora-window-16-5500"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$AUTO_DL_TMP/playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ~/.cache/huggingface/hub/llava-v1.5-7b-task-lora-window-16/checkpoint-5500 \
        --model-base liuhaotian/llava-v1.5-7b \
        --question-file $AUTO_DL_TMP/playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder $AUTO_DL_TMP/playground/data/eval/gqa/data/images \
        --answers-file $AUTO_DL_TMP/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$AUTO_DL_TMP/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $AUTO_DL_TMP/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
