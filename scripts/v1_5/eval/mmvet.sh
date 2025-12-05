#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/llava-v1.5-7b-newfuse-seedzeros \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /opt/project/p/share_data_p_fd/sjj_data/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/lava-v1.5-7b-newfuse-seedzeros.jsonl \
    --temperature 0.2 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/lava-v1.5-7b-newfuse-seedzeros.jsonl \
    --dst ./playground/data/eval/mm-vet/results/lava-v1.5-7b-newfuse-seedzeros.json

