#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-newfuse-seedzeros \
    --question-file /opt/project/p/share_data_p_fd/sjj_data/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /opt/project/p/share_data_p_fd/sjj_data/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-newfuse-seedzeros.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-newfuse-seedzeros.jsonl
