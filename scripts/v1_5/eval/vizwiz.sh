#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-newfuse-seedzeros \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /opt/project/p/share_data_p_fd/sjj_data/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b-newfuse-seedzeros.jsonl \
    --temperature 0.2 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b-newfuse-seedzeros.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-7b-newfuse-seedzeros.json
