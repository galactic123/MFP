# #!/bin/bash

# SPLIT="mmbench_dev_en_20231003"
# CUDA_VISIBLE_DEVICES=0 \
# python -m llava.eval.model_vqa_mmbench \
#     --model-path ./checkpoints/llava-v1.5-7b-newfuse-seedzeros\
#     --question-file /opt/project/n/mm_school/fudan/project/user/sjj/deepfuse/eval_d/mnt/data1/user/fan_xiaoran/data/eval_data/mmbench/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b-newfuse-seedzeros_r23.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /opt/project/n/mm_school/fudan/project/user/sjj/deepfuse/eval_d/mnt/data1/user/fan_xiaoran/data/eval_data/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b-newfuse-seedzeros_r23
