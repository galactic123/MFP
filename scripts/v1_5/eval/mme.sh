#!/bin/bash
# CUDA_VISIBLE_DEVICES=7 \
# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-v1.5-7b-newfuse-seedzeros \
#     --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder /opt/project/p/share_data_p_fd/sjj_data/eval2/MME_Benchmark_release_version/MME_Benchmark \
#     --answers-file ./playground/data/eval/MME/answers/llava-v1.5-7b-newfuse-seedzeros.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# cd ./playground/data/eval/MME

python scripts/convert_answer_to_mme.py --experiment llava-v1.5-7b-newfuse-seedzeros

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-newfuse-seedzeros
