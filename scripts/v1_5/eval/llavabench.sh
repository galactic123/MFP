#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python -m llava.eval.model_vqa \
    --model-path  ./checkpoints/llava-v1.5-7b-newfuse-seedzeros \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder /opt/project/p/share_data_p_fd/sjj_data/eval2/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b-newfuse-seedzeros.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

# python llava/eval/eval_gpt_review_bench.py \
#     --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
#     --rule llava/eval/table/rule.json \
#     --answer-list \
#         playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
#         playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b-newfuse-seedzeros.jsonl \
#     --output \
#         playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b-newfuse-seedzeros.jsonl

# python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b-newfuse-seedzeros.jsonl
