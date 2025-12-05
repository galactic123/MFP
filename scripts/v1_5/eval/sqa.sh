!/bin/bash
CUDA_VISIBLE_DEVICES=5 \
python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-v1.5-7b-newfuse-seedzeros \
    --question-file /opt/project/p/share_data_p_fd/sjj_data/scienceqa/llava_test_CQM-A.json\
    --image-folder /opt/project/p/share_data_p_fd/sjj_data/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-newfuse-seedzeros.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-newfuse-seedzeros.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-newfuse-seedzeros_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-newfuse-seedzeros_result.json
