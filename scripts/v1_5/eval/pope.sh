# # # !/bin/bash
ps -ef | grep train_gpu | awk '{print $2}' | xargs kill -9

CUDA_VISIBLE_DEVICES=3 \
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-newfuse-vicuna-clip-nohighs\
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /opt/project/p/share_data_p_fd/sjj_data/root/downcache/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b-newfuse-vicuna-clip-nohighs.jsonl \
    --temperature 0.2 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /opt/project/p/share_data_p_fd/sjj_data/root/downcache/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b-newfuse-vicuna-clip-nohighs.jsonl
