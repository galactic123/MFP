ps -ef | grep train_gpu | awk '{print $2}' | xargs kill -9
CUDA_VISIBLE_DEVICES=0 \
python llava/eval/eval_chairs.py \
--model llava-v1.5-7b-newfuse-seedzeros-time \
--model_path ./checkpoints/llava-v1.5-7b-newfuse-seedzeros \
--data-path /opt/project/n/mm_school/fudan/project/user/sjj/coco2014/minival \
--temperature 0.2

python llava/eval/chair.py \
--cap_file ./playground/data/eval/chairs/lllava-v1.5-7b-newfuse-vicuna-clip-nohighs-time/chair_eval.jsonl