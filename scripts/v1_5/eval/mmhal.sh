ps -ef | grep train_gpu | awk '{print $2}' | xargs kill -9
CUDA_VISIBLE_DEVICES=0 \
python llava/eval/model_eval_mmhalx.py \
--model_path ./checkpoints/llava-v1.5-7b-newfuse-seedzeros \
--input /opt/project/p/share_data_p_fd/sjj_data/mmhal/response_template_abs.json \
--output ./playground/data/eval/mmhal/mmhal_2.json