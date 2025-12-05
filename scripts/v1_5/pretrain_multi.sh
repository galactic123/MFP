# export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

#!/bin/bash
ps -ef | grep train_gpu | awk '{print $2}' | xargs kill -9
OUTPUT_DIR='./checkpoints/llava-v1.5-7b-mfp-pretrain'

deepspeed --hostfile ./scripts/v1_5/hostfile llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path path/to/model/vicuna-7b-v1.5 \
    --version plain \
    --data_path path/to/data/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 20 \
    --lazy_preprocess True \
    --report_to "tensorboard" \
2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
