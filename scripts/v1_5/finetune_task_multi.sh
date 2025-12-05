# export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
#!/bin/bash
ps -ef | grep train_gpu | awk '{print $2}' | xargs kill -9
# export NCCL_P2P_DISIBLE=1 
OUTPUT_DIR='./checkpoints/llava-v1.5-7b-mfp'
THETA_STATE=True


deepspeed  --hostfile ./scripts/v1_5/hostfile \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7b-newfuse-seedzero \
    --version v1 \
    --data_path path/to/data/llava_v1_5_mix665k.json \
    --image_folder path/to/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
    --theta_state ${THETA_STATE} \
2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"