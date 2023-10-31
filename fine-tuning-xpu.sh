#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

python finetune.py \
    --base_model /mnt/disk1/models/Llama-2-7b-chat-hf \
    --data-path garage-bAInd/Open-Platypus \
    --output_dir ./llama2-platypus-7b \
    --batch_size 16 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 0.0004 \
    --cutoff_len 4096 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --warmup_steps 100
