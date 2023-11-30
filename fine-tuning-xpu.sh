#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export WANDB_MODE=disabled

echo "Run starts at: $(date '+%Y-%m-%d %H:%M:%S')"

python finetune-xpu.py \
    --base_model /mnt/disk1/models/Llama-2-7b-chat-hf \
    --data-path /home/arda/shane/Platypus/Open-Platypus \
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
    --gradient_checkpointing True \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --seq_min 3000 \
    --seq_max 3200 \
    --warmup_steps 100

echo "Run ends at: $(date '+%Y-%m-%d %H:%M:%S')"

