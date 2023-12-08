export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=0 finetune-1gpu.py \
    --base_model /home/arda/llm_models/Llama-2-7b-chat-hf \
    --data-path /home/arda/shane/Platypus/Open-Platypus \
    --output_dir ./llama2-platypus-7b \
    --batch_size 16 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 0.0004 \
    --cutoff_len 512 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --gradient_checkpointing True \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --warmup_steps 100
