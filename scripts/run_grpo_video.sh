
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG_Activieynet

# mkdir -p /outputs/$WANDB_PROJECT/$WANDB_NAME

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=outputs_video

export DEBUG_MODE="true"
export LOG_PATH="./qwen2.5_7b_vl_tg_video.txt"

# --report_to wandb \
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12361" \
    src/open_r1/grpo_video.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path /share/wy/Video/R1-Video-Stable/open-r1-multimodal/Qwen2.5-VL_outputs_video_slid_flash_attention_2_activitynet_continue/checkpoint-200 \
    --preprocessed_data_path /share/wy/Video/R1-Video-Stable/open-r1-multimodal/activitynet_preprocessed_data_maxpix_3584 \
    --train_data_path /share/wy/Video/Video_Grounding/annotations/train.json \
    --eval_data_path /share/wy/Video/Video_Grounding/annotations/val_2.json \
    --video_folder /share/zsp/datasets/motion_raw/v2/activitynet/images \
    --dataset_name xxx \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to wandb \
    --save_steps 50 \
    --save_only_model true
