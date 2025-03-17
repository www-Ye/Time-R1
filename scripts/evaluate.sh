
export CUDA_VISIBLE_DEVICES=0

# MODEL_BASE=mllm/Qwen2.5-VL-7B-Instruct
MODEL_BASE=./outputs_video

python evaluate.py \
     --model_base $MODEL_BASE \
     --dataset charades \
     --checkpoint_dir ckpt_charades