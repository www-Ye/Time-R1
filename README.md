# TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM

<div style='display:flex; gap: 0.25rem; '>
  <a href='./TimeZero_TechReport.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
  <a href='https://huggingface.co/wwwyyy/TimeZero-Charades-7B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Charades-blue'></a>
  <a href='https://huggingface.co/wwwyyy/TimeZero-ActivityNet-7B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ActivityNet-blue'></a>
</div>

### Updates

- 2025-03-17: TimeZero initial release! Code and evaluation scripts are now available.
- 2025-03-17: TimeZero achieves SOTA performance on Charades-STA!

### Overview

TimeZero is a reasoning-guided Large Vision-Language Model (LVLM) for Temporal Video Grounding (TVG). It excels at identifying temporal segments within videos that correspond to a given natural language query.  TimeZero achieves this entirely through a reinforcement learning approach that allows the model to reason about video-language relationships *during inference*.

Key Features:

*   **Reinforcement Learning Training:** TimeZero is trained *entirely* using reinforcement learning, enhancing its ability to generate accurate temporal boundaries.
*   **Test-Time Reasoning:** The model exhibits emergent reasoning capabilities during inference, generating a chain of thought to justify its segment predictions.
*   **SOTA Performance:** TimeZero sets a new SOTA on the Charades-STA benchmark.


This README provides an overview of TimeZero, including setup instructions, the training process, and evaluation guidelines.

**Example:**

![image](https://github.com/user-attachments/assets/f5ac9e6b-58f5-41e9-878d-a5ae5045b155)


**Training Visualization:**

![0a466a4bca3bb8d9b2a2af0f15890b4](https://github.com/user-attachments/assets/df1c35f5-8c30-400b-bce6-14e1f766752c)

## Setup

```bash
conda create -n timezero python=3.11
conda env create -f environment.yml
conda activate timezero
```

## Training

TimeZero training involves the following steps:

1.  **Data Preprocessing:**

    Download the dataset [Charades-STA](https://github.com/jiyanggao/TALL#charades-sta-anno-download), [Charades-v1](https://huggingface.co/datasets/HuggingFaceM4/charades), [ActivityNet](https://cs.stanford.edu/people/ranjaykrishna/densevid/)

    Before training, you need to preprocess the video data.

    ```bash
    bash preprocess_video.sh
    ```
    Specify the path to the Charades-STA dataset (video files, annotations, etc.).

3.  **GRPO Training:**

    ```bash
    cd scripts
    bash run_grpo_video.sh
    ```

    **`run_grpo_video.sh`**

    ```bash
    #!/bin/bash
    
    export DEBUG_MODE="false"  # Set to "true" for verbose logging during training.
    export LOG_PATH="./debug_log.txt"
    
    torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12361" \
    src/open_r1/grpo_video.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path mllm/Qwen2.5-VL-7B-Instruct \
    --preprocessed_data_path ./Charades_preprocessed_data_maxpix_3584 \
    --train_data_path ./Charades/charades_annotation/train.json \
    --eval_data_path ./Charades/charades_annotation/val.json \
    --video_folder ./Charades/Charades_v1 \
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
    --num_train_epochs 2 \
    --run_name $WANDB_NAME \
    --report_to wandb \
    --save_steps 50 \
    --save_only_model true
    ```

## Evaluation

After training, evaluate your model's performance:

```bash
bash scripts/evaluate.sh # Use evaluate.sh for evaluation.
```
**`evaluate.sh`**
```
python evaluate.py --model_base <path_to_your_trained_model> --dataset <charades or activitynet>
```

> The evaluation script (`evaluate.py`) needs to be implemented to load your model, process the test data, and calculate the relevant metrics (R1@0.3, R1@0.5, R1@0.7, etc.).

## Results

-   **Charades-STA (Finetuned)**

TimeZero outperforms previous state-of-the-art methods by a large margin. 

| Method                | Type | R1@0.3 | R1@0.5 | R1@0.7 |
| --------------------- | ---- | ------ | ------ | ------ |
| EaTR (VLP sota)       | VLP  | -      | 68.4   | 44.9   |
| TimeSuite (LVLM sota) | SFT  | 79.4   | 67.1   | 43.0   |
| TimeZero (ours)       | RL   | 83.3   | 72.5   | 47.9   |

-   **ActivityNet (Finetuned)**

TimeZero surpasses previous state-of-the-art LVLMs. 

| Method            | Type | R1@0.3 | R1@0.5 | R1@0.7 |
| ----------------- | ---- | ------ | ------ | ------ |
| EaTR (VLP sota)   | VLP  | -      | 58.18  | 37.64  |
| TRACE (LVLM sota) | SFT  | 54.0   | 37.7   | 24.0   |
| TimeZero (ours)   | RL   | 68.6   | 47.3   | 26.9   |

## Acknowledgements

We thank the authors of the following projects for their contributions:

*   [TRACE](https://github.com/gyxxyg/TRACE)
*    [R1-V](https://github.com/Deep-Agent/R1-V)
*   [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)

## Citation


```bibtex
@article{wang2025timezero,
  title={TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM},
  author={Wang, Ye and Xu, Boshen and Yue, Zihao and Xiao, Zihan and Wang, Ziheng and Zhang, Liang and Yang, Dingyi and Wang, Wenxuan and Jin, Qin},
  booktitle={arxiv},
  year={2025}
}
```
