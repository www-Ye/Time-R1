# TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM

---

### Updates

- 2025-03-16: TimeZero initial release! Code and evaluation scripts are now available.
- 2025-03-16: TimeZero achieves SOTA performance on Charades-STA!

### Overview

TimeZero is a reasoning-guided Large Vision-Language Model (LVLM) designed for Temporal Video Grounding (TVG). It excels at identifying specific segments within lengthy videos that correspond to a given natural language query.  TimeZero achieves this entirely through a reinforcement learning approach that allows the model to reason about video-language relationships *during inference*.

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
conda activate timezero
```

## Training

TimeZero training involves the following steps:

1.  **Data Preprocessing:**
    Before training, you need to preprocess the video data.

    ```bash
    python pre_process_video.py
    ```
    Specify the path to the Charades-STA dataset (video files, annotations, etc.).  See the `dataset/` directory for an example structure you should follow.

2.  **GRPO Training:**

    ```bash
    cd scripts
    bash run_grpo_video.sh
    ```

    **`run_grpo_video.sh` (Create this file - Example Content)**

    ```bash
    #!/bin/bash

    export DEBUG_MODE="false"  # Set to "true" for verbose logging during training.
    export LOG_PATH="./debug_log.txt"

    torchrun --nproc_per_node="8" \
        --nnodes="1" \
        --node_rank="0" \
        --master_addr="127.0.0.1" \
        --master_port="12345" \
        ../src/open_r1/grpo.py \  # Adjust this path if your GRPO script is located elsewhere
        --output_dir <OUTPUT_DIR> \
        --model_name_or_path <PATH-TO-YOUR-BASE-MODEL> \  # e.g., Qwen2-VL-Instruct
        --dataset_name charades_sta \ # Use your dataset name within data_configs.
        --deepspeed zero3_offload.json \ # Path to the deepspeed config file.
        --max_prompt_length 512 \
        --max_completion_length 64 \ # Adjust as needed.
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --logging_steps 1 \
        --bf16 \
        --report_to wandb \
        --gradient_checkpointing false \
        --attn_implementation flash_attention_2 \  # If supported by your base model
        --num_train_epochs 2 \
        --run_name TimeZero-Charades-STA \
        --save_steps 100 \
        --save_only_model true \
        --num_generations 8   # Number of outputs to generate during RL.  Reduce for faster training/less memory.
    ```
> [!NOTE]

## Evaluation

After training, evaluate your model's performance:

```bash
cd scripts
bash evaluate.sh # Use evaluate.sh for evaluation.
```
**`evaluate.sh`**
```
python ../evaluate.py --model_path <path_to_your_trained_model> --dataset_path <path_to_charades_sta_test_data>
```

> [!NOTE] The evaluation script (`evaluate.py`) needs to be implemented to load your model, process the test data, and calculate the relevant metrics (R1@0.3, R1@0.5, R1@0.7, etc.).

## Results

**Charades-STA (Finetuned)**

![image]

**ActivityNet**

![image]

## Acknowledgements

We thank the authors of the following projects for their contributions:

*   [TRACE]
*    R1-V
*   Qwen2.5-VL

## Citation

```bibtex
@inproceedings{wang2025timezero,
  title={TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM},
  author={Wang, Ye and Xu, Boshen and Yue, Zihao and Xiao, Zihan and Wang, Ziheng and Zhang, Liang and Yang, Dingyi and Wang, Wenxuan and Jin, Qin},
  booktitle={arxiv},
  year={2025}
}
```
