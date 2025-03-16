from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import re
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")

    return parser.parse_args()


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union


def inference(video_path, prompt, model, processor, max_new_tokens=2048): # Modified inference function

    messages = [
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, 
                "total_pixels": 3584 * 28 * 28, 
                "min_pixels": 16 * 28 * 28,
                },
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def parse_timestamp_output(output_string):
    """
    解析模型输出的时间戳字符串，提取开始时间和结束时间 (秒)。

    Args:
        output_string: 模型输出的时间戳字符串，例如 "5.2 to 12.7"。

    Returns:
        一个元组 (start_time, end_time)，单位为秒。
        如果解析失败，返回 None。
    """
    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", output_string) # 使用 re.findall 查找所有匹配项
    if not matches:
        return None, None

    # print(matches)
    last_match = matches[-1]
    start_time_str = last_match[0]
    end_time_str = last_match[2]
    
    start_time = float(start_time_str)
    end_time = float(end_time_str)
    
    return start_time, end_time

GROUND_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

# GROUND_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

# Provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

def eval(data, model, processor, video_dir_path):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    
    # print(len(data.items()))
    pbar = tqdm(data.items())
    for vid, ann in pbar:
        # print(vid)
        query_json = []
        for i in range(len(ann['sentences'])):
            # query_json.append({'descriptions': [ann['sentences'][sent_i]]})
            prompt = GROUND_TEMPLATE.replace('[EVENT]', ann['sentences'][i])

            duration = ann['duration'] if 'duration' in ann else ann['video_duration']
            for ext in ['mp4', 'mkv', 'webm']:
                video_path = os.path.join(video_dir_path, f"{vid}.{ext}")
                if os.path.isfile(video_path):

                    ans = inference(video_path, prompt, model, processor)
                    sp, ep = parse_timestamp_output(ans)
                    print('ans', ans)
                    print("pred", sp, ep)
                    print("gt", ann['timestamps'][i])
                    if (sp is not None) and (ep is not None):
                        # for i in range(len(ans)):
                        s, e = ann['timestamps'][i]
                        # s, e = s + pad_sec, e + pad_sec

                        # sp, ep = ans[i]['response'][0]['start'], ans[i]['response'][0]['end']
                        iou_ = (min(e, ep) - max(s, sp)) / (max(e, ep) - min(s, sp))
                        ious.append(max(iou_, 0))
                        recall += thresh <= iou_
                    else:
                        ious.append(0)
                        # recall += thresh <= iou_

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_base,
        torch_dtype=torch.bfloat16,
        use_sliding_window=True,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_base)
    model = model.cuda()

    print('Evaluating', args.dataset, args.split)

    with open(dataset['splits'][args.split]['annotation_file']) as f:
        data = json.load(f)
    eval(data, model, processor, dataset['video_path'])
        
