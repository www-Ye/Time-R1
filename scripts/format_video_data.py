import argparse
import base64
import concurrent.futures
import io
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Dict, List

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm

from PIL import Image

import cv2


PROMPT_FORMAT = """I will provide you with an video, an original question, and its answer related to the video. Your task is to rewrite the question in such a way that answering it requires step-by-step Chain-of-Thought (CoT) reasoning with numerical or mathematical expressions where applicable. The reasoning process can include expressions like "let me think," "oh, I see," or other natural language thought expressions.

Please make sure your question is to ask for a certain answer with a certain value, do not ask for open-ended answer, and the answer is correct and easy to verify via simple protocol, like "2" or "A".

Please strictly do not include "Answer:" in the question part to avoid confusion and leakage.

Input Format:
Original Question: {original_question}
Original Answer: {original_answer}

Output Format:
Question: [rewrite the question if necessary]
Answer: [answer with reasoning steps, including calculations where applicable]
<think>step-by-step reasoning process</think>
<answer>easy to verify answer</answer>
"""

def process_video(video_path, seconds_per_frame=1, max_frames=15):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_time = total_frames / fps
    # frames_to_skip = int(fps * seconds_per_frame)
    frames_to_skip = int(total_frames / max_frames)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()
    
    print(len(base64Frames))
    if len(base64Frames)> max_frames+1:
        base64Frames = base64Frames[:max_frames+1]

    return base64Frames


def get_image_data_url(image_input):
    if isinstance(image_input, str) and image_input.startswith("data:"):
        return image_input

    if isinstance(image_input, str) and image_input.startswith("http"):
        image_input = load_image(image_input)

    if isinstance(image_input, str):
        image_input = Image.open(image_input)

    if not isinstance(image_input, Image.Image):
        raise ValueError("Unsupported image input type")

    if image_input.mode != "RGB":
        image_input = image_input.convert("RGB")

    buffer = BytesIO()
    image_input.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    base64_data = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"


def gpt4o_query(file_path, prompt, max_retries=1, initial_delay=3):
    if not os.path.exists(file_path):
        return None

    from openai import OpenAI
    client = OpenAI(
        base_url="https://api.ai-gaochao.cn/v1/",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    for attempt in range(max_retries):
        try:
            base64Frames = process_video(file_path, seconds_per_frame=0.5)
            # import pdb; pdb.set_trace()
            response = client.chat.completions.create(
                model="gemini-1.5-pro",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert to analyze the video and provide useful information for users.""",
                    },
                    {
                        "role": "user",
                        "content": [
                            "These are frames taken from the video",
                            *map(
                                lambda x: {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpg;base64,{x}",
                                        "detail": "low",
                                    },
                                },
                                base64Frames,
                            ),
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=8192,
            )
            # print(response.choices[0].message.content)
            return response.choices[0].message.content

        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(
                    f"Failed after {max_retries} attempts. Last error: {str(e)}"
                )
            delay = initial_delay * (2**attempt) + random.uniform(
                0, 0.1 * initial_delay * (2**attempt)
            )
            time.sleep(delay)


def process_single_item(example):
    try:
        video_path = example["video_path"]
        formatted_prompt = PROMPT_FORMAT.format(
            original_question=example["question"], original_answer=example["answer"]
        )

        response = gpt4o_query(video_path, formatted_prompt)
        example["gpt4o_response"] = response
        return example
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        example["gpt4o_response"] = None
        return example


def main():
    # dataset_path = "path/to/your/dataset"
    # full_dataset = load_from_disk(dataset_path)

    # processed_dataset = full_dataset.map(
    #     function=partial(process_single_item),
    #     num_proc=256,
    #     desc="Processing dataset with GPT-4o",
    #     keep_in_memory=True,
    # )

    # output_path = f"{dataset_path}_processed"
    # processed_dataset.save_to_disk(output_path)
    # print(f"Processed dataset saved to: {output_path}")
    
    video_root = "/share/wy/Video/R1-V-main/src/open-r1-multimodal/data/llava_video/LLaVA-Video-large-swift"
    jsonl_file = "/share/wy/Video/R1-V-main/src/open-r1-multimodal/data/llava_video/LLaVA-Video-large-swift/train.jsonl"

    jsonl_data = []

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            jsonl_data.append(data)
    
    count = 0
    
    answers_file = os.path.join("/share/wy/Video/R1-V-main/src/open-r1-multimodal/data", f"LLaVA-Video-large-swift-origin.jsonl")
    ans_file = open(answers_file, "w")
    
    for item in tqdm(jsonl_data):

        sample_set = {}
        video_ = item["videos"]
        # sample_set['id'] = item["id"]
        
        # first QA
        question = item["query"]
        question = question.replace("<image>", "").replace("<video>", "")

        answer = item["response"]
        
        # sample_set["problem"] = question
        # sample_set["solution"] = "<answer>"+answer[:2]+"</answer>" # A.
        
        video_path = os.path.join(video_root, video_)
        
        if not os.path.exists(video_path):
            continue
        
        count = count + 1
    
        # example = {
        #     "video_path": video_path,
        #     "question": question,
        #     "answer": answer,
        # }
        # new_example = example.copy()
        
        # new_example.pop("video_path")
        # new_example.pop("question")
        # new_example.pop("answer")
        
        new_example = {}
        
        new_example["problem"] = question.split("Please")[0]
        new_example["solution"] = "<answer>"+answer[:2]+"</answer>" # A.
        new_example["video"] = video_path
        
        new_example["original_question"] = question
        new_example["original_answer"] = answer
        # gpt4o_response
        
        ans_file.write(json.dumps(new_example, ensure_ascii=False) + "\n")
        ans_file.flush()

if __name__ == "__main__":
    main()