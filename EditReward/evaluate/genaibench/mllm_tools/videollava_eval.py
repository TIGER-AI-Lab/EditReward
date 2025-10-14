"""
pip install transformers>=4.41.0
"""
import os
import torch
import numpy as np
from typing import List
from mantis.mllm_tools.mllm_utils import load_images
from mantis.models.conversation import conv_videollava as default_conv
import re
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, AutoConfig
from ..utils import process_video_into_frames, merge_video_into_frames
class VideoLlava():
    support_multi_image = True
    support_video_input = True
    merged_image_files = []
    def __init__(self, model_path:str='LanguageBind/Video-LLaVA-7B-hf') -> None:
        """Llava model wrapper

        Args:
            model_path (str): Video Llava model name
        """
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16).eval()
        self.processor = VideoLlavaProcessor.from_pretrained(model_path)
        self.image_token = "<image>"
        self.video_token = "<video>"
        self.conv_template = default_conv

        
    def __call__(self, inputs: List[dict]) -> str:
        """
        Args:
            inputs (List[dict]): [
                {
                    "type": "image",
                    "content": "https://example.com/test_image_1.jpg"
                },
                {
                    "type": "image",
                    "content": "https://example.com/test_image_2.jpg"
                },
                {
                    "type": "text",
                    "content": "What is difference between two images?"
                }
            ]
            Supports any form of interleaved format of image and text.
        """
        image_links = [x["content"] for x in inputs if x["type"] == "image"]
        images = load_images(image_links) if len(image_links) > 0 else None
        videos = [x["content"] for x in inputs if x["type"] == "video"]
        # video_frames = None if len(video) == 0 else merge_video_into_frames(video, max_num_frames=8)
        video_frames = None if len(videos) == 0 else [process_video_into_frames(video, max_num_frames=8) for video in videos]
        
        text_prompt = ""
        for i, message in enumerate(inputs):
            if message["type"] == "text":
                text_prompt += message["content"]
            elif message["type"] == "image":
                text_prompt += f"{self.image_token} \n"
            elif message["type"] == "video":
                text_prompt += f"{self.video_token} \n"
                        
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], text_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = self.processor(text=prompt, images=images, videos=video_frames, return_tensors="pt", padding=True)
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output_ids = self.model.generate(**inputs, do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True)
        input_ids = inputs["input_ids"]
        outputs = self.processor.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        return outputs       
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
    
if __name__ == "__main__":
    model = VideoLlava()
    # 0 shot
    zero_shot_exs = [
        {
            "type": "image",
            "content": "https://example.com/test_image_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    # 1 shot
    one_shot_exs = [
        {
            "type": "image",
            "content": "https://example.com/test_image_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A zebra."
        },
        {
            "type": "image",
            "content": "https://example.com/test_image_2.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    # 2 shot
    two_shot_exs = [
        {
            "type": "image",
            "content": "https://example.com/test_image_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A zebra."
        },
        {
            "type": "image",
            "content": "https://example.com/test_image_2.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A black cat."
        },
        {
            "type": "image",
            "content": "https://hips.hearstapps.com/hmg-prod/images/rabbit-breeds-american-white-1553635287.jpg?crop=0.976xw:0.651xh;0.0242xw,0.291xh&resize=980:*"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    print("### 0 shot")
    print(model(zero_shot_exs))
    print("### 1 shot")
    print(model(one_shot_exs))
    print("### 2 shot")
    print(model(two_shot_exs))
    """
    Output: a tiger and a zebra
    ### 0 shot
    The image features a zebra grazing on grass in a field.
    ### 1 shot
    A black cat.
    ### 2 shot
    A rabbit.
    """
    
    video_ex = [
        {
            "type": "video",
            "content": "/home/dongfu/WorkSpace/GenAI-Bench/genaibench/video_cache/e1e7018729ca400b9aac5ef966e1324e.mp4"
        },
        {
            "type": "text",
            "content": "What is in the video?"
        }
    ]
    print("### Video")
    print(model(video_ex))
    