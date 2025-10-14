"""GPT-5-mini evaluation module with custom API configuration
"""
import os
import torch
import time
import torch.nn as nn
from typing import List, Union, Optional, Dict
from .model_utils.openai_utils import openai_completions, local_image_to_data_url


class GPT5Mini():
    support_multi_image = True
    merged_image_files = []
    
    def __init__(self, model_path: str = "gpt-5-mini") -> None:
        """GPT-5-mini model wrapper with custom API configuration

        Args:
            model_path (str): model name
        """
        self.model_path = model_path
        # Set custom API configuration for GPT-5-mini
        self.base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Set environment variables for the API configuration
        os.environ["OPENAI_API_BASE"] = self.base_url
        os.environ["OPENAI_API_KEY"] = self.api_key

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
        if self.support_multi_image:
            from PIL import Image  # 只为类型判断
            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant that helps people find information."
                        }
                    ]
                }
            ]
            messages += [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            for message in inputs:
                if message["type"] == "image":
                    # Convert image to proper data URL format for GPT-5-mini API
                    image_data_url = local_image_to_data_url(message["content"])
                    messages[-1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    })
                elif message["type"] == "text":
                    messages[-1]["content"].append({
                        "type": "text",
                        "text": message["content"]
                    })
                else:
                    raise NotImplementedError

            results = openai_completions(
                [messages],
                model_name=self.model_path,
                temperature=0.0,
                top_p=1.0,
                max_tokens=1024,  # 降低上限，避免响应太长占内存
                openai_api_base=self.base_url,
                openai_api_keys=[self.api_key],
            )
            return results['completions'][0]
        else:
            raise NotImplementedError

    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)


if __name__ == "__main__":
    model = GPT5Mini()
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
            "content": "https://example.com/test_image_3.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    # difference
    difference_exs = [
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
        },
    ]
    print("### 0 shot")
    print(model(zero_shot_exs))
    print("### 1 shot")
    print(model(one_shot_exs))
    print("### 2 shot")
    print(model(two_shot_exs))
    print("### difference")
    print(model(difference_exs))
